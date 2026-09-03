"""
skill_acquisition.py
=====================
v2 of Dynamic Skill Selection: live, runtime skill acquisition.

See skill_implementation_plan.md for the full design. Everything here
routes through one function:

    SkillAcquisitionManager.ensure_skill(capability_description, task_description) -> AcquisitionResult

called from exactly two places (both wired up as of Phase C/D --
see dynamic_langgraph_backend.py):
  A. Config-time, inside DynamicAgentFactory.create_agent()/refresh_tools(),
     when the selection LLM's JSON reports a "skill_gap" (via
     DynamicAgentFactory._resolve_skill_gap).
  B. Execution-time, from the always-available `request_skill_acquisition`
     tool the agent can call mid-task (special-cased in
     DynamicAgentManager._tools_node, same as update_tasks).

Current phase: **Phase B -- real `skills find` / `skills add` integration.**
`_search()` tries `npx skills find "<query>"` first, falling back to
`uvx skillsmd find "<query>"` if `npx` isn't on PATH (plan §4.2's open
item -- both have an identical output shape, so one parser covers both).
`_install()` shells out to `npx skills add <owner/repo> --skill <name> -y
--agent generic` and then, because the CLI's exact output-directory
behavior wasn't confirmed ahead of time (plan §4.4's open item), searches
a short list of plausible landing spots and copies whatever it finds into
`github_skills/<skill_name>/`; if nothing turns up there, it falls back to
the scoped download API (`https://skills.sh/api/download/{owner}/{repo}/{skill}`)
and writes the response directly -- this also sidesteps the
whole-monorepo-download failure mode `_verify`'s size/count check exists
to catch anyway.

Phase A's hand-placed "staging" folder path is kept, not removed --
`Candidate.owner_repo` values of the form `"local/<name>"` (the shape
Phase A's own `_search` stub produces) still resolve through the simple
copy-from-staging path in `_install`. This means `TRUSTED_OWNER_ALLOWLIST`
having `"local"` in it still exercises the exact same isolated test setup
described in Phase A's original docstring below, unchanged, alongside the
real CLI path -- useful for hand-testing `_verify`/`index_single`/caching/
locking without needing network access or the CLI installed.

To exercise the Phase A staging path by hand: create
`skill_acquisition_staging/<skill_name>/SKILL.md` (same frontmatter shape
skill_discovery.py already parses) and call `ensure_skill("<capability
phrase containing skill_name>", "<any task description>")` -- `_search`
will find it (as a "local/<name>" candidate, alongside any real search
hits), `_install` will copy it into `github_skills/`, `_verify` will
check it, and `index_single` will register it for real.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from logging_utils import _log
from skills import SkillRegistry
from skill_discovery import SkillDiscovery

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover -- mirrors optional-dep handling elsewhere in this repo
    ChatOpenAI = None

try:
    import requests
except ImportError:  # pragma: no cover -- same optional-dep handling as ChatOpenAI above
    requests = None


# ---------------------------------------------------------------------------
# Config (mirrors the style of the constants block at the top of
# dynamic_langgraph_backend.py -- one place to see every tunable)
# ---------------------------------------------------------------------------
SEARCH_TIMEOUT_SECONDS = 30          # real value once Phase B shells out to `skills find`
INSTALL_TIMEOUT_SECONDS = 60         # real value once Phase B shells out to `skills add`
INSTALL_COUNT_FLOOR = 100            # hard-gate floor from the plan's §4.3 vetting rule
TRUSTED_OWNER_ALLOWLIST = {"vercel-labs", "anthropics", "local"}  # "local" = Phase A staging stand-in
NEGATIVE_CACHE_TTL_SECONDS = 600     # 10 minutes, per plan §4.16
MAX_PACKAGE_FILES = 200              # oversized/scope-creep fetch guard, plan §4.8
MAX_PACKAGE_BYTES = 20 * 1024 * 1024  # 20MB, same guard
MAX_INSTALL_ATTEMPTS = 2             # try up to 2 ranked candidates before giving up, plan §4.8
GITHUB_SKILLS_ROOT = Path("github_skills")            # matches skill_discovery.SKILL_ROOTS["github"]
STAGING_ROOT = Path("skill_acquisition_staging")       # Phase A dev/test path -- see module docstring

# Phase B config -- real `skills find`/`skills add` integration (plan §4.2/§4.4)
SKILLS_FIND_CMD_CANDIDATES = (
    ["npx", "skills", "find"],
    ["uvx", "skillsmd", "find"],
)
SKILLS_ADD_CMD_CANDIDATES = (
    ["npx", "skills", "add"],
    ["uvx", "skillsmd", "add"],
)
SKILLS_DOWNLOAD_API_TEMPLATE = "https://skills.sh/api/download/{owner}/{repo}/{skill}"
DOWNLOAD_TIMEOUT_SECONDS = 30
# Candidate.installs sentinel meaning "we couldn't parse an install count
# out of the CLI's text output" -- treated as below-floor (fails the hard
# gate) unless the owner is on the trusted allowlist, rather than silently
# defaulting to something that would pass the gate by accident.
UNKNOWN_INSTALL_COUNT = -1
# `skills find` output is one block per candidate, roughly:
#   owner/repo@skill_name
#   1,234 installs
#   https://github.com/owner/repo
# Line order/exact wording isn't guaranteed across CLI versions, so this
# scans for the pieces independently within each blank-line-delimited
# block rather than assuming a fixed line order.
_CANDIDATE_HEADER_RE = re.compile(r"^([\w.-]+/[\w.-]+)@([\w.-]+)\s*$")
_INSTALL_COUNT_RE = re.compile(r"([\d,]+)\s*install", re.IGNORECASE)
_URL_RE = re.compile(r"https?://\S+")


@dataclass
class Candidate:
    owner_repo: str      # e.g. "vercel-labs/skills" ("local/<name>" for Phase A staged candidates)
    skill_name: str
    installs: int
    url: str


@dataclass
class AcquisitionResult:
    status: str  # "already_present" | "installed" | "not_found" | "failed"
    skill_name: str | None
    reason: str


class SkillAcquisitionManager:
    """Discovers, installs, verifies, and registers skills live, at task
    execution time. One instance is shared across the whole process (like
    DynamicToolRegistry/SkillRegistry), so its lock table and cache are
    process-wide -- see `_get_lock`'s docstring for the multi-process
    caveat the plan explicitly defers (§4.14)."""

    def __init__(
        self,
        skill_registry: SkillRegistry,
        skill_discovery: SkillDiscovery,
        github_skills_root: Path = GITHUB_SKILLS_ROOT,
        staging_root: Path = STAGING_ROOT,
        llm=None,
    ):
        self.skill_registry = skill_registry
        self.skill_discovery = skill_discovery
        self.github_skills_root = Path(github_skills_root)
        self.staging_root = Path(staging_root)
        # Injectable for tests; defaults to the same model family the rest
        # of the backend uses. Only invoked when _rank_and_gate has more
        # than one candidate to choose between (§4.3) -- Phase A's stub
        # _search rarely produces that, so this stays cold most of the time.
        self.llm = llm if llm is not None else (ChatOpenAI(model="openai.gpt-oss-120b") if ChatOpenAI else None)

        self._positive_cache: dict[str, str] = {}       # capability key -> resolved skill_name
        self._negative_cache: dict[str, float] = {}      # capability key -> failure timestamp
        self._install_locks: dict[str, threading.Lock] = {}
        self._locks_meta_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def ensure_skill(self, capability_description: str, task_description: str) -> AcquisitionResult:
        """Synchronous, idempotent, safe to call repeatedly and concurrently.
        Never raises -- every failure mode funnels into a `failed`/`not_found`
        AcquisitionResult so callers (create_agent/refresh_tools/the
        request_skill_acquisition tool handler) always have a value to act
        on (plan §4.8)."""
        key = self._normalize(capability_description)

        cached = self._check_cache_and_registry(key)
        if cached is not None:
            _log(
                "SKILL-ACQUISITION", "Resolved from cache/registry, no acquisition needed",
                capability=capability_description, status=cached.status, phase="ready",
            )
            return cached

        lock = self._get_lock(key)
        with lock:
            # Re-check now that we hold the lock: another thread may have
            # just finished installing the exact same capability while we
            # were waiting (plan §4.14/§4.15 -- race and duplicate-request
            # handling are the same mechanism).
            cached = self._check_cache_and_registry(key)
            if cached is not None:
                _log(
                    "SKILL-ACQUISITION", "Resolved from cache/registry after acquiring lock",
                    capability=capability_description, status=cached.status, phase="ready",
                )
                return cached

            return self._acquire_locked(key, capability_description, task_description)

    def _acquire_locked(self, key: str, capability_description: str, task_description: str) -> AcquisitionResult:
        """Everything from here down assumes the per-key lock is already held."""
        _log("SKILL-ACQUISITION", "Searching for skill", capability=capability_description, phase="searching")
        try:
            candidates = self._search(capability_description)
        except Exception as e:
            _log("ERROR", "Skill search failed", capability=capability_description, error=str(e), phase="failed")
            candidates = []

        gated = self._rank_and_gate(candidates, task_description)
        if not gated:
            self._negative_cache[key] = time.time()
            _log(
                "SKILL-ACQUISITION", "No candidate passed the quality gate; nothing installed",
                capability=capability_description, candidates_found=len(candidates), phase="failed",
            )
            return AcquisitionResult(status="not_found", skill_name=None, reason="no candidate passed the quality gate")

        last_reason = "no candidates attempted"
        for candidate in gated[:MAX_INSTALL_ATTEMPTS]:
            _log(
                "SKILL-ACQUISITION", "Candidate selected, installing",
                capability=capability_description, candidate=candidate.skill_name, source=candidate.owner_repo, phase="candidate_selected",
            )
            result = self._install_and_verify_one(candidate)
            if result.status == "installed":
                self._positive_cache[key] = result.skill_name
                return result
            last_reason = result.reason

        self._negative_cache[key] = time.time()
        _log(
            "SKILL-ACQUISITION", "All candidates failed; giving up",
            capability=capability_description, last_reason=last_reason, phase="failed",
        )
        return AcquisitionResult(status="failed", skill_name=None, reason=last_reason)

    def _install_and_verify_one(self, candidate: Candidate) -> AcquisitionResult:
        """Install one candidate, verify it, and index it on success.
        Returns a `failed` AcquisitionResult (never raises) so the caller's
        retry loop can just move on to the next candidate."""
        _log("SKILL-ACQUISITION", "Installing skill", candidate=candidate.skill_name, phase="installing")
        try:
            install_path = self._install(candidate)
        except subprocess.TimeoutExpired:
            _log("ERROR", "Skill install timed out", candidate=candidate.skill_name, phase="failed")
            return AcquisitionResult(status="failed", skill_name=None, reason="install_timeout")
        except Exception as e:
            _log("ERROR", "Skill install failed", candidate=candidate.skill_name, error=str(e), phase="failed")
            return AcquisitionResult(status="failed", skill_name=None, reason=f"install_error: {e}")

        _log("SKILL-ACQUISITION", "Verifying installed package", candidate=candidate.skill_name, phase="verifying")
        ok, verify_reason = self._verify(install_path)
        if not ok:
            self._quarantine(install_path)
            _log(
                "WARNING", "Skill package failed verification; quarantined",
                candidate=candidate.skill_name, reason=verify_reason, phase="failed",
            )
            return AcquisitionResult(status="failed", skill_name=None, reason=verify_reason)

        _log("SKILL-ACQUISITION", "Indexing verified skill", candidate=candidate.skill_name, phase="indexing")
        skill = self.skill_discovery.index_single(install_path, source="github")
        if skill is None:
            self._quarantine(install_path)
            _log(
                "WARNING", "Verified package could not be indexed; quarantined",
                candidate=candidate.skill_name, phase="failed",
            )
            return AcquisitionResult(status="failed", skill_name=None, reason="incomplete_package: index_single could not parse it")

        _log("SKILL-ACQUISITION", "Skill ready", skill=skill.name, source=candidate.owner_repo, phase="ready")
        return AcquisitionResult(status="installed", skill_name=skill.name, reason=f"installed from {candidate.owner_repo}")

    # ------------------------------------------------------------------
    # §4.16 -- cache + registry short-circuit
    # ------------------------------------------------------------------

    def _check_cache_and_registry(self, key: str) -> AcquisitionResult | None:
        """Nearly-free check run before touching the lock or doing any
        search/install work at all (plan §4.16). Returns None if nothing
        short-circuits and the full pipeline needs to run."""
        cached_name = self._positive_cache.get(key)
        if cached_name is not None:
            if self.skill_registry.get_skill(cached_name) is not None:
                return AcquisitionResult(status="already_present", skill_name=cached_name, reason="positive cache hit")
            # Stale entry -- the skill was removed from the registry since
            # we cached it (e.g. manual removal, or a higher-precedence
            # skill of the same name took over). Drop it and fall through
            # to a real search instead of trusting stale state.
            del self._positive_cache[key]

        failed_at = self._negative_cache.get(key)
        if failed_at is not None:
            if time.time() - failed_at < NEGATIVE_CACHE_TTL_SECONDS:
                return AcquisitionResult(status="not_found", skill_name=None, reason="cached negative result (within TTL)")
            del self._negative_cache[key]

        return None

    # ------------------------------------------------------------------
    # §4.2/§4.3 -- discovery + ranking/gating
    # ------------------------------------------------------------------

    def _search(self, capability_description: str) -> list[Candidate]:
        """Real search (Phase B, plan §4.2): shells out to `npx skills find
        "<query>"`, falling back to `uvx skillsmd find` if `npx` isn't on
        PATH (identical output shape either way, so `_parse_find_output`
        covers both -- confirm at deploy time which one is actually
        installed in the runtime image; this tries both rather than
        guessing). Combined with the Phase A staging-folder scan so hand-
        placed dev/test candidates (plan's `STAGING_ROOT`) still surface
        alongside real hits -- see module docstring for why that path is
        kept rather than removed."""
        candidates: list[Candidate] = list(self._search_staging(capability_description))

        for cmd_prefix in SKILLS_FIND_CMD_CANDIDATES:
            try:
                proc = subprocess.run(
                    [*cmd_prefix, capability_description],
                    capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=SEARCH_TIMEOUT_SECONDS,
                )
            except FileNotFoundError:
                continue  # this CLI isn't installed in this runtime; try the next one
            except subprocess.TimeoutExpired:
                _log("WARNING", "skills find timed out", cmd=cmd_prefix[0], capability=capability_description)
                return candidates
            if proc.returncode != 0 and not proc.stdout.strip():
                _log(
                    "WARNING", "skills find exited non-zero with no output",
                    cmd=cmd_prefix[0], exit_code=proc.returncode, stderr=proc.stderr[:300],
                )
                continue
            candidates.extend(self._parse_find_output(proc.stdout))
            break  # first CLI that actually ran wins; don't double-search with both
        else:
            _log("WARNING", "No skills-search CLI (npx/uvx) available on PATH; only staging candidates considered", capability=capability_description)

        return candidates

    def _search_staging(self, capability_description: str) -> list[Candidate]:
        """Phase A dev/test path: any hand-placed folder under
        `staging_root` whose name loosely matches the capability
        description becomes a `"local/<name>"` candidate. `installs` is
        set to exactly the trust floor so it clears the hard gate by
        construction (staged candidates are developer-placed, i.e.
        implicitly trusted for testing purposes)."""
        if not self.staging_root.is_dir():
            return []
        slug = self._slugify(capability_description)
        candidates: list[Candidate] = []
        for entry in sorted(p for p in self.staging_root.iterdir() if p.is_dir()):
            entry_slug = self._slugify(entry.name)
            if slug in entry_slug or entry_slug in slug:
                candidates.append(
                    Candidate(
                        owner_repo=f"local/{entry.name}",
                        skill_name=entry.name,
                        installs=INSTALL_COUNT_FLOOR,
                        url=f"file://{entry.resolve()}",
                    )
                )
        return candidates

    @staticmethod
    def _parse_find_output(stdout: str) -> list[Candidate]:
        """Parse `skills find`/`skillsmd find`'s plain-text stdout into
        Candidates. One candidate per blank-line-delimited block; within a
        block, the `owner/repo@skill_name` header line, an "N installs"
        line, and a URL line can appear in any order (plan §4.2 -- exact
        formatting isn't a stable contract across CLI versions), so this
        scans each block's lines independently rather than assuming a
        fixed layout. Blocks whose header line doesn't parse are skipped
        rather than raising -- a partially-understood CLI output is
        treated as "no candidate found here", not a hard failure."""
        candidates: list[Candidate] = []
        for block in re.split(r"\n\s*\n", stdout.strip()):
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            owner_repo = skill_name = None
            for line in lines:
                m = _CANDIDATE_HEADER_RE.match(line)
                if m:
                    owner_repo, skill_name = m.group(1), m.group(2)
                    break
            if not owner_repo or not skill_name:
                continue

            installs = UNKNOWN_INSTALL_COUNT
            url = ""
            for line in lines:
                if installs == UNKNOWN_INSTALL_COUNT:
                    count_m = _INSTALL_COUNT_RE.search(line)
                    if count_m:
                        try:
                            installs = int(count_m.group(1).replace(",", ""))
                        except ValueError:
                            pass
                if not url:
                    url_m = _URL_RE.search(line)
                    if url_m:
                        url = url_m.group(0)

            candidates.append(Candidate(owner_repo=owner_repo, skill_name=skill_name, installs=installs, url=url))
        return candidates

    def _rank_and_gate(self, candidates: list[Candidate], task_description: str) -> list[Candidate]:
        """Two-stage filter from plan §4.3:
        1. Hard gate (pure Python): drop anything below the install-count
           floor unless its owner is on the trusted allowlist.
        2. Selection: if more than one candidate survives, ask the LLM to
           pick the single best match (or "none"); skip the LLM call
           entirely if 0 or 1 candidates survive the gate.

        Returns a ranked list (best first) so the caller can try a second
        candidate if the first fails install/verification -- not just a
        single winner.
        """
        gated = [
            c for c in candidates
            if c.installs >= INSTALL_COUNT_FLOOR or self._owner(c.owner_repo) in TRUSTED_OWNER_ALLOWLIST
        ]
        if not gated:
            return []
        gated.sort(key=lambda c: c.installs, reverse=True)
        if len(gated) == 1 or self.llm is None:
            return gated

        chosen_name = self._llm_select(gated[:5], task_description)
        if chosen_name is None:
            # LLM explicitly said "none of these fit" -- don't fall back to
            # guessing; an unwanted install is worse than a clean not_found.
            return []
        reordered = [c for c in gated if c.skill_name == chosen_name]
        reordered += [c for c in gated if c.skill_name != chosen_name]
        return reordered or gated

    def _llm_select(self, candidates: list[Candidate], task_description: str) -> str | None:
        """One small LLM call, same pattern as agent_creation_llm in
        dynamic_langgraph_backend.py. Returns the chosen skill_name, or
        None if the LLM picked "none" or its reply couldn't be parsed
        (fail closed -- an ungated guess is worse than no install)."""
        import json as _json

        options = [
            {"skill_name": c.skill_name, "source": c.owner_repo, "installs": c.installs, "url": c.url}
            for c in candidates
        ]
        prompt = f"""
A task needs this capability: "{task_description}"

Candidate skills found (pick the single best match, or "none" if none
genuinely fit):
{_json.dumps(options, indent=2)}

Respond with ONLY JSON: {{"choice": "skill_name" or "none"}}
"""
        try:
            response = self.llm.invoke(
                prompt,
                config={"run_name": "skill_candidate_selection_llm", "tags": ["skill_acquisition"]},
            )
            text = getattr(response, "content", "") or ""
            if isinstance(text, list):  # multimodal content shape, just in case
                text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
            import re as _re

            match = _re.search(r"\{.*\}", text, _re.DOTALL)
            parsed = _json.loads(match.group(0)) if match else {}
        except Exception as e:
            _log("WARNING", "Skill candidate selection LLM call failed; treating as no match", error=str(e))
            return None

        choice = (parsed.get("choice") or "").strip()
        if not choice or choice.lower() == "none":
            return None
        valid_names = {c.skill_name for c in candidates}
        if choice not in valid_names:
            _log("WARNING", "Skill candidate selection returned an unrecognized name; ignoring", choice=choice)
            return None
        return choice

    # ------------------------------------------------------------------
    # §4.4 -- installation
    # ------------------------------------------------------------------

    def _install(self, candidate: Candidate) -> Path:
        """Install one candidate into `github_skills/<skill_name>/` and
        return that path. Dispatches to one of two paths:

        - `candidate.owner_repo` starting with `"local/"` (Phase A staging
          candidates, see module docstring): copy straight from
          `staging_root`, unchanged from Phase A.
        - Everything else (real search hits): real install (Phase B, plan
          §4.4) via `_install_from_cli`, falling back to
          `_install_from_download_api` if the CLI ran but its output
          couldn't be located on disk in any of the plausible spots it
          might have landed in (the open item plan §4.4 flagged -- the
          CLI's documented flags target `./<agent>/skills/`, with no
          confirmed way to point it at an arbitrary directory).

        Raises on failure (missing staging folder, CLI + download API both
        failing) so the caller's existing exception handling in
        `_install_and_verify_one` covers this path the same way it always
        has -- nothing about the calling contract changes between Phase A
        and Phase B.
        """
        if candidate.owner_repo.startswith("local/"):
            _, _, staged_name = candidate.owner_repo.partition("/")
            source_dir = self.staging_root / staged_name
            if not source_dir.is_dir():
                raise FileNotFoundError(f"staged skill folder not found: {source_dir}")
            dest_dir = self._fresh_dest_dir(candidate.skill_name)
            shutil.copytree(source_dir, dest_dir)
            return dest_dir

        owner, _, repo = candidate.owner_repo.partition("/")
        found = self._install_from_cli(candidate)
        if found is not None:
            return found
        _log(
            "SKILL-ACQUISITION", "CLI install did not land where expected; falling back to download API",
            candidate=candidate.skill_name, source=candidate.owner_repo, phase="installing",
        )
        return self._install_from_download_api(owner, repo, candidate.skill_name)

    def _fresh_dest_dir(self, skill_name: str) -> Path:
        """`github_skills/<skill_name>/`, cleared first if something's
        already physically there (most likely a leftover from an earlier
        failed run -- we only reach `_install` after the cache/registry
        short-circuit in `_check_cache_and_registry` already missed)."""
        self.github_skills_root.mkdir(parents=True, exist_ok=True)
        dest_dir = self.github_skills_root / skill_name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        return dest_dir

    def _install_from_cli(self, candidate: Candidate) -> Path | None:
        """Shell out to `npx skills add <owner/repo> --skill <name> -y
        --agent generic` (or the `uvx skillsmd add` equivalent). Since the
        CLI's exact output directory isn't guaranteed (plan §4.4), this
        then checks a short list of plausible landing spots relative to
        the current working directory and, if one has a matching
        `<skill_name>/SKILL.md`, moves it into `github_skills/`. Returns
        None (not an exception) if the subprocess ran but nothing was
        found where expected -- that's a normal, anticipated outcome that
        `_install` falls back from, not a failure in itself. Raises
        subprocess.TimeoutExpired (propagated as-is, same as before) so
        `_install_and_verify_one`'s existing timeout handling still
        applies unchanged."""
        for cmd_prefix in SKILLS_ADD_CMD_CANDIDATES:
            try:
                subprocess.run(
                    [*cmd_prefix, candidate.owner_repo, "--skill", candidate.skill_name, "-y", "--agent", "generic"],
                    capture_output=True, text=True, timeout=INSTALL_TIMEOUT_SECONDS,
                )
            except FileNotFoundError:
                continue  # this CLI isn't installed in this runtime; try the next one
            # Not checking returncode here on purpose -- plan §4.7: a 0
            # exit doesn't guarantee a well-formed package and a nonzero
            # exit with partial output is possible either way, so
            # completion is decided by what's actually on disk below, not
            # by the exit code.
            for landing_spot in (
                Path("generic") / "skills" / candidate.skill_name,
                Path.home() / "generic" / "skills" / candidate.skill_name,
                Path("skills") / candidate.skill_name,
                Path(candidate.skill_name),
            ):
                if (landing_spot / "SKILL.md").is_file():
                    dest_dir = self._fresh_dest_dir(candidate.skill_name)
                    shutil.copytree(landing_spot, dest_dir)
                    shutil.rmtree(landing_spot, ignore_errors=True)
                    return dest_dir
            return None  # CLI ran (or wasn't found at this landing spot); nothing usable found
        return None  # neither npx nor uvx was available at all

    def _install_from_download_api(self, owner: str, repo: str, skill_name: str) -> Path:
        """Fallback (plan §4.4(b)): hit the scoped download API directly
        instead of the CLI, and write the response into
        `github_skills/<skill_name>/` ourselves. This also sidesteps the
        whole-monorepo-download failure mode entirely, since the scoped
        endpoint returns exactly one skill's worth of content by
        construction rather than something `_verify`'s size/count check
        has to catch after the fact."""
        if requests is None:
            raise RuntimeError("requests is not installed; cannot use the skill download API fallback")
        url = SKILLS_DOWNLOAD_API_TEMPLATE.format(owner=owner, repo=repo, skill=skill_name)
        response = requests.get(url, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        response.raise_for_status()

        dest_dir = self._fresh_dest_dir(skill_name)
        dest_dir.mkdir(parents=True, exist_ok=True)
        content_type = response.headers.get("Content-Type", "")
        if "zip" in content_type or url.endswith(".zip"):
            import io
            import zipfile
            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                zf.extractall(dest_dir)
            # A zip that wraps everything in one top-level folder (common
            # for GitHub-style archives) would otherwise leave SKILL.md one
            # level too deep for _verify to find -- flatten that case.
            entries = list(dest_dir.iterdir())
            if len(entries) == 1 and entries[0].is_dir() and not (dest_dir / "SKILL.md").exists():
                inner = entries[0]
                for item in inner.iterdir():
                    shutil.move(str(item), str(dest_dir / item.name))
                inner.rmdir()
        else:
            # Assume the endpoint returned SKILL.md's raw content directly.
            (dest_dir / "SKILL.md").write_bytes(response.content)
        return dest_dir

    # ------------------------------------------------------------------
    # §4.9 -- verification
    # ------------------------------------------------------------------

    def _verify(self, path: Path) -> tuple[bool, str]:
        """Re-verify against the filesystem regardless of the install
        call's reported success (plan §4.7/§4.9) -- a 0 exit code, or a
        stub copy that "succeeded", doesn't guarantee a well-formed
        package. Only a package that passes this gets indexed."""
        if not path.is_dir():
            return False, "incomplete_package: install path is missing"

        skill_md = path / "SKILL.md"
        if not skill_md.is_file():
            return False, "incomplete_package: missing SKILL.md"

        try:
            skill = self.skill_discovery._parse_skill_md(skill_md, source="github")
        except (OSError, UnicodeDecodeError) as e:
            return False, f"incomplete_package: SKILL.md failed to parse ({e})"

        if not skill.name or not skill.description:
            return False, "incomplete_package: SKILL.md is missing required name/description frontmatter"

        file_count = 0
        total_bytes = 0
        for p in path.rglob("*"):
            if p.is_file():
                file_count += 1
                total_bytes += p.stat().st_size
        if file_count > MAX_PACKAGE_FILES or total_bytes > MAX_PACKAGE_BYTES:
            return False, (
                f"incomplete_package: oversized fetch ({file_count} files, {total_bytes} bytes) "
                f"-- likely grabbed more than one skill's worth of content"
            )

        return True, "ok"

    def _quarantine(self, path: Path) -> None:
        """Rename a failed install out of the way with a `.partial-<ts>`
        suffix rather than leaving it looking like a valid, cached install
        -- this is what keeps §4.16's disk-existence check from later
        mistaking it for a real one (plan §4.8)."""
        if not path.exists():
            return
        quarantined = path.with_name(f"{path.name}.partial-{int(time.time())}")
        try:
            path.rename(quarantined)
        except OSError as e:
            _log("WARNING", "Could not quarantine failed skill install", path=str(path), error=str(e))

    # ------------------------------------------------------------------
    # §4.14 -- locking
    # ------------------------------------------------------------------

    def _get_lock(self, key: str) -> threading.Lock:
        """Classic double-checked locking so two threads racing to create
        a lock for the same brand-new key don't end up with two different
        Lock objects. Process-local only -- scaling to multiple backend
        worker processes needs a filesystem `.installing` marker instead
        (plan §4.14 caveat, explicitly deferred, same posture as the
        already-flagged SQLite connection-leak issue)."""
        lock = self._install_locks.get(key)
        if lock is not None:
            return lock
        with self._locks_meta_lock:
            lock = self._install_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._install_locks[key] = lock
            return lock

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(capability_description: str) -> str:
        """Cache/lock key for a capability description. Deliberately
        light-touch (strip + lowercase + collapse whitespace) -- this is
        not meant to catch every paraphrase, just avoid re-triggering a
        full search for trivially-identical repeated phrasings (plan
        §4.16's stated goal)."""
        return " ".join((capability_description or "").strip().lower().split())

    @staticmethod
    def _slugify(text: str) -> str:
        normalized = (text or "").strip().lower()
        for sep in ("_", "-"):
            normalized = normalized.replace(sep, " ")
        return "-".join(normalized.split())

    @staticmethod
    def _owner(owner_repo: str) -> str:
        owner, _, _ = (owner_repo or "").partition("/")
        return owner
