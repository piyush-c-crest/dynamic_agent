"""
skill_acquisition.py
=====================
v2 of Dynamic Skill Selection: live, runtime skill acquisition.

See skill_implementation_plan.md for the full design. Everything here
routes through one function:

    SkillAcquisitionManager.ensure_skill(capability_description, task_description) -> AcquisitionResult

called from exactly two places (neither wired up yet -- that's Phase C/D):
  A. Config-time, inside DynamicAgentFactory.create_agent()/refresh_tools(),
     when the selection LLM's JSON reports a "skill_gap".
  B. Execution-time, from a new always-available `request_skill_acquisition`
     tool the agent can call mid-task.

Current phase: **Phase A -- manager skeleton + manual trigger.**
`_search()` and `_install()` are stubbed to operate on a hand-placed
"staging" folder instead of shelling out to `npx skills find`/`skills add`,
so the rest of the pipeline -- ranking/gating, installation, verification,
quarantine, `index_single`, positive/negative caching, and per-key locking
-- can be built and tested end-to-end in isolation first. Phase B swaps
just `_search`/`_install`'s bodies for real subprocess calls; nothing else
in this file should need to change when that happens (see each method's
docstring for exactly what Phase B replaces).

To exercise this phase by hand: create
`skill_acquisition_staging/<skill_name>/SKILL.md` (same frontmatter shape
skill_discovery.py already parses) and call `ensure_skill("<capability
phrase containing skill_name>", "<any task description>")` -- `_search`
will find it, `_install` will copy it into `github_skills/`, `_verify`
will check it, and `index_single` will register it for real.
"""

from __future__ import annotations

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
STAGING_ROOT = Path("skill_acquisition_staging")       # Phase A only -- gone once Phase B lands


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
        """PHASE A STUB. Real implementation (Phase B) shells out to
        `npx skills find "<query>"` (or `uvx skillsmd find` -- confirm
        which is available in the runtime image first, per plan §4.2) and
        parses its stdout into Candidates.

        For now: treat any hand-placed folder under `staging_root` whose
        name loosely matches the capability description as a single
        candidate, so the rest of the pipeline can be exercised without
        the real CLI. `installs` is set to exactly the trust floor so it
        clears the hard gate by construction (Phase A candidates are
        developer-staged, i.e. implicitly trusted for testing purposes)."""
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
        """PHASE A STUB. Real implementation (Phase B) shells out to
        `npx skills add <owner/repo> --skill <name> -y --agent generic`
        (spike the actual output directory first -- plan §4.4's open item
        -- before finalizing this) and returns wherever the CLI actually
        landed the files, or writes the scoped-download-API response
        directly if `skills add` can't be pointed at `github_skills/`.

        For now: copy the matching staged folder straight into
        `github_skills/<skill_name>/`, simulating what a real install
        would produce. Raises on a missing/malformed staging folder so
        the caller's normal exception handling covers this path too."""
        _, _, staged_name = candidate.owner_repo.partition("/")
        source_dir = self.staging_root / staged_name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"staged skill folder not found: {source_dir}")

        self.github_skills_root.mkdir(parents=True, exist_ok=True)
        dest_dir = self.github_skills_root / candidate.skill_name
        if dest_dir.exists():
            # Not yet registered (we only get here after the cache/registry
            # short-circuit missed) but something's physically there --
            # most likely a leftover from an earlier failed run. Clear it
            # rather than let copytree fail or silently merge into it.
            shutil.rmtree(dest_dir)
        shutil.copytree(source_dir, dest_dir)
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
