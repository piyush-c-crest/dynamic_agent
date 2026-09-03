"""
skill_discovery.py
===================
Phase 2 of Dynamic Skill Selection: automatic folder-based indexing.

Walks a fixed set of root directories, each one level deep, looking for
`<root>/<skill_name>/SKILL.md`. Each SKILL.md is a YAML-ish frontmatter
block followed by a markdown body:

    ---
    name: pdf_reports
    description: Generates PDF reports from tabular data
    tools: read_file, create_artifact
    triggers: pdf, report, generate document
    trust: trusted
    version: 1.0
    ---
    # Instructions
    When asked to produce a PDF report:
    1. Read the source data with `read_file`.
    2. ...

`instructions` (the markdown body) is what Phase 3 will splice into an
agent's system prompt when this skill is selected; `tools` are existing
tool names (from DynamicToolRegistry) the skill wants bound; `triggers`
are optional keywords Phase 3's selection prompt may surface as a hint.
A `bundled_tools` frontmatter field, if present, must be a JSON array of
`{"name": ..., "prompt": ...}` objects -- tools the skill needs that
aren't registered yet (Phase 3 routes these through the same sandboxed
`create_tool_from_prompt` validation path used for auto-created tools,
never executed directly).

Root directories and their `Skill.source` tag:

    skills/            -> "local"      first-party skills checked into this repo
    github_skills/     -> "github"     synced from external GitHub repos (untrusted)
    community_skills/  -> "community"  shared/downloaded skills (untrusted)
    project_skills/    -> "project"    project-specific skills for this deployment
    <workdir>/.skills/ -> "workdir"    per-thread, only when a cowork folder is
                                        selected (see set_working_directory) --
                                        lets a project the user cowork's into
                                        bring its own skills

This module intentionally does NOT fetch anything over the network (no
`git clone` of `github_skills/` sources) -- it only indexes whatever is
already on disk. Populating `github_skills/`/`community_skills/` is a
separate, later concern; this phase is "index what's there automatically",
not "go get more of it".
"""

from __future__ import annotations

import json
from pathlib import Path

from logging_utils import _log
from skills import Skill, SkillRegistry

# Fixed roots scanned on startup and on every manual reindex. Paths are
# relative to the process's working directory, same convention as
# ToolSandboxExecutor's "tool_envs/shared" and the SqliteSaver's "DB/..." .
SKILL_ROOTS: dict[str, Path] = {
    "local": Path("skills"),
    "github": Path("github_skills"),
    "community": Path("community_skills"),
    "project": Path("project_skills"),
}

# Sources that default to "untrusted" when a SKILL.md doesn't explicitly
# set `trust:` itself -- external, not-checked-in-by-you content.
_UNTRUSTED_DEFAULT_SOURCES = {"github", "community"}


def _split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a SKILL.md's leading '---' delimited block from its body.

    Deliberately hand-rolled instead of depending on PyYAML/python-
    frontmatter, since this project pins its own dependency set tightly
    (AWS Bedrock + open-weight models, no extras pulled in without
    asking). Only flat `key: value` lines are understood -- enough for
    the fields Skill actually has. If a project later wants richer YAML
    (nested structures, multi-line scalars), swap this function for a
    `yaml.safe_load()` call; nothing else in this module needs to change.
    """
    stripped = text.lstrip("\ufeff")  # tolerate a UTF-8 BOM
    lines = stripped.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, stripped

    end_idx = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end_idx = i
            break
    if end_idx is None:
        return {}, stripped

    frontmatter: dict[str, str] = {}
    for line in lines[1:end_idx]:
        line = line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        frontmatter[key.strip().lower()] = value.strip()

    body = "\n".join(lines[end_idx + 1:]).strip()
    return frontmatter, body


# The four recognized resource subfolders (v2 plan §6). Order here is the
# order they're recorded in on a Skill -- purely cosmetic, doesn't affect
# behavior.
_RESOURCE_SUBFOLDERS = ("references", "scripts", "assets", "templates")


def _scan_resource_folder(skill_dir: Path, folder_name: str) -> list[str]:
    """List every file under `<skill_dir>/<folder_name>/`, recursively, as
    paths relative to `skill_dir` (posix-style, e.g. "scripts/build.py").
    Records paths only -- content is never read here, same principle as
    the rest of skill parsing; an agent pulls actual content on demand via
    read_skill_resource() later. Missing folder -> empty list, not an error."""
    folder = skill_dir / folder_name
    if not folder.is_dir():
        return []
    return sorted(
        p.relative_to(skill_dir).as_posix()
        for p in folder.rglob("*")
        if p.is_file()
    )


def _parse_list(raw: str) -> list[str]:
    """Parse a frontmatter value like `a, b, c` or `[a, b, c]` into a list."""
    if not raw:
        return []
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    return [item.strip().strip('"').strip("'") for item in raw.split(",") if item.strip()]


def _parse_bundled_tools(raw: str, skill_name: str) -> list[dict]:
    """Parse a `bundled_tools:` frontmatter value as a JSON array of
    {"name", "prompt"} objects. Anything else is dropped with a warning
    rather than silently producing a malformed spec that Phase 3 would
    later choke on."""
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _log("WARNING", "Could not parse bundled_tools as JSON; ignoring", skill=skill_name, raw=raw[:200])
        return []
    if not isinstance(parsed, list):
        _log("WARNING", "bundled_tools must be a JSON array; ignoring", skill=skill_name)
        return []
    specs = [item for item in parsed if isinstance(item, dict) and item.get("name") and item.get("prompt")]
    if len(specs) != len(parsed):
        _log("WARNING", "Some bundled_tools entries were missing name/prompt and were dropped", skill=skill_name)
    return specs


class SkillDiscovery:
    """Walks skill root directories and registers what it finds into a SkillRegistry."""

    def __init__(self, registry: SkillRegistry):
        self.registry = registry

    def _parse_skill_md(self, skill_md: Path, source: str) -> Skill:
        text = skill_md.read_text(encoding="utf-8", errors="replace")
        frontmatter, body = _split_frontmatter(text)

        name = frontmatter.get("name") or skill_md.parent.name
        description = frontmatter.get("description", "")
        if not description:
            _log("WARNING", "SKILL.md has no description; agents won't be able to tell what it's for", skill=name, path=str(skill_md))

        default_trust = "untrusted" if source in _UNTRUSTED_DEFAULT_SOURCES else "trusted"
        skill_dir = skill_md.parent

        return Skill(
            name=name,
            description=description,
            instructions=body,
            source=source,
            path=str(skill_dir),
            tool_names=_parse_list(frontmatter.get("tools", "")),
            bundled_tool_specs=_parse_bundled_tools(frontmatter.get("bundled_tools", ""), name),
            references=_scan_resource_folder(skill_dir, "references"),
            scripts=_scan_resource_folder(skill_dir, "scripts"),
            assets=_scan_resource_folder(skill_dir, "assets"),
            templates=_scan_resource_folder(skill_dir, "templates"),
            triggers=_parse_list(frontmatter.get("triggers", "")),
            version=frontmatter.get("version", "1.0"),
            trust=frontmatter.get("trust", default_trust),
        )

    def index_root(self, root: Path, source: str) -> int:
        """Index one root directory (one level deep: `<root>/<name>/SKILL.md`).
        Returns how many skills were newly registered or refreshed from it."""
        root = Path(root)
        if not root.exists() or not root.is_dir():
            return 0

        resolved_root = root.resolve()
        registered = 0
        for entry in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
            skill_md = entry / "SKILL.md"
            if not skill_md.is_file():
                continue
            # Guard against a symlinked skill folder pointing outside its
            # own root -- discovery should only ever register skills that
            # actually live where they claim to.
            if not entry.resolve().is_relative_to(resolved_root):
                _log("WARNING", "Skipping skill folder that escapes its root via symlink", path=str(entry), root=str(root))
                continue
            try:
                skill = self._parse_skill_md(skill_md, source)
            except (OSError, UnicodeDecodeError) as e:
                _log("WARNING", "Could not read SKILL.md", path=str(skill_md), error=str(e))
                continue
            if self.registry.register_skill(skill):
                registered += 1
        return registered

    def index_single(self, skill_dir: Path, source: str = "github") -> Skill | None:
        """Parse and register exactly one already-verified skill folder,
        without walking any root directory tree. This is the mid-task hot
        path `SkillAcquisitionManager.ensure_skill()` calls right after a
        live-installed package passes `_verify()` (v2 plan §4.10) -- it's
        deliberately NOT `index_root`/`index_all` run against a single
        entry, because those re-derive the whole `<root>/<name>/SKILL.md`
        walk and re-check every skill's precedence, which is unnecessary
        latency for "one folder just appeared" (§4.11).

        Returns the parsed Skill on success (even if `register_skill`
        rejected it for being lower-precedence than an existing skill of
        the same name -- that's a registry decision, not a parse failure),
        or None if `SKILL.md` is missing or unreadable.
        """
        skill_dir = Path(skill_dir)
        skill_md = skill_dir / "SKILL.md"
        if not skill_md.is_file():
            _log("WARNING", "index_single: no SKILL.md in target folder", path=str(skill_dir))
            return None
        try:
            skill = self._parse_skill_md(skill_md, source)
        except (OSError, UnicodeDecodeError) as e:
            _log("WARNING", "index_single: could not read SKILL.md", path=str(skill_md), error=str(e))
            return None
        self.registry.register_skill(skill)
        return skill

    def index_workdir(self, workdir: Path) -> int:
        """Index the `.skills/` folder of a selected cowork working
        directory, source="workdir". Call this from
        DynamicAgentManager.set_working_directory() so a project a thread
        cowork's into can bring its own skills."""
        return self.index_root(Path(workdir) / ".skills", "workdir")

    def index_all(self, roots: dict[str, Path] | None = None) -> dict[str, int]:
        """Index every fixed root in SKILL_ROOTS (or a caller-supplied
        override). Does NOT touch the "workdir" source -- that's indexed
        separately, per-thread, via index_workdir()."""
        roots = roots if roots is not None else SKILL_ROOTS
        results = {source: self.index_root(root, source) for source, root in roots.items()}
        _log("REGISTRY", "Skill discovery complete", results=results, total_registered=len(self.registry.skills))
        return results
