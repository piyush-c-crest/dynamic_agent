"""
skills.py
=========
Phase 1 of Dynamic Skill Selection: the Skill data model and SkillRegistry.

A Skill is a higher-level bundle than a Tool: it carries instructions (a
system-prompt playbook an agent should follow), metadata, and the set of
tools it depends on -- as opposed to DynamicToolRegistry's Tool, which is
a single callable.

This module only defines the registry (register/get/list/remove +
persistence). It deliberately does NOT do any of the following yet --
those are later phases and will build on the `Skill` shape defined here:

  Phase 2 -- Skill Discovery: auto-indexing skills/, github_skills/,
             community_skills/, and per-workdir project skills (SKILL.md
             frontmatter -> Skill objects -> register_skill()).
  Phase 3 -- Skill Selection: DynamicAgentFactory.create_agent/refresh_tools
             show `SkillRegistry.list_skills()` to the LLM alongside
             `DynamicToolRegistry.list_tools()`, and splice a selected
             skill's `instructions` into the agent's system prompt.
  Phase 4 -- Trust boundary handling for bundled_tool_specs coming from
             untrusted (github/community) sources.

That's why `source`, `triggers`, `bundled_tool_specs`, and `trust` already
exist on `Skill` even though nothing populates them outside of manual
registration yet -- so Phase 2/3 are additive, not a reshape of this file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime

from logging_utils import _log

# Precedence used when two sources register a skill with the same name --
# highest-precedence source wins and becomes the "active" skill for that
# name. "workdir" (a project the user cowork'd into) and "project" (this
# repo's own skills/ folder) are trusted first-party sources; "github" and
# "community" are external and rank below them. "manual" (registered via
# the management API, e.g. for testing) ranks lowest since it has no
# on-disk source to be re-discovered from.
_SOURCE_PRECEDENCE = ("workdir", "project", "github", "community", "manual")

_DEFAULT_PERSIST_PATH = os.path.join("DB", "skills_index.json")


@dataclass
class Skill:
    name: str
    description: str
    instructions: str = ""
    source: str = "manual"          # "manual" | "project" | "github" | "community" | "workdir"
    path: str | None = None         # folder this skill was loaded from (set by Phase 2 discovery)
    tool_names: list[str] = field(default_factory=list)
    """Existing tool names (already in DynamicToolRegistry) this skill wants bound to an agent that selects it."""
    bundled_tool_specs: list[dict] = field(default_factory=list)
    """[{"name": "...", "prompt": "..."}] -- tools this skill needs but that
    aren't registered yet. Phase 3 routes these through
    DynamicToolRegistry.create_tool_from_prompt(), the same sandboxed
    validation path used for auto-created tools -- never executed directly."""
    triggers: list[str] = field(default_factory=list)
    """Optional keywords/phrases that hint when this skill applies. Not
    used for matching yet in Phase 1; Phase 3's selection prompt may
    surface these to the LLM as extra context."""
    version: str = "1.0"
    trust: str = "trusted"          # "trusted" | "untrusted" -- gates bundled tool execution from Phase 4 onward
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def id(self) -> str:
        """Stable identity across sources -- two skills named the same
        thing from different sources are different ids, only one of which
        is "active" in the registry at a time (see _SOURCE_PRECEDENCE)."""
        return f"{self.source}:{self.name}"

    def to_dict(self) -> dict:
        return asdict(self)


class SkillRegistry:
    """Manages skill registration and lookup -- the Skill-level counterpart
    to DynamicToolRegistry. Storage only: Phase 2 (folder discovery) and
    Phase 3 (LLM-driven selection) are built on top of this class rather
    than inside it.
    """

    def __init__(self, persist_path: str = _DEFAULT_PERSIST_PATH):
        self.skills: dict[str, Skill] = {}  # keyed by display name (the name shown to the LLM/UI)
        self.persist_path = persist_path
        self._load_persisted()

    def register_skill(self, skill: Skill, *, persist: bool = True) -> bool:
        """Register a skill under its display name.

        If a skill with the same name is already registered from a
        DIFFERENT source, precedence decides the winner (see
        _SOURCE_PRECEDENCE): a lower-precedence incoming registration is
        rejected (returns False) rather than silently overwriting a more
        trusted skill of the same name. Re-registering from the SAME
        source (e.g. Phase 2 re-indexing after a file edit) always
        replaces the previous entry.
        """
        existing = self.skills.get(skill.name)
        if existing is not None and existing.source != skill.source:
            existing_rank = _SOURCE_PRECEDENCE.index(existing.source) if existing.source in _SOURCE_PRECEDENCE else len(_SOURCE_PRECEDENCE)
            new_rank = _SOURCE_PRECEDENCE.index(skill.source) if skill.source in _SOURCE_PRECEDENCE else len(_SOURCE_PRECEDENCE)
            if new_rank > existing_rank:
                _log(
                    "REGISTRY", "Skill registration skipped; lower precedence than existing",
                    skill=skill.name, incoming_source=skill.source, existing_source=existing.source,
                )
                return False
            _log(
                "REGISTRY", "Skill overridden by higher-precedence source",
                skill=skill.name, old_source=existing.source, new_source=skill.source,
            )

        self.skills[skill.name] = skill
        _log(
            "REGISTRY", "Skill registered",
            skill=skill.name, source=skill.source, tools=skill.tool_names, total_skills=len(self.skills),
        )
        if persist:
            self._save_persisted()
        return True

    def get_skill(self, name: str) -> Skill | None:
        return self.skills.get(name)

    def list_skills(self) -> dict[str, str]:
        """name -> description, the same {name: description} shape
        DynamicToolRegistry.list_tools() already uses -- this is what
        Phase 3's tool/skill-selection prompt will show the LLM."""
        return {name: skill.description for name, skill in self.skills.items()}

    def list_skills_full(self) -> list[dict]:
        """Full metadata for the management API / UI."""
        return [s.to_dict() for s in self.skills.values()]

    def remove_skill(self, name: str, *, persist: bool = True) -> bool:
        if name not in self.skills:
            _log("WARNING", "Skill removal skipped; skill is not registered", skill=name)
            return False
        del self.skills[name]
        _log("REGISTRY", "Skill removed", skill=name, remaining_skills=len(self.skills))
        if persist:
            self._save_persisted()
        return True

    # ---------- persistence ----------
    # Mirrors DynamicToolRegistry's write-through-to-disk pattern for
    # generated tool source, so manually- or discovery-registered skills
    # survive a process restart. Phase 2's folder discovery will still
    # re-index skills/, github_skills/, etc. on startup and take
    # precedence over whatever was persisted here for the same source.

    def _load_persisted(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for entry in raw:
                skill = Skill(**entry)
                self.skills[skill.name] = skill
            _log("REGISTRY", "Loaded persisted skills", count=len(self.skills), path=self.persist_path)
        except (json.JSONDecodeError, OSError, TypeError) as e:
            _log("WARNING", "Could not load persisted skills; starting empty", error=str(e), path=self.persist_path)

    def _save_persisted(self):
        try:
            os.makedirs(os.path.dirname(self.persist_path) or ".", exist_ok=True)
            with open(self.persist_path, "w", encoding="utf-8") as f:
                json.dump([s.to_dict() for s in self.skills.values()], f, indent=2)
        except OSError as e:
            _log("WARNING", "Could not persist skills index", error=str(e), path=self.persist_path)
