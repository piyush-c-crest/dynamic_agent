# Dynamic Skill Architecture — Evaluation & Implementation Plan

## 1. Verdict

**`vercel-labs/skills` is useful, but only as an *installer/content source*, not as a runtime component.** It's a Node.js CLI (`npx skills add ...`) that copies whole skill directories onto disk. There's no Python SDK, no importable library, no in-process API — your backend can't "call into" it at request time. So it cannot become "the discovery/installation mechanism" in the sense of a live dependency your orchestrator talks to per-task.

What it *is* good for: it's a convenient way to physically fetch a full, well-formed skill package (`SKILL.md` + `references/` + `scripts/` + `assets/`) from a GitHub repo onto your disk, via a shell-out (`subprocess`/`run_shell_command`), landing it inside `github_skills/`. After that, it's out of the picture — your own `SkillDiscovery` → `SkillRegistry` → `DynamicAgentFactory` pipeline takes over exactly as it does today.

**Your existing architecture should absolutely stay.** It's already, structurally, the right shape:
- `Skill`/`SkillRegistry` already model source, trust, path, tool bindings.
- `SkillDiscovery` already does folder-based, format-compatible indexing.
- `DynamicAgentFactory.create_agent`/`refresh_tools` already do LLM-driven selection with a `"skills": [...]` field, already splice instructions via `skill_directive()`, already pull a skill's tools in via `_apply_skills()`.

You've unknowingly already built two of the three levels of "progressive disclosure" the vercel spec itself describes (name+description always shown → full instructions loaded on selection). The only real gap is **level 3**: `references/`, `scripts/`, `assets/` aren't discovered or loadable yet. That's a targeted extension, not a redesign.

**Two real gaps this exercise surfaced, unrelated to vercel-labs/skills itself, worth fixing regardless:**
- `DynamicAgentFactory.get_or_create` never calls `refresh_tools` — the call is present but literally commented out (`# not need for now`). Once a role exists, it can never pick up an additional skill on a later task with that role. This is the single biggest blocker to requirement #8 ("discover another skill if the first was insufficient").
- The system prompt (where `skill_directive()` gets spliced in) is only built once per task, on the *first* attempt (`if not task_messages: ...`). On a retry within the same task, even if `refresh_tools` ran and added a skill, the SystemMessage is never rebuilt — the new instructions would never reach the LLM unless injected some other way.

Both are addressed in section 6.

---

## 2. Current architecture — what's reusable as-is

| Piece | File | Reuse as-is? |
|---|---|---|
| `Skill` dataclass (`name`, `description`, `instructions`, `source`, `path`, `tool_names`, `bundled_tool_specs`, `triggers`, `trust`) | `skills.py` | Yes — extend with 3 new optional fields, don't restructure |
| `SkillRegistry` (register/get/list/persist, source precedence) | `skills.py` | Yes, unchanged |
| `SkillDiscovery.index_root` / `_parse_skill_md` (walks `<root>/<name>/SKILL.md`, parses frontmatter) | `skill_discovery.py` | Yes, extend `_parse_skill_md` to also record resource subfolders |
| `SKILL_ROOTS` (`local`/`github`/`community`/`project`) + `index_workdir` | `skill_discovery.py` | Yes, unchanged |
| `DynamicAgentFactory.create_agent` / `refresh_tools` — LLM sees `skill_registry.list_skills()`, returns `"skills": [...]`, `_apply_skills()` resolves them | `dynamic_langgraph_backend.py` | Yes — this IS your selection engine, extend it, don't replace it |
| `skill_directive()` (splices instructions into system prompt) | `dynamic_langgraph_backend.py` | Yes, extend to also append a resource manifest |
| `run_shell_command` tool, supports arbitrary `cwd` | `dynamic_langgraph_backend.py` | Yes — this is your `scripts/` execution mechanism, no new tool needed |
| `_update_tasks_tool` / `_apply_task_plan_update` (agent can mid-task insert/edit/remove upcoming tasks) | `dynamic_langgraph_backend.py` | Yes — this is your multi-skill decomposition mechanism (see §9), not something to duplicate |
| Evaluator's RETRY branch (`_evaluator_node`), which injects a `HumanMessage` and loops back to `_agent_executor_node` | `dynamic_langgraph_backend.py` | Yes — this is the natural hook point for "try again with a new skill" (see §6) |
| `Skill.trust` + untrusted-bundled-tool gate in `_apply_skills` | `dynamic_langgraph_backend.py` | Keep the code, don't rip it out — just default synced skills to `trust="trusted"` for this phase instead of deleting the mechanism (see §14) |

**What's missing** (net-new, not a rewrite of anything above): resource-file awareness on `Skill`, a sync utility to populate `github_skills/` with full packages, a scoped resource-reading tool, and re-enabling the two wiring gaps above.

---

## 3. Target architecture

```
                     ┌─────────────────────────────┐
                     │  vercel-labs/skills CLI      │   (external, shell-out only)
                     │  npx skills add <repo> --skill <name> -y
                     └───────────────┬─────────────┘
                                     │ writes full package to disk
                                     ▼
                        github_skills/<name>/
                          ├── SKILL.md
                          ├── references/*.md
                          ├── scripts/*.py
                          └── assets/*

                                     │ (Phase 2, extended)
                                     ▼
                        SkillDiscovery.index_root()
                          -> parses SKILL.md
                          -> RECORDS (doesn't read) references/scripts/assets paths
                                     │
                                     ▼
                        SkillRegistry  (unchanged)
                                     │
                                     ▼
              DynamicAgentFactory.create_agent / refresh_tools
                  Level 1: name+description shown to LLM (existing)
                  Level 2: instructions spliced on selection (existing)
                  Level 3: resource manifest appended, NOT full content (new)
                                     │
                                     ▼
                    Agent's system/task prompt + tool set
                       - skill.tool_names           (existing _apply_skills)
                       - read_skill_resource tool    (new, scoped to skill.path)
                       - run_shell_command w/ cwd    (existing, for scripts/)
```

Nothing upstream of `SkillRegistry` changes shape. Everything downstream of it (selection, splicing, tool binding) is the same mechanism you already have, extended by one more disclosure level.

---

## 4. What `vercel-labs/skills` (the CLI) should be responsible for

- Fetching a **complete, correctly-scoped** skill package from a GitHub/GitLab repo into `github_skills/<name>/` on disk.
- Nothing else. It should never run inside your process, never be given selection authority, never be trusted to decide *which* skill is relevant to a task — that's still yours.

Invoked as a subprocess, e.g. (illustrative, not literal code to write yet):
```
npx skills add <owner>/<repo> --skill <name> -y --agent generic
```
targeting a path under `github_skills/`. If a non-interactive/agent-agnostic flag doesn't exist for your use case, fall back to their documented download API (`https://skills.sh/api/download/{owner}/{repo}/{skill}`), which returns a scoped tarball/snapshot of just that skill's own folder — avoiding the whole-monorepo bug you saw reported.

## 5. What your Dynamic Agent system remains responsible for

- Deciding **whether** a task needs a skill at all (already: LLM may return `"skills": []`).
- Deciding **which** skill, from your own registry, not from vercel's live catalog at request time (your registry is the source of truth; vercel's catalog is only ever consulted to *populate* `github_skills/`, offline/on-demand, never mid-task).
- Binding a skill's tools into the agent (`_apply_skills`, existing).
- Splicing instructions into the prompt (`skill_directive`, existing).
- Deciding whether one skill was enough, or a retry/second skill/task-split is needed (existing evaluator + `update_tasks`, extended per §6/§9/§10).
- All trust/execution-safety decisions (existing `Skill.trust` gate).

---

## 6. Detailed execution flow

**A. Populating the catalog (offline / on-demand, not part of the hot path)**
1. You (or an admin tool) run a sync step — either manually via `npx skills add`, or via a small wrapper function — targeting `github_skills/`.
2. Next reindex (`POST /skills/reindex`, or next startup) picks it up via the existing `SkillDiscovery.index_root("github_skills", "github")`.

**B. Per-task selection (hot path — unchanged trigger points, extended payload)**
1. Planner produces `task_plan` as today — **no change**. Skill awareness stays out of the planner, consistent with your own stated principle that weaker OSS models drift when given more to self-track; the planner keeps doing task decomposition only.
2. `_agent_executor_node` calls `agent_factory.get_or_create(role, task_description, goal)`.
3. **First time this role is created:** `create_agent()` runs as today — LLM sees tools + skills (name/description), returns `"skills": [...]`, `_apply_skills()` resolves them, `skill_directive()` splices instructions (Level 2) **plus a new resource manifest** (Level 3 — filenames only, e.g. `references/api-patterns.md — API design conventions`, `scripts/generate_chart.py — renders a chart from CSV`) into the system prompt.
4. **If the role already exists (repeat task, same or later in the plan):** `get_or_create` currently short-circuits and returns the cached agent untouched. **Change this** to call `refresh_tools` when the task description differs meaningfully from what created the agent (simplest version: always call `refresh_tools`, since it already only *expands* tools/skills and is cheap — one extra LLM call per task, in line with your existing per-task LLM call budget).
5. If, mid-task, the agent needs a resource file it only knows about by name (from the Level-3 manifest), it calls the new `read_skill_resource(skill_name, relative_path)` tool, which is confined to that skill's own folder and returns the file's text — pulled in on demand, not pre-loaded.
6. If the agent needs to run a bundled script, its own spliced instructions (which the skill author wrote, e.g. "run `scripts/build_report.py` with `run_shell_command`") tell it to call the **existing** `run_shell_command(command=..., cwd=skill.path)` — no new tool.

**C. Discovering a skill was insufficient (mid-task)**
1. Task executes, evaluator runs as today.
2. On `RETRY`, right before appending `feedback_msg` and looping back, call `agent_factory.refresh_tools(role, task_description)` (this already exists and is safe — expand-only). If it returns new `skill_names`, build a short `HumanMessage` announcing the newly-available skill's instructions (same injection pattern already used for `forced_cutoff` and `feedback_msg` — **not** a SystemMessage rebuild, since `task_messages` persists across retries and LangChain conversations don't re-read the system message mid-thread) and append it alongside the feedback message.
3. This gives the SAME task a second attempt with a broadened toolkit/skillset — directly satisfies requirement #8, using machinery you already have (the RETRY loop), with one added call.

**D. Discovering a *different* task/role is needed entirely**
- Don't try to cram a second, unrelated skill into the same agent. If a task-level agent realizes mid-execution that the real fix is a separate specialized step (e.g. it's doing a "write report" task but realizes it first needs to "clean the data" — a genuinely different skill/role), it already has `update_tasks` — have it `insert_after_current` a new task with a distinct `agent_role`. That new task then goes through `create_agent()` fresh and gets its own 0–2 skill selection. This is cleaner than expanding one agent's skill budget indefinitely, and matches your "external planner, don't self-track" learning — the *plan* changes, not one agent's internal complexity.

---

## 7. Required code changes (by file/function — no full implementations yet, per your instruction)

**`skills.py`**
- Add to `Skill`: `references: list[str] = field(default_factory=list)`, `scripts: list[str] = field(default_factory=list)`, `assets: list[str] = field(default_factory=list)` — each a list of paths *relative to `skill.path`*, not content.
- No changes to `SkillRegistry`.

**`skill_discovery.py`**
- `_parse_skill_md`: after building the `Skill`, list the sibling `references/`, `scripts/`, `assets/` subfolders (if present) under `skill_md.parent` and populate the three new fields with relative filenames. Don't read file contents at index time — keeps `skills_index.json` small and discovery cheap.
- New (optional, separate module `skill_sync.py` is cleaner than bloating this file): `sync_skill_from_repo(owner_repo: str, skill_name: str | None, target_root: Path = Path("github_skills")) -> Path` — shells out to the CLI or hits the download API, lands the package under `github_skills/`, returns the path. Idempotent: skip the fetch if `github_skills/<skill_name>/SKILL.md` already exists, unless a `force=True` refresh is requested.

**`dynamic_langgraph_backend.py`**
- `DynamicAgentFactory.skill_directive()`: after the existing instructions block, append a one-line-per-file manifest built from `skill.references`/`skill.scripts`/`skill.assets` (filenames only — keep it short; this is the "level 3 pointer", not the content).
- `DynamicAgentFactory.get_or_create()`: uncomment/re-enable the `refresh_tools` call for the cached-role branch.
- New tool factory on `DynamicToolRegistry` (or a small standalone function registered the same way `_read_file_tool` etc. are): `_read_skill_resource_tool()` → `read_skill_resource(skill_name: str, relative_path: str) -> dict`. Confinement logic mirrors `path_utils`' workdir confinement, just rooted at `skill_registry.get_skill(skill_name).path` instead of the thread's workdir — resolve, verify it stays inside that skill's own folder, reject `..` traversal, reject anything not in `references/`/`scripts/`/`assets/`.
- `_evaluator_node`: in the `RETRY` branch, call `refresh_tools` before constructing `feedback_msg`; if `skill_names` grew, append a second `HumanMessage` carrying the new skill's directive text (via `skill_directive()`) so it reaches the LLM without needing a SystemMessage rebuild.
- `main.py`: one new endpoint, `POST /skills/sync`, body `{owner_repo, skill_name}`, calls the new `sync_skill_from_repo` then `reindex_skills()` — mirrors the existing `/skills/reindex` endpoint pattern exactly.
- `index.html`: optional — add a "Sync from GitHub" input next to the existing "Reindex" button in the Skills modal. Not required for the architecture to work; purely a convenience.

That's the complete file list: **`skills.py`, `skill_discovery.py`, `dynamic_langgraph_backend.py`, `main.py`**, optionally `index.html`. No new top-level modules besides the small `skill_sync.py`, and even that could just live as functions inside `skill_discovery.py` if you'd rather not add a file.

---

## 8. Skill package/resource handling design

- **`SKILL.md`** — parsed as today, unchanged.
- **`references/*.md`** — path recorded at discovery, content read on demand only, via `read_skill_resource`. Rationale: reference docs can be large; pre-loading all of them into every task's system prompt defeats the point of "progressive disclosure" and burns context on your free-tier models unnecessarily.
- **`scripts/*`** — path recorded, **never read into context** — executed via `run_shell_command(cwd=skill.path)` per the skill's own instructions. This is the cheapest possible integration: zero new execution machinery.
- **`assets/*`** (templates, images, boilerplate files) — path recorded; exposed the same way as references (readable via `read_skill_resource`) or, if a task needs to *copy* an asset into the working directory, that's just `read_file`... no — `read_skill_resource` to get content, then the agent's own `write_file` (already confined to workdir) to place it where needed. No new tool required for that either.
- **Unsupported for now, explicitly**: nested nested subfolders inside `references/`/`scripts/`/`assets/`, and root-level `SKILL.md` layouts (matches the limitation already flagged in the Phase 2 discussion). Call this out as a known gap, not silently.

---

## 9. Dynamic multi-skill selection strategy

- Keep the existing prompt rule as the primary lever: **0–2 skills per agent role, selected only on genuine description match** — this cap already exists in your `create_agent`/`refresh_tools` prompts and is doing real work preventing over-selection.
- If a task plausibly needs 3+ skills, treat that as a signal the **planner under-decomposed the task**, not a reason to raise the cap. The fix is `update_tasks` splitting it into multiple role-scoped tasks (§6D), each with its own tight 0–2 skill budget. This keeps per-agent context small and keeps skill selection legible/debuggable (you can look at `GET /agents` and see exactly which skill went with which role).
- `refresh_tools` already unions with previously-applied skills (`set(agent_conf.get("skill_names", [])) | set(applied_skill_names)`) rather than replacing — so a role naturally accumulates skills across retries/tasks without ever losing one, which is the right default (never surprise-remove a working capability).

---

## 10. How the agent decides when to search for/install another skill

Two distinct triggers, don't conflate them:

1. **"This exact task needs more" (same task, same role)** — driven by the evaluator's `RETRY` verdict (§6C). Fully automatic, no agent self-judgment required — this is exactly the kind of externally-driven signal your own "weaker models need external structure" principle favors over asking the agent to introspect "do I need a new skill?"
2. **"A different kind of work is needed" (new task, possibly new role)** — driven by the agent's own `update_tasks` call, which already exists for exactly this purpose (inserting/editing upcoming plan steps). No new decision-making logic needed here either — just make sure the agent's system prompt (or a shared instruction fragment) mentions that inserting a task with a more specific `agent_role` is the right move when it recognizes it's missing a distinct capability, not just a missing tool.

Neither path requires the agent to know about `vercel-labs/skills`, GitHub, or installation at all — that layer is invisible to it. It only ever sees "Available skills: {name: description}", same as today.

---

## 11. Caching/reuse strategy

- **On disk (persistent, restart-safe):** `github_skills/<name>/` — `sync_skill_from_repo` checks for existence before fetching, so a skill is only ever downloaded once unless explicitly forced.
- **Registry (persistent, restart-safe):** `SkillRegistry`'s existing `DB/skills_index.json` write-through — unchanged, already handles this.
- **In-memory (process lifetime only):** `DynamicAgentFactory.agents` cache — a role's resolved tool/skill set persists for the life of the process once created; `refresh_tools` only *adds*, never *re-fetches* from disk. This is already correct and needs no change.
- **What's genuinely new to cache-manage:** nothing — the resource-manifest approach (§8) means you're never caching file *content* in memory beyond what `read_skill_resource` returns for that one call, so there's no new cache invalidation problem to design for.

---

## 12. Testing plan — concrete scenarios

1. **Format compatibility smoke test:** hand-copy one real skill folder from `vercel-labs/agent-skills` (e.g. `frontend-design`, including its `references/` if any) into `github_skills/frontend-design/`, hit `/skills/reindex`, confirm `GET /skills` shows it with correct `references`/`scripts`/`assets` lists populated.
2. **Selection test:** ask a task that clearly matches that skill's description; confirm via `GET /agents` that `skills` includes it and `tools` includes anything it declared.
3. **Resource pull-through test:** confirm the agent, when it needs something from `references/`, actually calls `read_skill_resource` rather than hallucinating content — check the trace/logs for the tool call.
4. **Script execution test:** a skill whose instructions say "run scripts/x.py via run_shell_command" — confirm the tool call happens with the correct `cwd`.
5. **Confinement test:** call `read_skill_resource` with a `relative_path` like `../../etc/passwd` or an absolute path — confirm it's rejected, mirroring the symlink-escape test already written for `SkillDiscovery`.
6. **Mid-task augmentation test:** engineer a task that fails its first attempt for a reason a second skill would fix (two skills registered, only the wrong one auto-selected initially); confirm the evaluator's RETRY path calls `refresh_tools`, the second skill gets added, and the injected `HumanMessage` actually reaches the LLM (check `task_messages` in the trace).
7. **Task-split test:** a task genuinely needing two unrelated skills — confirm the agent (or you, manually, if the model doesn't self-trigger this reliably) uses `update_tasks` to split it, and each resulting task gets its own clean 0–2 skill selection rather than one bloated role.
8. **Idempotent sync test:** call `/skills/sync` twice for the same skill — confirm the second call doesn't re-fetch (check logs/timestamps on the folder).
9. **Regression test:** run your existing Phase 1–3 test suite (precedence, malformed-JSON bundled_tools, untrusted-skill gating) unchanged — none of this should behave differently for skills that don't declare `references/scripts/assets`.

---

## 13. Phased implementation plan (simplest working version first)

**Phase 5a — Resource awareness (read-only, no execution change)**
- Extend `Skill` with `references`/`scripts`/`assets`.
- Extend `SkillDiscovery._parse_skill_md` to populate them.
- Extend `skill_directive()` to append the manifest.
- No new tools yet. Test: manifest shows up correctly in a spliced system prompt for a hand-crafted multi-file skill folder.

**Phase 5b — On-demand resource reading**
- Add `read_skill_resource` tool + confinement.
- Test: agent successfully pulls a `references/*.md` file it was told about but not shown.

**Phase 5c — Re-enable adaptive refresh**
- Un-comment `refresh_tools` in `get_or_create`.
- Wire `refresh_tools` into the evaluator's `RETRY` branch + the follow-up `HumanMessage` injection.
- Test: mid-task skill augmentation scenario (§12.6).

**Phase 5d — GitHub sync utility**
- `sync_skill_from_repo` (shell-out or download API), `/skills/sync` endpoint.
- Test: idempotent sync test (§12.8), full pipeline against a real `vercel-labs/agent-skills` entry.

**Phase 5e — Multi-skill task-splitting guidance**
- Add a short shared instruction fragment (in the base system prompt template, not a new mechanism) nudging agents to use `update_tasks` when they recognize a distinct-capability need, rather than asking for more skills on themselves.
- Test: task-split scenario (§12.7).

This ordering means you have something demoable after 5a+5b alone (skills with real reference material, working end-to-end) before touching the retry-loop or sync-automation pieces, which are individually higher-risk changes to a graph that's already carrying real production behavior (retry/evaluator logic).

---

## 14. Potential problems / tradeoffs

- **Re-enabling `refresh_tools` in `get_or_create` adds one LLM call per task for any role that's reused**, even when nothing new is needed (it already returns early if nothing changed, but it still costs a call to find that out). On a free-tier Bedrock model this is a real, measurable cost increase — worth watching in your token-usage logs after 5c ships. If it's too expensive, a cheaper middle ground is: only call `refresh_tools` when the evaluator actually issues a RETRY (skip it on cache-hit-and-first-attempt-succeeds paths), i.e. do 6C but not step B.4 in §6.
- **`npx skills add` requires Node.js/npm available in your runtime environment.** If your deployment target is Python-only (no Node), you'd need the download-API fallback exclusively — confirm this before committing to the CLI path.
- **Trust deferred, not removed, has a real consequence:** with `github`-sourced skills defaulted to `trust="trusted"` for this phase (per your instruction), their `bundled_tool_specs` (if any) WILL get auto-created and sandboxed-executed the same as your own generated tools. That's a deliberate, temporary widening of your attack surface — fine for now per your explicit scope decision, but flag it clearly (e.g. a log line or a README note) so it isn't forgotten before this goes anywhere production-facing.
- **Skill-authored `scripts/` running via `run_shell_command`** aren't sandboxed the way your auto-created *tools* are (those go through `ToolSandboxExecutor`/smoke-testing). A skill's `scripts/foo.py` runs with whatever `run_shell_command` already allows. This is a second, separate trust gap from the tool one above — same "acceptable for now, must revisit" category.
- **Context bloat is still possible even with manifests-only.** If a role accumulates several skills over a long-running thread (via the union-not-replace behavior in `refresh_tools`), the system prompt grows monotonically for that role's lifetime. Worth a simple cap later (e.g. don't add a skill if the role already has N), but not needed for the initial version.
- **The download-API scoping bug you found in the vercel-labs issue tracker** is a live reminder not to trust "it only grabbed the skill folder" blindly — a basic file-count/size sanity check on whatever `sync_skill_from_repo` pulls down (reject/warn if a "single skill" fetch returns hundreds of files) is cheap insurance worth including even in the "don't worry about trust yet" phase, since it's a correctness issue, not a security one.