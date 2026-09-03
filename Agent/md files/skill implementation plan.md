# Dynamic Skill Architecture v2 — Live Runtime Discovery, Acquisition & Usage

## 0. What changed from v1

v1 treated skill installation as an **admin operation**: someone runs `npx skills add` or hits `/skills/sync` ahead of time, `SkillDiscovery` indexes it, and only *then* does it become selectable. That's gone.

v2's rule: **a skill that doesn't exist locally yet is just a candidate the agent hasn't acquired yet.** Discovery, installation, verification, and indexing all happen inline, during task execution, triggered by the running agent itself — not by you.

The core mechanism that makes this work without redesigning your graph: **skill acquisition becomes a capability, exposed the same way `update_tasks` already is — as something the agent's own execution loop can reach for.** No new LangGraph node, no new async orchestration layer, no bespoke state machine. One new manager class, one new always-available tool, and one new inline check inside the existing agent-creation/refresh prompts.

---

## 1. What we already have (unchanged reuse)

| Piece | File | Role in v2 |
|---|---|---|
| `Skill` dataclass | `skills.py` | Unchanged shape (v1's `references`/`scripts`/`assets` additions still apply, plus `templates` — see §6) |
| `SkillRegistry` | `skills.py` | Unchanged — still the single source of truth the agent selection prompt reads from |
| `SkillDiscovery.index_root` / `_parse_skill_md` | `skill_discovery.py` | Reused, but no longer the only entry point — see new `index_single` in §4.10 |
| `DynamicAgentFactory.create_agent` / `refresh_tools` / `_apply_skills` / `skill_directive` | `dynamic_langgraph_backend.py` | Unchanged mechanics, extended with one new field each (`skill_gap`) — see §4.1 |
| `_evaluator_node` RETRY branch, `feedback_msg` injection pattern | `dynamic_langgraph_backend.py` | Reused as the message-injection pattern for "a skill just became available" — see §4.13 |
| `run_shell_command(cwd=...)` | `dynamic_langgraph_backend.py` | Reused both for skill `scripts/` execution *and* as the mechanism that shells out to the `skills` CLI itself |
| `update_tasks` — always-available agent tool | `dynamic_langgraph_backend.py` | Direct precedent for the new `request_skill_acquisition` tool — same pattern, same trust level |
| Existing SSE status-event streaming (plan/status/evaluation "Working" blocks) | `main.py` (not in this repo snapshot, referenced from memory) | Reused, not replaced — see §5 |

**What's net-new:** a `SkillAcquisitionManager` (new module `skill_acquisition.py`), one new always-on tool, two small inline hooks in `create_agent`/`refresh_tools`, and a targeted single-skill indexer.

---

## 2. Core architectural decision: acquisition as a capability, not a phase

Everything in this document routes through one function:

```
SkillAcquisitionManager.ensure_skill(capability_description: str, task_description: str) -> AcquisitionResult
```

`AcquisitionResult` = `{status: "already_present" | "installed" | "not_found" | "failed", skill_name: str | None, reason: str}`

This function is **synchronous, idempotent, and safe to call repeatedly.** It's called from exactly two places:

**A. Config-time (upfront gap)** — inside `create_agent()` / `refresh_tools()`, when the selection LLM's JSON reports a capability gap (new `"skill_gap"` field, sibling to the existing `"new_tools"` field — same pattern you already use for tools). This covers your example: *"Create a professional PDF report"* → no local skill matches → `ensure_skill("PDF/report generation")` runs **before** the agent's first system prompt is finalized, so the very first `skill_directive()` splice already includes it.

**B. Execution-time (self-service mid-task)** — a new tool, `request_skill_acquisition(capability_description: str)`, always registered for every agent (exactly like `update_tasks` is). The agent calls it itself when it recognizes a gap while already working — your "started with Skill A, realizes it needs B" case. It runs through `_tools_node` like any other tool call: no graph change, no state loss, `task_messages` just gets a new `ToolMessage`.

Both call sites share one manager, one lock table, one cache, one verification routine. Nothing about *how* acquisition works differs between "obvious upfront gap" and "discovered mid-task" — only *when* it's triggered differs.

---

## 3. Runtime execution flow (your example, mapped to real code)

```
User: "Create a professional PDF report from these documents."
  │
  ▼
_planner_node                          — unchanged, produces task_plan
  │
  ▼
_agent_executor_node → agent_factory.get_or_create(role="report_writer", ...)
  │
  ▼
create_agent(): selection LLM sees Available tools + Available skills (local registry)
  → none matches "PDF generation" well
  → LLM returns "skill_gap": "PDF report generation with styling"
  │
  ▼
SkillAcquisitionManager.ensure_skill("PDF report generation with styling", task_description)
  │
  ├─ 1. Registry/cache check (§4.16) — not present, continue
  ├─ 2. Acquire per-key lock (§4.14) — "pdf report generation" key
  ├─ 3. Discover candidates — shell `skills find "pdf report generation"` (§4.2/4.3)
  ├─ 4. Rank + quality-gate candidates (§4.3)
  ├─ 5. Install top candidate — shell `skills add <owner/repo> --skill <name> -y` (§4.4/4.5)
  ├─ 6. Verify package on disk — SKILL.md + declared subfolders (§4.9)
  ├─ 7. Targeted index — SkillDiscovery.index_single() registers it (§4.10/4.11)
  ├─ 8. Release lock, cache result
  │
  ▼
create_agent() resumes: skill now appears in `skill_registry.list_skills()`
  → _apply_skills() binds its tools, skill_directive() splices instructions + resource
    manifest into the system prompt (Level 2 + Level 3, unchanged from v1 §8)
  │
  ▼
Agent executes: reads references/ on demand, runs scripts/ via run_shell_command,
  pulls assets/templates via read_skill_resource, generates the report
  │
  ▼
_evaluator_node: PASS → _assembler_node → END
                 RETRY → loop back to _agent_executor_node (unchanged v1 mechanism;
                          agent may call request_skill_acquisition again if the
                          feedback reveals a second missing capability)
```

Nothing upstream of `create_agent`/`_agent_executor_node` changes. Nothing about the graph's shape changes. The acquisition pipeline is a function call that happens to shell out and touch disk, same category of operation as `_create_missing_tools()` already is today.

---

## 4. The installation lifecycle — point by point

### 4.1 How the agent triggers live discovery

Two triggers, one manager (§2). Concretely:
- `create_agent`/`refresh_tools` prompts get one new instruction: *"If no available skill's description is a genuine match but this role clearly needs a specialized capability (not just a tool), set `skill_gap` to a short phrase describing that capability. Otherwise omit it."* Mirrors the existing `new_tools` instruction verbatim in tone.
- The new tool `request_skill_acquisition(capability_description: str)` is described to every agent in its tool list the same way `update_tasks` is, with instructions to use it "when you recognize mid-task that you need a distinct capability no current tool or skill provides."

### 4.2 How `vercel-labs/skills` is used for discovery

`skills find "<query>"` (non-interactive, query provided) — a real remote search against the skills.sh index, returns plain text: `owner/repo@skill_name`, install count, URL, one candidate per block. `SkillAcquisitionManager._search(query)` shells this out via `subprocess.run(["npx","skills","find",query], capture_output=True, timeout=SEARCH_TIMEOUT_SECONDS)` and parses the stdout with a small regex/line-scanner into a list of `Candidate(owner_repo, skill_name, installs, url)`.

*Open item to verify before implementation:* whether your deployment has Node/`npx` available at all. If not, `skillsmd` (the Python port, `uvx skillsmd find "<query>"`) is a drop-in replacement with an identical command surface — confirm which one is actually installed in your runtime image before writing `_search`.

### 4.3 How we identify the exact skill/repo to install

Two-stage filter, mirroring what the `find-skills` meta-skill itself instructs an LLM agent to do — you're just doing it in code instead of relying on an external agent's judgment:
1. **Hard gate (pure Python, no LLM):** drop candidates below an install-count floor (config, default e.g. 100) unless the source repo owner is on a small trusted allowlist (`vercel-labs`, `anthropics`, your own org). This matches the vetting rule the CLI's own docs describe.
2. **Selection (one small LLM call, same pattern as `agent_creation_llm`):** if more than one candidate survives the gate, send the top N (e.g. 5) with their descriptions/URLs plus `capability_description` to the LLM and ask it to pick the single best match or return `"none"`. If only one candidate survives the gate, skip the LLM call — resolve directly.

If nothing survives even the hard gate, `ensure_skill` returns `not_found` immediately — no install attempted, no LLM call wasted.

### 4.4 How installation is started from the running agent

`SkillAcquisitionManager._install(candidate) -> Path` shells out: `subprocess.run(["npx","skills","add", candidate.owner_repo, "--skill", candidate.skill_name, "-y", "--agent","generic"], cwd=PROJECT_ROOT, timeout=INSTALL_TIMEOUT_SECONDS, capture_output=True)`.

*Open item, carried over and sharpened from v1:* the CLI's documented flags target `./<agent>/skills/` (or `~/<agent>/skills/` with `-g`) — there is no confirmed flag to point it at an arbitrary directory like your existing `github_skills/`. Before committing to this exact invocation, spike it once by hand and see where `--agent generic` actually lands the files. If it doesn't land where you need it, two fallbacks, both already anticipated in v1 §4: (a) install to its default location then move/symlink the folder into `github_skills/<skill_name>/` yourself, or (b) skip the CLI's `add` command entirely and hit the scoped download API (`https://skills.sh/api/download/{owner}/{repo}/{skill}`) directly, writing the response yourself into `github_skills/<skill_name>/` — this also sidesteps the whole-monorepo-download bug already flagged in v1 §14.

### 4.5 Synchronous, asynchronous, or hybrid?

**Hybrid, resolved simply: synchronous from the task's point of view, off the request thread underneath it.** Concretely:
- From `_agent_executor_node`'s perspective (and the agent's), `ensure_skill()` is a plain blocking call — same as an LLM call or a tool execution already is. No `interrupt()`/resume, no separate async job queue.
- Because your API layer already runs graph execution inside Starlette's threadpool (this is exactly why the ContextVar token-tracking approach broke, per your own prior findings), the subprocess call is *already* off the main event loop by construction. You get "doesn't block other users' requests" for free, without adding a task queue.
- Net effect: one task's execution pauses for a few seconds while its own skill installs; other concurrent threads/tasks are unaffected. This is the same latency shape a slow LLM call already has, so it needs no new UX pattern beyond what streaming already surfaces (§5).

A true async job-queue (Celery/RQ-style, with polling) would only be justified if installs commonly took tens of seconds to minutes. `npx skills add` copying a handful of markdown/script files is not that; don't build for a problem you don't have yet.

### 4.6 How the agent waits for installation to finish

It doesn't have to do anything special — `ensure_skill()` is a normal Python function call in the middle of `create_agent()`/`refresh_tools()`/the tool executor. The calling code (`_agent_executor_node` or `_tools_node`) simply doesn't proceed past that line until the function returns. "Waiting" is just the call stack, not a mechanism you build.

### 4.7 How we detect installation completion

Not by trusting the subprocess exit code alone (a `0` exit doesn't guarantee a well-formed package, and a non-zero exit with partial output is possible). Completion is detected by **re-verification against the filesystem** (§4.9) after the subprocess returns, regardless of exit code. Exit code is used only to short-circuit an obvious hard failure (nonzero + empty output) before even trying to verify.

### 4.8 Handling errors, timeouts, partial/corrupted installs

All failure modes funnel into the same `AcquisitionResult(status="failed", reason=...)` shape:
- **Subprocess timeout** → kill the process, `status="failed", reason="install_timeout"`.
- **Nonzero exit** → `status="failed", reason="install_error: <stderr excerpt>"`.
- **Partial package** (dir created, `SKILL.md` missing or fails to parse, or frontmatter missing required `name`/`description`) → `status="failed", reason="incomplete_package"`. The partial directory is renamed with a `.partial-<timestamp>` suffix rather than left as a valid-looking `github_skills/<name>/` — this prevents a later call from mistaking it for a real, cached install (§4.16 depends on this).
- **Oversized/scope-creep fetch** (v1 §14's "grabbed the whole monorepo" bug) → a file-count/size sanity check (e.g. reject if >200 files or >20MB for what should be one skill folder) runs as part of verification; over-limit is treated as `incomplete_package` too, not partially accepted.
- **Retry policy:** try up to 2 ranked candidates (§4.3) before giving up. On final failure, `ensure_skill` returns `not_found`/`failed` — never raises — so the calling code always has a value to act on.
- **What the agent does with a failure:** the manager's result is turned into ordinary message content. In the config-time path (§2A), `create_agent` just proceeds without that skill (same as if the LLM had returned `skills: []`) and the system prompt notes the gap so the agent doesn't silently pretend it has a capability it doesn't. In the tool-call path (§2B), the `ToolMessage` result literally says "Could not acquire a skill for X (reason). Continue with available tools, or ask the user for guidance if this capability is essential." — feeding the same evaluator/RETRY loop you already have; a real capability gap surfaces as a normal RETRY-with-feedback cycle, not a crash.

### 4.9 Verifying the complete package before continuing

`SkillAcquisitionManager._verify(path: Path) -> bool`, reusing `SkillDiscovery._parse_skill_md` as the parser:
1. `SKILL.md` exists and parses; `name`/`description` frontmatter present.
2. File-count/size sanity check (§4.8).
3. For each of `references/`, `scripts/`, `assets/`, `templates/` that exists, just confirm it's a real directory (content isn't read at this stage — same "record paths, don't read content" principle as v1 §8).
4. If the instructions text explicitly names a file (e.g. "run `scripts/build_report.py`") that isn't present in the recorded `scripts` list, log a warning but don't hard-fail — the agent will simply get a tool error later if it tries to run something that isn't there, same as any other missing-file case today.

Only a package that passes step 1 gets registered at all (§4.10).

### 4.10 How the newly installed skill becomes visible to `SkillDiscovery`/`SkillRegistry`

New, small function in `skill_discovery.py`: `SkillDiscovery.index_single(skill_dir: Path, source: str = "github") -> Skill | None` — parses exactly one `SKILL.md` (the same parsing logic `index_root` already uses internally, factored out rather than duplicated) and calls `skill_registry.register()` for that one skill. This is what `ensure_skill` calls after `_verify` passes.

### 4.11 Do we need to call `reindex_skills()` automatically?

**No — and that's the point.** `reindex_skills()` walks every root and re-parses everything; calling it mid-task for one new folder is wasteful and adds latency proportional to your whole skill library, not to the one thing that changed. `index_single()` (§4.10) is the mid-task path. `POST /skills/reindex` and startup indexing remain exactly as they are for admin/bulk use (picking up anything installed by other means) — unchanged, still useful, just not on this hot path.

### 4.12 How the running agent gets access without recreating the whole agent

This is what `refresh_tools()` already does, unchanged: it's expand-only, and it edits `self.agents[role]` in place (`tool_names`/`skill_names` lists), not replacing the cached config. The only thing v2 adds is *when* it's called: right after `ensure_skill` reports `installed`, the calling code (either `create_agent` before first use, or the `request_skill_acquisition` tool handler) calls `agent_factory.refresh_tools(role, task_description)` so the now-registered skill gets picked up by the very next tool/prompt resolution — no new agent object, no lost `task_messages`.

### 4.13 How the agent's prompt/tool configuration refreshes after a new skill appears

This closes the gap v1 flagged and deferred (v1 §1: "the system prompt is only built once per task; a retry never rebuilds it"). v2 makes this mandatory, not optional, because it's now on the hot path every time acquisition succeeds mid-task:
- After `refresh_tools` returns new `skill_names`, build the skill's `skill_directive()` text (same function as always) and inject it as a fresh `HumanMessage` appended to `task_messages` — the exact same injection pattern already used for `feedback_msg`/`forced_cutoff` in `_evaluator_node`. `task_messages` persists across the rest of the task, so this is enough; there's still no need to rebuild the original `SystemMessage`.
- Tool binding is separate and already handled: `_apply_skills()` returns the skill's `tool_names`, which get unioned into `all_tool_names` the next time the agent is invoked — LangChain resolves the tool list per-invocation from the agent's bound tools, so a newly added tool is callable on the very next turn without any special-casing.

### 4.14 Preventing race conditions with concurrent installs

An in-process lock table keyed by normalized skill identifier (`owner_repo@skill_name`, or the capability query string before resolution): `_install_locks: dict[str, threading.Lock]`, guarded by one small meta-lock for dict mutation (classic double-checked-locking pattern). Two tasks wanting the *same* skill simultaneously: the second blocks on the same lock, and once it acquires the lock, it re-checks the registry first (§4.16) and finds the skill already there — no duplicate install, it just proceeds.

Two tasks wanting *different* skills concurrently: different lock keys, no contention, both proceed in parallel — this is fine, they touch different directories.

*Caveat to flag, not solve now:* this lock is process-local. If you ever scale to multiple backend worker processes, this needs a filesystem lock (a `.installing` marker file with a PID/timestamp, checked and cleaned up on failure) instead of an in-memory `threading.Lock`. Not needed at your current single-process scale — noted as a forward-looking limitation, matching how you've already flagged the SQLite connection-leak issue as a known, deferred-for-now item.

### 4.15 Handling duplicate installation requests

Same mechanism as §4.14 — "duplicate request" and "race condition" are the same problem from two different angles (concurrent vs. sequential-but-repeated). A duplicate request that arrives *after* a prior install already completed is just a normal cache hit (§4.16); a duplicate that arrives *while* one is in flight blocks on the lock and then also resolves to a cache hit once it wakes up.

### 4.16 Reusing already-installed skills instead of re-downloading

`ensure_skill()`'s very first step, before touching the lock or the network at all: check `skill_registry.get_skill(name)` (if the capability description already resolved to a known skill name in a prior call — see the negative/positive cache below) or check `github_skills/<candidate_name>/SKILL.md` existence directly on disk. Either hit short-circuits straight to `already_present`, skipping discovery, installation, and verification entirely. This is nearly free (dict lookup / single `Path.exists()`), so it's always run first, not just as an optimization for the common case.

A small in-memory cache also maps `capability_description → resolved skill_name` (positive) and `capability_description → "not_found" (with timestamp)` (negative, short TTL e.g. 10 minutes) so that repeated *phrasings* of the same underlying need don't re-trigger a full search-and-rank cycle, and a recently-failed search doesn't get retried on every subsequent task in a short window.

---

## 5. Agent UX / execution state — is a formal state machine needed?

**No.** The phases you sketched (`DISCOVERING_SKILL → INSTALLING_SKILL → WAITING_FOR_INSTALL → VERIFYING_SKILL → LOADING_SKILL → USING_SKILL → CONTINUE_TASK`) are real and worth naming, but they don't need to be a control-flow state machine with persisted transitions — they're already fully represented by:
- **Control flow:** an ordinary sequential Python function (`ensure_skill`) with try/except around each phase. `WAITING_FOR_INSTALL` isn't a state at all in this model — it's just "the subprocess call hasn't returned yet," i.e., the calling frame is still on the stack (§4.6). There's nothing to persist or resume, because nothing crosses a request boundary.
- **Visibility/UX:** each phase becomes a `_log(...)` call at the point it happens, using the same structured logging you already have (`_log`, `_log_block`). Tag them with a consistent event kind, e.g. `_log("SKILL-ACQUISITION", "Installing skill", skill=candidate.skill_name, phase="installing")`. Since your SSE layer already streams `_log`-driven plan/status/evaluation events into collapsible "Working" blocks in the frontend, these phase logs ride the exact same channel — no new streaming infrastructure, no new event schema beyond one more recognized "kind." This is what gives you the example UX transcript (*"Searching for a skill... Found a suitable skill. Installing it... Skill installed and verified..."*) essentially for free.

Building a real persisted FSM would only be justified if installation needed to survive a process restart mid-flight, or if you wanted a user to be able to navigate away and come back mid-install across separate HTTP requests. Neither applies here — installs are seconds-scale and happen fully within one already-blocking node/tool call, same as an LLM turn does today. Treat the phase list as **event labels for tracing and streaming UX**, not as an execution model.

Suggested phase labels for `_log` calls (for consistency, not enforcement): `searching`, `candidate_selected`, `installing`, `verifying`, `indexing`, `ready`, `failed`.

---

## 6. Skill resource-loading mechanism (extended from v1)

Unchanged principle from v1 §8: **progressive disclosure, three levels** — name+description always visible, instructions spliced on selection, resource *paths* (not content) appended as a manifest. v2 adds one recognized subfolder and reaffirms that a live-acquired skill goes through the identical path as a pre-existing one — acquisition just makes the folder exist; everything downstream is unchanged v1 machinery.

| Subfolder | When the agent uses it | Mechanism |
|---|---|---|
| `SKILL.md` | Always — defines how to approach the task | Instructions spliced into system prompt via `skill_directive()`, unchanged |
| `references/*.md` | When it needs domain knowledge it doesn't already have | `read_skill_resource(skill_name, relative_path)` — new tool, confined to `skill.path`, reads on demand |
| `scripts/*` | When the skill's own instructions say to automate something | Existing `run_shell_command(command=..., cwd=skill.path)` — no new tool |
| `assets/*` | Templates/boilerplate the output should be based on | `read_skill_resource` to pull content; agent's existing `write_file` (already workdir-confined) to place a working copy where needed |
| `templates/*` *(new in v2)* | Same treatment as `assets/` — kept as a separate recognized name because skill authors commonly distinguish "reusable boilerplate to copy" (`templates/`) from "reference material to read" (`assets/`) | `read_skill_resource`, same confinement rules |

The manifest line format stays `path — one-line purpose`, generated from the `Skill.references`/`scripts`/`assets`/`templates` lists (all four now first-class fields on `Skill`, populated at parse time by `_parse_skill_md`/`index_single`, content never read until requested).

**Deciding when to read vs. execute** is left to the skill author's own instructions in `SKILL.md` — that's the whole point of the manifest-plus-on-demand-tool design: your orchestrator doesn't need any new judgment logic here, because the skill's own text already tells the agent "read `references/x.md` if you need Y" or "run `scripts/build.py` to do Z." This was already true in v1 and doesn't change.

---

## 7. Multi-skill handling (extended from v1, now live)

v1's rule ("don't cram two skills into one agent, split into a new task with `update_tasks` instead") still applies for genuinely *different* work. But your example — realizing mid-execution that the *same* task also needs a second capability, not a second task — is now directly served by §2B: the agent just calls `request_skill_acquisition` again, from wherever it is in its own tool-use loop.

```
Task: "Create PDF report"
  agent role: report_writer, skill: pdf-generation (acquired at create_agent time, §2A)
  │
  ▼
Agent starts writing content, realizes it also needs chart-image generation
  │
  ▼
Agent calls request_skill_acquisition("generate charts from tabular data")
  │  → runs the SAME ensure_skill() pipeline (§3), returns as a ToolMessage
  │  → on success, refresh_tools() unions the new skill in (§4.12), directive
  │    injected as a HumanMessage (§4.13)
  ▼
Agent continues the SAME task, task_messages unbroken, now has both skills
```

No new task, no new agent, no lost context — `task_messages` just accumulates one more `ToolMessage` + one more `HumanMessage`, exactly like any other mid-task tool call already does. This is strictly additive to v1's task-splitting guidance, not a replacement for it: use `update_tasks` when the *role* itself is wrong for the remaining work (e.g. "I'm a writer but this needs data cleaning first"); use `request_skill_acquisition` when the *role* is still right but it's missing one more capability (e.g. "I'm a report writer and I still am, I just also need chart rendering").

`refresh_tools`'s existing union-not-replace behavior (v1 §9) still applies unchanged — skills accumulate on a role across the task's lifetime, never silently dropped.

---

## 8. Required code changes (by file)

**`skills.py`**
- `Skill`: add `templates: list[str] = field(default_factory=list)` alongside v1's `references`/`scripts`/`assets`.

**`skill_discovery.py`**
- Factor the SKILL.md-parsing body of `index_root` into a reusable `_parse_skill_md(path) -> Skill | None` if not already fully isolated (v1 already assumed this existed for extension — confirm it's callable standalone).
- New: `index_single(skill_dir: Path, source: str = "github") -> Skill | None` — parses one folder, registers it, returns the `Skill` or `None` on parse failure. This is the function `ensure_skill` calls post-verification.

**New file `skill_acquisition.py`**
- `class SkillAcquisitionManager`:
  - `ensure_skill(capability_description, task_description) -> AcquisitionResult` — the orchestrating function, §2–§4.
  - `_check_cache_and_registry(...)` — §4.16.
  - `_search(query) -> list[Candidate]` — §4.2, shells `skills find`.
  - `_rank_and_gate(candidates, task_description) -> Candidate | None` — §4.3.
  - `_install(candidate) -> Path` — §4.4/4.5, shells `skills add`.
  - `_verify(path) -> bool` — §4.9.
  - `_get_lock(key) -> threading.Lock` — §4.14.
  - Small in-memory positive/negative cache — §4.16.

**`dynamic_langgraph_backend.py`**
- `DynamicAgentFactory.create_agent()` / `refresh_tools()`: add `"skill_gap"` to the expected JSON shape and its prompt instructions (sibling to `new_tools`); when present, call `SkillAcquisitionManager.ensure_skill(...)` before finalizing `_apply_skills()`.
- New tool factory (alongside `_read_file_tool`, `_create_missing_tools`, etc.): `_request_skill_acquisition_tool()` → wraps `ensure_skill`, registered as an always-available tool the same way `update_tasks` is bound in `create_agent`/`refresh_tools`'s `all_tool_names`.
- New tool factory: `_read_skill_resource_tool()` → `read_skill_resource(skill_name, relative_path)`, confined to `skill_registry.get_skill(skill_name).path`, covering `references/`, `scripts/`, `assets/`, `templates/`.
- `DynamicAgentFactory.skill_directive()`: append the four-folder resource manifest (v1 §7, now including `templates`).
- `_evaluator_node` RETRY branch: unchanged mechanism, prompt wording updated to mention `request_skill_acquisition` as an available option when feedback indicates a missing capability rather than a wrong approach.
- Structured `_log` calls at each acquisition phase (§5) — no new logging infrastructure, just consistent tagging (`kind="SKILL-ACQUISITION"`, `phase=...`).

**`main.py`** *(not in this snapshot — describe the touch point)*
- No new endpoint required for the live path (that's the point — it's no longer an admin action). `POST /skills/sync` and `POST /skills/reindex` remain, unchanged, for bulk/admin use only.
- If your SSE layer filters `_log` events by kind before forwarding to the frontend, add `"SKILL-ACQUISITION"` to the forwarded set so the phase events actually reach the "Working" disclosure blocks — verify against however that filter is currently implemented.

**`index.html`** — no required change. The v1-proposed "Sync from GitHub" manual button is now optional/secondary (still useful for pre-warming the cache with skills you know you'll need), not part of the critical path.

---

## 9. Testing plan (extends v1 §12, live-path scenarios)

1. **Cold-start acquisition:** ask for a task with zero matching local skills, whose need clearly maps to a real, well-known public skill (e.g. a PDF/report skill). Confirm: `skill_gap` fires → `ensure_skill` runs all phases → `GET /agents` shows the skill attached → the task actually completes using it.
2. **Cache-hit path:** run the same request twice in a row (two separate task threads). Confirm the second run short-circuits at §4.16 — no second subprocess call (check logs/timestamps), same as v1's idempotent-sync test but now triggered live.
3. **Concurrent identical requests:** fire two tasks needing the same missing skill at the same time. Confirm only one install subprocess runs (lock behavior, §4.14) and both tasks end up with the skill attached.
4. **Mid-task self-service (multi-skill):** engineer a task where the agent's first skill is genuinely insufficient partway through; confirm it calls `request_skill_acquisition`, the second skill gets attached without a new task/agent, and `task_messages` shows the injected directive (§4.13) actually reaching the LLM.
5. **No-match search:** a capability with no reasonable public skill available. Confirm `not_found` is returned cleanly, the agent proceeds without crashing, and the negative cache prevents repeat searches within its TTL.
6. **Install failure — timeout:** simulate a hung subprocess (mock or artificially low timeout). Confirm graceful `failed` result, no partial `github_skills/<name>/` left in a state that a later `already_present` check would wrongly accept.
7. **Install failure — malformed package:** hand-craft a "skill" with a missing/broken `SKILL.md` at the expected path (simulate a bad install). Confirm `_verify` rejects it, it's quarantined (`.partial-*`), and it never reaches `SkillRegistry`.
8. **Oversized fetch:** simulate an install that dumps far more files than a single skill should have. Confirm the sanity check in `_verify` rejects it.
9. **Resource pull-through and script execution:** same as v1 §12.3/12.4, now exercised against a *live-acquired* skill rather than a hand-copied one — confirms the acquisition path produces something structurally identical to a pre-seeded one.
10. **Regression:** v1's Phase 1–3 test suite (precedence, malformed-JSON bundled tools, untrusted-skill gating) still passes unchanged for pre-seeded skills that never go through `ensure_skill` at all.

---

## 10. Phased implementation plan

**Phase A — Manager skeleton + manual trigger (no CLI yet)**
Build `SkillAcquisitionManager` with `_search`/`_install` stubbed to operate on a hand-placed folder (simulate acquisition), so `_verify`, `index_single`, cache, and locking can be tested in isolation before the CLI is in the loop at all.

**Phase B — Real `skills find` / `skills add` integration**
Wire the real subprocess calls. Spike the CLI's actual output directory behavior first (§4.4's open item) before finalizing `_install`. Confirm end-to-end against one real, well-known public skill.

**Phase C — Config-time trigger (`skill_gap` in `create_agent`/`refresh_tools`)**
Wire §2A. Test scenario 1 (§9).

**Phase D — Self-service tool (`request_skill_acquisition`)**
Wire §2B, the tool, and the `refresh_tools` + directive-injection follow-through (§4.12/4.13). Test scenario 4.

**Phase E — Resource loading extension**
`read_skill_resource` tool + `templates/` support (§6), reusing v1 Phase 5a/5b design essentially unchanged, now exercised against live-acquired skills.

**Phase F — Failure modes, locking, caching hardening**
Scenarios 3, 5, 6, 7, 8 from §9. This is where most of the actual robustness work lives — get the happy path working end-to-end first (A–E), then harden.

**Phase G — Streaming/UX polish**
Add the `_log` phase tags (§5) and confirm they surface in the existing "Working" disclosure blocks. Purely additive, doesn't gate correctness of anything above.

This ordering gets you a genuinely live, end-to-end demoable pipeline (Phases A–D) before touching failure-mode hardening or UX polish — consistent with your own stated preference for phased, sequentially-confirmed delivery.

---

## 11. Deferred / carried-forward from v1

- **Trust/security:** unchanged decision from v1 §14 — live-acquired skills default to `trust="trusted"` for this phase, same as v1's synced skills did, with the same explicit acknowledgment that this is a deliberate, temporary widening of attack surface now that it's *fully automatic* rather than admin-reviewed. Flag this more prominently than v1 did, precisely because a human is no longer in the loop before code from an arbitrary GitHub repo gets executed — revisit before anything production-facing.
- **Multi-process lock upgrade** (§4.14 caveat) — filesystem-based lock instead of in-memory, only needed if you scale beyond one backend worker process.
- **Node/`npx` vs. `skillsmd` decision** (§4.2) — confirm which is actually available in your runtime before writing `_search`/`_install`.
- **Exact CLI target-directory behavior** (§4.4) — spike before finalizing, may require the download-API fallback instead of `skills add` directly.
