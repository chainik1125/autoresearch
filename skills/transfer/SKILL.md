---
name: transfer
description: Take an existing research experiment and run it under a changed parameter (different model, different dataset, different domain, etc.) on cloud GPU. Opens with a brief intent-discovery conversation, recommends hardware automatically (with a chance for the user to override on borderline cases), then dispatches. Use whenever the user wants to re-run a measurement under a varied condition.
---

# /transfer — re-run an existing experiment under a changed condition

The user has a working measurement on some setup and wants to see what it
does in a related setup — same code on a different model, same model on a
different dataset, same dataset with a different measurement, etc.
"TRANSFER" is the broad category of "run the thing again but vary one axis."

Your job has four phases. Move through them in order; **don't dispatch
before completing Phase 1 and 2**, and **don't run make_compatible before
completing Phase 0** unless the call is unambiguous (see the skip rule).

---

## Phase 0 — Intent discovery

**Cap: 5 follow-up questions, total.** Get to clarity efficiently or
declare you have enough to proceed.

**Skip Phase 0 entirely if ALL of these hold:**
- The user explicitly named both the **pipeline** and the **changed value**
  (e.g. `/transfer sae_training Qwen/Qwen2.5-7B-Instruct`)
- A Pipeline class with that name already exists in the project's
  `pipelines/` directory (check by listing files locally)
- The project has an `autoresearch.toml`
- Nothing in the user's request suggests a non-obvious change

If skipping, jump straight to Phase 1 (hardware).

**Otherwise, conduct the conversation.** You're trying to fill in five
fields. Ask only about the ones you can't infer from context:

| Field | What you're trying to know |
|---|---|
| **One-sentence summary** | "You want to take X and run it on Y, right?" Restate to confirm. |
| **Axis of variation** | What's changing? Model / dataset / measurement / hyperparam / domain. |
| **What stays fixed** | What's the baseline they're comparing against? |
| **Prerequisites** | Are there resources needed that don't exist yet? (e.g. an SAE for the new model, a dataset for the new domain.) |
| **Intent for the run** | "Quick canary" / "production paper run" / "exploratory sweep, optimize for cost". Used for the hardware recommendation. |

Prefer to **infer aggressively** from what they said and the project state
you can read locally. Each unnecessary question is friction.

**Output of Phase 0**: score the fit as one of three, and tell the user:

- **`fit`** — the project already has a Pipeline class that takes this exact
  parameter shape; just dispatch.
- **`fit-with-adapter`** — the project's measurement exists but isn't yet
  wrapped as a Pipeline class. Run `make_compatible` next.
- **`mismatch`** — autoresearch isn't the right tool for this (e.g. needs
  prerequisite work first, or it's not a re-run at all). Advisory, not
  gating — the user can override with "do the best you can with it."

---

## Phase 1 — Hardware recommendation

**This phase exists because GPU selection used to be the most common source
of dispatch friction.** The MCP exposes `recommend_hardware` for exactly
this — call it before `start_transfer` so the user sees the proposed
choice and confidence level.

### What to call

Pull `required_vram_gb` from the Pipeline class (or estimate it from the
target model — Qwen-7B ~30GB, Qwen-14B ~50GB, Qwen-32B ~80GB working
set), then:

```
recommend_hardware(
    required_vram_gb = <from Pipeline.required_vram_gb or estimate>,
    intent           = <the intent string from Phase 0>,
    pipeline_name    = <pipeline name>,
    estimated_minutes = <from Pipeline.estimated_minutes if known>,
)
```

Returns `{picks, confidence, rationale, alternatives}`.

### Branch on confidence

**`confidence == "high"`** — just narrate the choice and proceed:

> "I'll dispatch on `<picks[0]>` (${X}/hr): <rationale>. Hit go or tell me
> to override."

Then call `start_transfer(... gpu=picks)`.

**`confidence == "needs_review"`** — surface the trade-off and ask:

> "Hardware needs a sanity check: <rationale>. Options:
>   - `<picks[0]>`  → fast, $X/hr
>   - `<alternatives[0]>` → cheaper, $Y/hr but slower
>   - `<alternatives[1]>` → ...
> What would you prefer?"

After the user answers, call `start_transfer(... gpu=[<user-choice>, ...fallbacks])`.

**`picks == []`** — no GPU fits. Tell the user the constraint and ask what
to relax (DC, VRAM floor, wait for inventory).

### When to skip Phase 1

- The user explicitly passed a GPU choice in their request (`/transfer ...
  on L40S`). Use that directly.
- The Pipeline class has no `required_vram_gb` and the user gave no hint.
  Skip; the dispatcher falls back to `settings.default_gpu`.

---

## Phase 2 — Readiness check

Verify locally that everything is in place to dispatch. Stop at the first
failing check.

### 2a. Project structure

```
! ls autoresearch.toml pipelines/ 2>/dev/null
```

- `autoresearch.toml` present?
- `pipelines/` directory present?
- A pipeline class with the matching `name` attribute present?

If any is missing AND fit-score is `fit`, re-score as `fit-with-adapter`
and run `make_compatible`.

### 2b. Git remote check

```
! cd <project-dir> && git remote -v
```

If no remote, tell the user: "Your project needs a git remote before
autoresearch can dispatch. v0.1 doesn't support tarball uploads."

### 2c. Required env vars (private repos only)

If `project_repo_url` points at a private repo, the controller needs a
GitHub PAT. Discovery chain:

1. **User's shell env**: `! printenv | grep -iE '^(GIT_PAT|GITHUB_TOKEN|GH_TOKEN)='`
2. **Conversation context**: any PAT variable the user has mentioned.
3. **Ask for the env-var name only** (never the token value):
   > "I need a GitHub PAT with `repo` scope. What env var holds it?
   > Paste the *name*, not the value."

Public repos: skip this whole step.

### 2d. Storage tier reminder (advisory)

For pipeline authors:
- **R2** (`storage` arg) → small structured outputs only (per-layer scores,
  summary stats). Never large tensors.
- **Workspace** (`workspace` arg, the persistent volume) → everything heavy.
- **HF Hub** → raw inputs.

---

## Phase 3 — Dispatch

By Phase 3 you have:
- A pipeline name that exists in the project
- `params` (at minimum `target_model`; everything else pipeline-specific)
- A hardware choice from Phase 1 (either `picks` list or user override)
- A budget (default from `autoresearch.toml`)
- Optional: `project_repo_token`, `project_repo_branch`

Call the MCP tool:

```
start_transfer(
    pipeline_name      = "...",
    target_model       = "...",                    # convenience shorthand, goes into params
    source_model       = "...",                    # optional convenience shorthand
    params             = {...},                    # arbitrary pipeline-specific params
    gpu                = picks,                    # from recommend_hardware OR user override
    required_vram_gb   = <num>,                    # optional; lets the controller re-rank
                                                   #   if `gpu` list goes dry
    intent             = "...",                    # from Phase 0 — useful if the
                                                   #   controller falls back to recommend()
    budget_usd         = ...,                      # optional
    project_repo_url   = "...",                    # ALWAYS pass from autoresearch.toml
    project_repo_branch = "...",                   # pass when make_compatible made a branch
    project_repo_token = "...",                    # optional; private repos
)
```

**Always pass `project_repo_url`** from the project's `autoresearch.toml`.
If you omit it, the pod inherits the controller's default (typically stale).

Returns `{run_id, status, pod_handle}`. Confirm to the user:

> "Dispatched run `<id>` on a `<gpu>` pod. First boot ~5-15 min if HF
> model download is fresh. Check in with `summarize_run <id>` or
> `list_findings <id>` anytime; you don't need to keep this session open."

---

## Mid-flight check-ins

After dispatch the user may ask "how's it going?" later. MCP tools:

- `summarize_run(run_id)` — LLM-summarized digest. Costs Anthropic tokens.
- `list_findings(run_id)` — raw structured findings, oldest first.
- `tail_log(run_id, lines=200)` — recent log chunks.
- `get_run(run_id)` — full Run record.
- `takeover(run_id)` — pause + return SSH command for the pod.
- `release(run_id)` — resume from takeover.
- `cancel(run_id)` — terminate pod, mark failed.

---

## Sharp edges to know

### Budget enforcement is advisory

The runner checks budget before each FSM step and hard-stops after the
step that crosses cap. In-flight tokens or pod-seconds can push past.
**Don't promise hard limits** when reporting budget back.

### First-run-on-a-new-model is slow

HF downloads happen on first use per network volume. Qwen-32B is ~60GB,
Qwen-7B ~14GB. User should expect 5-15 min before training even starts on
a brand-new volume. Subsequent runs are warm.

### Takeover is not instant

`takeover` is effective at the next FSM boundary (between phases), not
mid-step.

### When NOT to use this skill

- **Writing a new pipeline from scratch**: write the Pipeline class first,
  then come here.
- **Interactive debugging**: SSH into a pod via `takeover` directly.
- **Replicating someone else's paper**: that's the REPLICATE workflow,
  deferred from v1.
- **A "what if" question without a measurement attached**: shape the
  experiment manually first.
