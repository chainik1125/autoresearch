---
name: transfer
description: Take an existing research experiment and run it under a changed parameter (different model, different dataset, different domain, etc.) on cloud GPU. Opens with a brief intent-discovery conversation to make sure the experiment is a good fit for the autoresearch pipeline, then either dispatches it as-is, adapts the user's project to fit the Pipeline protocol first, or honestly advises that this isn't a good autoresearch use case. Use whenever the user wants to re-run their measurement under a varied condition.
---

# /transfer — re-run an existing experiment under a changed condition

The user has a working measurement on some setup and wants to see what it
does in a related setup — same code on a different model, same model on a
different dataset, same dataset with a different measurement, etc.
"TRANSFER" is the broad category of "run the thing again but vary one axis."

Your job has three phases. Move through them in order; **don't dispatch
before completing Phase 1**, and **don't run make_compatible before
completing Phase 0** unless the call is unambiguous (see the skip rule).

---

## Phase 0 — Intent discovery

**Cap: 5 follow-up questions, total.** Get to clarity efficiently or
declare you have enough to proceed.

**Skip Phase 0 entirely if ALL of these hold:**
- The user explicitly named both the **pipeline** and the **changed value**
  (e.g. `/transfer fra_em_steering Qwen/Qwen2.5-32B`)
- A Pipeline class with that name already exists in the project's
  `pipelines/` directory (check by listing files locally)
- The project has an `autoresearch.toml`
- Nothing in the user's request suggests a non-obvious change
  (e.g. they don't mention "but I want it to use a different dataset" —
  that's an extra axis you'd need to ask about)

If skipping, jump straight to Phase 2 (Dispatch).

**Otherwise, conduct the conversation.** You're trying to fill in five
fields. Ask only about the ones you can't infer from context:

| Field | What you're trying to know |
|---|---|
| **One-sentence summary** | "You want to take X and run it on Y, right?" Restate to confirm. |
| **Axis of variation** | What's changing? Pick one or more: model / dataset / measurement / hyperparam / domain. |
| **What stays fixed** | What's the baseline they're comparing against? (Often implied by "I have results on X" — they want to know if it reproduces on Y.) |
| **Prerequisites** | Are there resources needed that don't exist yet? (e.g. an SAE for the new model, a dataset for the new domain, a finetune that doesn't exist for the new base.) If yes — that's research work the user has to do, not autoresearch's job. |
| **Success criterion** | What does "this worked" look like? A number to match? A qualitative behavior? Just exploratory? Helps frame the result. |

Prefer to **infer aggressively** from what they said and the project state
you can read locally. Each unnecessary question is friction. The 5-question
cap is a ceiling, not a floor.

**Output of Phase 0**: score the fit as one of three, and tell the user:

- **`fit`** — the project already has a Pipeline class that takes this exact
  parameter shape; just dispatch. *(Most common when skipping Phase 0.)*
- **`fit-with-adapter`** — the project's measurement exists but isn't yet
  wrapped as a Pipeline class, OR the existing Pipeline doesn't expose the
  axis the user wants to vary. Run `make_compatible` next.
- **`mismatch`** — autoresearch isn't the right tool for this. Examples:
  - Prerequisites missing (need to train a new SAE first, no LoRA for the
    new base model). Recommend they do the prerequisite manually first.
  - The experiment requires interactive iteration (debugging, deciding what
    to do next based on intermediate results). autoresearch dispatches
    deterministic runs; interactive work belongs in a regular notebook.
  - It's not really a re-run — it's a fresh experiment design. Recommend
    they prototype manually, then come back when they have a working version
    to transfer.

**Advisory, not gating.** A `mismatch` score is your recommendation. The
user can override with something like "do the best you can with it." If they
do, treat it as a `fit-with-adapter` and proceed; flag the concerns as
findings the eventual run report should surface.

---

## Phase 1 — Readiness check

Once Phase 0 has scored the fit, verify locally that everything is in place
to actually dispatch. Run these checks in order; stop at the first one that
fails and resolve before proceeding.

### 1a. Project structure

In the user's project directory:

```
! ls autoresearch.toml pipelines/ 2>/dev/null
```

- `autoresearch.toml` present?
- `pipelines/` directory present?
- A pipeline class with the matching `name` attribute present?
  (You can check by inspecting files in `pipelines/` for `name = "..."` lines.)

If any of these are missing AND fit-score is `fit`, that's a contradiction —
re-score as `fit-with-adapter` and run `make_compatible`. If fit-score is
`fit-with-adapter`, run `make_compatible` now.

### 1b. Git remote check

The pod clones from `project_repo_url`. The repo needs a remote that the
user has push access to (for any branch the bridge skill creates):

```
! cd <project-dir> && git remote -v
```

- Remote configured? If not — autoresearch can't dispatch this project as-is.
  Tell the user: "Your project needs a git remote (e.g., GitHub) before
  autoresearch can dispatch it. v1 doesn't support tarball uploads."

### 1c. Required env vars

For private project repos, the controller needs `AUTORESEARCH_PROJECT_REPO_TOKEN`.
If the controller already has one set (check by reading user's `autoresearch.toml`
for any indication, OR just try to dispatch and handle the failure), no
work. If not, discover and pass per-dispatch using the chain:

1. **User's shell env**: try common names from `~/.zshenv` / `~/.bashrc` /
   current env:
   ```
   ! printenv | grep -iE '^(GIT_PAT|GITHUB_TOKEN|GH_TOKEN|GITHUB_PAT)='
   ```
   If found, you'll pass its value as `project_repo_token` in the
   `start_transfer` call. **Do not echo the token value in your reply** —
   confirm "found a PAT in env" and proceed.

2. **Conversation context**: scan this conversation for any PAT-like string
   the user may have mentioned (e.g. "my token is in `$MY_PAT`"). Use the
   variable name they reference. **Do not paste any literal token value back
   to the user.**

3. **Ask, but only for the env var name** — never ask for the token value
   directly:
   > "I need a GitHub PAT with `repo` scope to clone your private project.
   > What env var holds it? (e.g. `GIT_PAT`, `GITHUB_TOKEN`) — paste the
   > variable *name*, not the value."
   Then `! printenv $NAME` and pass through.

Public repos: skip the whole step.

### 1d. Storage tier reminder (advisory)

This is for the pipeline author's awareness — useful to mention if the user
is writing or adapting a pipeline:

- **R2** (`storage` arg in `Pipeline.run`) → small structured outputs only.
  Per-layer scores, summary stats. Never large tensors or model dumps.
- **Workspace** (`workspace` arg, on the persistent volume) → everything
  heavy. HF caches happen automatically via `HF_HOME`. Pipeline can also
  write intermediate state here.
- **HF Hub** → raw inputs. `from_pretrained` lands caches on the volume
  on first miss.

---

## Phase 2 — Dispatch

By the time you reach Phase 2 you have:
- A pipeline name that exists in the project's `pipelines/` (or that
  `make_compatible` just created)
- Identified `params` (at minimum `target_model`, optionally `source_model`,
  and any others from the conversation)
- A budget (default from `autoresearch.toml`, or per-call)
- A GPU class (default from `autoresearch.toml`, or per-call)
- Optional: `project_repo_token`, `project_repo_branch` from the readiness
  check

Call the MCP tool:

```
start_transfer(
    pipeline_name = "...",
    target_model  = "...",          # convenience shorthand; goes into params
    source_model  = "...",          # convenience shorthand; goes into params
    params        = {...},          # arbitrary pipeline-specific params
                                    #   (hook_name, dataset, training_tokens, etc.)
                                    #   This is the general way — use whenever the
                                    #   pipeline takes something beyond target_model.
    gpu           = "...",          # optional, overrides default
    budget_usd    = ...,            # optional, overrides default
    project_repo_token = "...",     # optional
    project_repo_branch = "...",    # optional, from make_compatible
)
```

**On `params` vs the convenience args**: Phase 0 may have identified an axis
of variation that isn't "the model" — e.g., hookpoint, dataset, hyperparam.
Pass those keys via `params`. The Pipeline reads them in its `run()`. The
controller's MCP API does not have an opinion about which keys are "real."

Returns `{run_id, status, pod_handle}`. Confirm to the user:

> "Dispatched run `<id>` on a `<gpu>` pod. First boot will take ~5-15 min
> (HF model download for first-time use on this volume). You can check in
> with `summarize_run <id>`, `list_findings <id>`, or `tail_log <id>` from
> here whenever you want — you don't need to keep this session open."

---

## Mid-flight check-ins

After dispatch, the user may ask "how's it going?" later. Available MCP tools:

- `summarize_run(run_id)` — LLM-summarized digest of progress + findings.
  Costs Anthropic tokens but is the cleanest single-call status check.
- `list_findings(run_id)` — raw structured findings, oldest first.
- `tail_log(run_id, lines=200)` — recent log chunks from the pod.
- `get_run(run_id)` — full Run record (status, budget, pod handle).
- `takeover(run_id)` — pause at next FSM boundary, get SSH command. Use when
  the user wants to debug or inspect the pod directly.
- `release(run_id)` — resume from takeover.
- `cancel(run_id)` — terminate pod, mark failed. Use when the user wants to
  stop a run mid-flight.

---

## Sharp edges to know

### Budget enforcement is advisory

The runner checks budget before each FSM step and hard-stops after the step
that crosses cap. In-flight tokens or in-flight pod-seconds can push past.
**Don't promise hard limits** when reporting budget back to the user.

### First-run-on-a-new-model is slow

HF downloads happen on first use per network volume. Qwen-32B is ~60GB.
The user should expect 5-15 min before the pipeline even starts running on
a brand-new volume. Subsequent runs on the same volume are warm.

### Takeover is not instant

`takeover` is effective at the next FSM boundary (between phases), not
mid-step. If the pipeline is mid-`pipeline.run()`, the tool call can't be
paused — only the next checkpoint transition will honor the takeover.

### When NOT to use this skill

- **Writing a new pipeline from scratch**: skip this skill, write the
  Pipeline class first, *then* come here.
- **Interactive debugging**: SSH into a pod via `takeover` directly, not
  via `/transfer`.
- **Replicating someone else's paper**: that's the REPLICATE workflow,
  deferred from v1.
- **A "what if" question that doesn't have a measurement attached yet**:
  shape the experiment manually first.
