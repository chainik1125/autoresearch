---
name: transfer
description: Run an existing measurement pipeline on a different model (e.g. "FRA on Qwen-32B"). Dispatches a RunPod pod via the autoresearch controller and streams findings back. Use when the user wants to take a working pipeline they already have and run it against a new model.
---

# /transfer — run a pipeline against a new model

You're orchestrating a transfer: the user has a Pipeline class they've already
written (in their project's `pipelines/` directory) and wants to run it on a
different model than they last ran it on. The autoresearch controller (running
on Railway) dispatches a GPU pod, the runner executes the pipeline, and findings
stream back to R2.

## Workflow

1. **Identify the pipeline.** Call the MCP tool `list_pipelines` to see what's
   available. If the user named one (`/transfer fra_measurement qwen-32b`),
   confirm it's in the list. If not, ask which pipeline.

   **If `list_pipelines` returns empty or the named pipeline isn't found**:
   the project hasn't been adapted yet. Switch to the `make_compatible`
   sub-skill (TODO: not yet implemented) which creates a branch in the user's
   project repo and wraps their existing measurement code as a Pipeline class.
   After the branch is pushed, come back here with the new branch name and
   pass it as `project_repo_branch` to `start_transfer`.

1a. **Private-repo PAT discovery.** If the user's project repo is private
   (or might be), the pod needs a GitHub PAT to clone it. Resolve in order:

   a. **Existing controller config**: if `start_transfer` succeeds without a
      `project_repo_token` arg, the controller already has one set. Done.

   b. **User's shell env**: try common names from `~/.zshenv` /
      `~/.bashrc` / current env:
      ```
      ! printenv | grep -iE '^(GIT_PAT|GITHUB_TOKEN|GH_TOKEN|GITHUB_PAT)='
      ```
      If found, pass its value as `project_repo_token`. **Do not echo the
      token value in your reply** — confirm "found a PAT in env" and proceed.

   c. **Conversation context**: scan this conversation for any PAT-like
      string the user may have mentioned (e.g. they said "my token is in
      `$MY_PAT`"). Use it if applicable. **Do not paste any literal token
      value back to the user.**

   d. **Prompt the user**: ask them what env var their PAT lives in (not
      the token value itself — keep it env-resident):
      > "I need a GitHub PAT with `repo` scope to clone your private
      > project. What env var holds it? (e.g. `GIT_PAT`, `GITHUB_TOKEN`)
      > Or paste the variable *name*, not the value."
      Then `! printenv $NAME` and pass through.

   For public repos, skip the whole step — `start_transfer` works without
   a token.

2. **Confirm the model arguments.**
   - `target_model` — the model to run against (e.g. `Qwen/Qwen2.5-32B`). Use
     full HuggingFace identifiers when possible — the pod loads via `from_pretrained`.
   - `source_model` (optional) — the model whose existing result we'll compare
     against in postflight validation. If the user has prior runs for the same
     pipeline, suggest using one as source. Skip if there's nothing meaningful
     to compare to.

3. **Storage tier reminder.** Quietly check the project's pipeline expects the
   tier conventions (you can ignore this if the user knows what they're doing):
   - **R2** (`storage` arg in `Pipeline.run`) only for structured outputs you
     want to compare across runs — scores, summaries.
   - **Workspace** (`workspace` arg, mounted on the persistent volume) for
     everything heavy — model weights cache via `HF_HOME`, datasets,
     intermediate tensors.
   - **HF Hub** for raw inputs — `from_pretrained` will land caches on the
     volume automatically.

4. **Pick a budget.** Default to whatever is in `autoresearch.toml`
   (`default_budget_usd`). If the run is expected to take many hours, suggest a
   higher cap so it doesn't stop mid-pipeline. Caps are advisory — the runner
   hard-stops at the next FSM step after spend crosses the cap.

5. **Dispatch.** Call the MCP tool `start_transfer` with the pipeline name,
   target model, optional source model, optional `gpu` override, and
   `budget_usd`. The tool returns a `run_id` and the pod handle.

6. **First-run on a new model takes longer.** HF downloads cache to the
   persistent volume the first time; subsequent runs on the same volume are
   warm. Tell the user the first pod boot will likely be 5-15 min before the
   pipeline even starts running, depending on model size.

7. **What to do next.** Tell the user how to check in:
   - `summarize_run(run_id)` — server-side LLM digest of progress + findings
   - `list_findings(run_id)` — raw structured findings
   - `tail_log(run_id)` — most recent log lines
   - `takeover(run_id)` — pause at next FSM boundary, get SSH command
   - `release(run_id)` — resume after takeover
   - `cancel(run_id)` — terminate pod, mark failed

   If they want to be notified when it finishes, suggest they ask Claude Code
   to poll `get_run(run_id)` periodically.

## When NOT to use this skill

- The pipeline doesn't exist yet — the user wants you to *write* one. Skip this
  skill; write the Pipeline class in their `pipelines/` directory first.
- The user wants to debug a broken pipeline — they should SSH into a pod
  directly via takeover, not start a new run.
- The user wants to replicate a paper (write a pipeline from scratch given a
  paper + claim) — that's the REPLICATE workflow, deferred from v1.

## On budget honesty

Budget enforcement is advisory. In-flight LLM tokens and pod-seconds can push
past the cap. The runner checks budget before each FSM step and stops after the
step that crosses the cap; it cannot interrupt mid-step. Don't promise hard
limits when reporting budget back to the user.
