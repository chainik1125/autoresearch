# Pre-dispatch user-input checklist

This is the fast checklist the local `/transfer` skill scans **before**
dispatching the prep agent. The point is to catch the things only the user
can answer — auth, scope, budget, ambiguous defaults — in a quick turn so
the user can disengage their laptop ASAP.

**Local Claude does NOT read project code in detail.** That's the prep
agent's job (off-laptop, takes minutes, can read everything). Local Claude
runs this checklist in seconds.

Each item below has:
- a **Check** (deterministic — file read, env grep, params inspect)
- a **Ask** (only fired when the check surfaces ambiguity)

Iterate on this list as new "should have asked the user" cases bite.

---

## 1. Auth — gated model weights

- **Check:** does `target_model` look like a known gated repo (Llama-*, Mistral-*,
  some Qwen variants under license)? Read `~/.zshenv` for `HF_TOKEN` or check
  Railway's env (the controller forwards it to pods).
- **Ask:** "This model may require an HF token. I see `HF_TOKEN` in your
  shell env — using that. (Or pass a different name?)"

## 2. Auth — private project repo

- **Check:** is `project_repo_url` a private GitHub URL? (Check by attempting
  `git ls-remote` without auth, OR just ask if `/.../private` looks private.)
- **Ask:** "Your project repo looks private. I'll forward `GIT_PAT` from
  your shell as `project_repo_token`. OK?"

## 3. Auth — WandB logging

- **Check:** does the pipeline's defaults enable wandb? (read its
  `pipelines/<name>.py` for `wandb_*` references, or check `params`).
- **Ask:** "WandB logging looks enabled. Using `WANDB_API_KEY` from env.
  Project will land in `<entity>/<project>`. Override?"

## 4. Hardware confidence

- **Check:** `recommend_hardware(...)`'s `confidence` field. If `"high"`,
  no question. If `"needs_review"`, escalate.
- **Ask:** surface the rationale + alternatives so the user picks.

## 5. Budget cap

- **Check:** estimated cost from the pipeline's `estimated_minutes` ×
  the recommended GPU's `$/hr` + an LLM-overhead allowance. Compare to
  `settings.default_budget_usd` (today $30).
- **Ask:** "Estimated total ~$X for this run (compute + LLM). Default
  cap is $30; bump to $Y or proceed?"

## 6. Branch destination

- **Check:** is there an `autoresearch/prepared-*` branch already on the
  remote for this pipeline+target? (May indicate a prior prep pass.)
- **Ask:** "I'll dispatch on `<branch>`. (If you want a different branch
  or a fresh prep pass on `main`, tell me.)"

## 7. Backend choice (when pipeline has multiple)

- **Check:** does the Pipeline class expose a `backend` param? (Read
  the class for an `_BACKENDS` tuple or similar.)
- **Ask:** "This pipeline has backends `<list>`. Which one? (default:
  `<first>`)"

## 8. First-time-on-this-model warning

- **Check:** has this `<volume, target_model>` pair been seen before? (Look
  for `<volume>/<...>/.huggingface/hub/models--<model_slug>` via a
  cheap controller-side check, if available; otherwise assume first time
  if no prior Run with this target_model exists.)
- **Ask (heads-up only — no answer expected):** "First time on
  `<target_model>` for this volume — expect a ~`<size>`GB download on
  pod boot."

## 9. Dataset access

- **Check:** does the pipeline reference datasets the pod may not have
  access to (gated HF, S3-private)? Read `params` for known fields like
  `dataset_path`, `dataset_url`.
- **Ask:** "Pipeline uses `<dataset>`. Public — should be fine. (Or:
  this is gated; HF token will be needed.)"

## 10. Pipeline-specific override surface

- **Check:** does the Pipeline class have any `required_*` attributes
  beyond `required_gpu` / `required_vram_gb`? (Future-proofing — today
  there aren't any others.)
- **Ask:** N/A for v0.

## 11. Volume disk usage (advisory)

- **Check:** none locally — the pod runs `disk_preflight` on boot and writes
  a finding with `df -h /workspace`. You don't need to ask the user about
  this in advance.
- **Ask:** N/A. But: if the user reports an earlier run failed with the
  message `DISK PREFLIGHT FAILED` (or you see one in `list_findings`),
  tell them the prep agent can prune the HF cache for models already
  durably stored on HF Hub — it has authority to do that. Re-dispatch
  prep and it'll clean up before the next transfer attempt.

---

## What this checklist intentionally does NOT cover

These belong to the **prep agent**, not the local conversation:

- Hardcoded `/root/` / `/home/` / `/Users/` paths in the user's code
- Missing system deps that pip will catch
- Quirky module-level side effects on import
- Project-specific env var names (`OPENAI_API_KEY_MATS` vs `OPENAI_API_KEY`)
- Any "read and judge whether this is correct" task
- Any "edit and commit" task

The local skill's role: **decide what to ask, ask it, move on.** Time
budget: 30-90 seconds total for the checklist. If you find yourself
about to grep through someone's pipelines/ directory, stop — that's
prep agent work.
