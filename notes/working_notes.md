# Working notes

Living document for in-progress autoresearch work. Distinct from
`how-it-works.md` (stable reference) and `ideas.md` (broader roadmap).

## Current focus (2026-05-12)

**Building the bridge layer for `/transfer`.** Going through the FRA use case
(Qwen-14B → Qwen-32B, from `chainik1125/fra_proj` on the `dmitry-em-repl`
branch) showed that the Pipeline protocol can't be a one-size-fits-all
interface for arbitrary user code — the gap should be bridged by a Claude
that adapts the user's existing project on the fly, not by forcing every
project to be manually restructured to fit the protocol.

**Approach (Path B):** Write a generic `templates/skills/make_compatible.md`
skill that instructs the local Claude how to adapt *any* project to
autoresearch's Pipeline protocol — create a branch, wrap the existing entry
point as a Pipeline class, parameterize hardcoded values, drop in
`autoresearch.toml`, check for missing env vars. Then test it on fra_proj as
the first real case and refine when we find gaps.

The `/transfer` skill is updated to detect-then-bridge-then-dispatch: ask the
controller `is_transfer_ready(project_repo_url, branch)`; if ready, dispatch;
if not, invoke `make_compatible.md`.

## TODOs

### Near-term (block FRA case)

- [ ] `templates/skills/make_compatible.md` — project-agnostic bridge skill
- [ ] `templates/skills/setup_check.md` — env-var verification
- [ ] MCP tool `is_transfer_ready(project_repo_url, branch)` on the controller
- [ ] Rewrite `templates/skills/transfer.md` to: detect → bridge → setup_check → dispatch
- [ ] Pod entrypoint: support `PROJECT_REPO_BRANCH` so dispatched pods can clone
      the bridge-created branch instead of the default branch
- [x] **Private-repo cloning support.** (2026-05-12, commit `547761d`)
      `Settings.project_repo_token` + per-dispatch override via
      `start_transfer(project_repo_token=...)`. `core/secrets.py` emits
      `PROJECT_REPO_TOKEN` into pod env when set. `docker/entrypoint.sh`
      rewrites the clone URL just-in-time with basic-auth, redacts the
      token in logs, unsets the auth-bearing URL after clone, uses
      `GIT_TERMINAL_PROMPT=0` to fail-fast. PAT-discovery fallback chain
      documented in `templates/skills/transfer.md` (controller default →
      user shell env grep → conversation scan → prompt for env var
      name). Also added `PROJECT_REPO_BRANCH` propagation for the bridge
      skill to clone a specific branch. *Needs controller redeploy
      (`railway up`) to pick up.*
- [ ] First end-to-end test: bridge Claude adapts `fra_proj` on a branch,
      `/transfer` dispatches against that branch, FRA runs on Qwen-32B.

### Long-term

- [ ] **Generalize the `start_transfer` params model — Phase 2.** *(Surfaced
      while wiring fra_proj's SAE training pipeline 2026-05-12.)* The MCP
      signature hardcodes `target_model` + `source_model` as the only varying
      axes; Phase 1 (in flight as of this entry) adds an additive
      `params: dict` to unblock pipelines that vary other things (hook_name,
      hook_layer, training_tokens, etc.). Phase 2 is the proper refactor:
        - Make `params: dict` the *primary* arg; drop or deprecate the rigid
          target_model/source_model args.
        - Add `baseline_params: dict | None` so the postflight validator can
          compare *any* axis. e.g., dataset-swap → baseline_params =
          {"dataset": <old>}; SAE-hookpoint swap → baseline_params =
          {"hook_name": <old>}.
        - Refactor `workflows/transfer.py`'s postflight hook to look up the
          baseline result by `baseline_params` rather than the current
          "look up source_model" semantics.
        - Update `cli.py:cmd_run` to take arbitrary `--params` JSON without
          requiring `--target-model` / `--source-model`.
      Why it matters: lets Claude's intent-discovery resolve any axis of
      variation into a flat params dict, instead of the MCP API having an
      opinion about which axis is "real." Aligned with the "use Claude's
      intelligence flexibly" goal.
- [ ] **Non-repo / tarball-upload path.** Users whose project has no git
      remote (or no push access) currently can't use `/transfer` at all.
      Support: package project as a tarball, upload to R2, pod fetches +
      extracts instead of `git clone`. ~Half a day of work; not urgent
      because every research repo we care about has a remote.
- [ ] Multi-pass `make_compatible`: bridge may need follow-ups (first attempt
      fails at `import`; retry with `sys.path` tweak). May want a small
      state machine or natural-language loop.
- [ ] Per-workflow bridge variants. `REPLICATE` and `SWEEP` will need
      different adaptation logic than `TRANSFER`. Either one skill per
      workflow or a meta-skill that branches at the top.
- [ ] `compare_runs(id_a, id_b)` MCP tool — diff result dicts + LLM-summarize
      the delta. ("Does Qwen-32B reproduce Qwen-14B's Δalign of +37.7?")
- [ ] Track non-Anthropic LLM spend. fra_proj's judge calls GPT-4o; that cost
      is invisible to autoresearch's budget primitive today.

## Decisions log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-12 | Path B (generic skill first, then test on fra_proj) over Path A | The artifact we actually want is the reusable skill; doing it concretely on fra_proj first risks producing fra_proj-shaped code rather than a general skill. |
| 2026-05-12 | Require git remote; defer tarball upload | Universal for research repos; tarball is real complexity for a v1 edge case. |
| 2026-05-12 | The bridge runs as the same local Claude that called `/transfer` | No new agent type. The local Claude already has Read/Edit/Bash on the project; just give it instructions. |
| 2026-05-12 | "Wrap, don't restructure" is the make_compatible default | Less invasive on user code. Fall back to refactor only if wrap is impossible. |

## Open questions

- **Is `chainik1125/fra_proj` public or private?** Determines whether
  private-repo cloning is a near-term blocker for the FRA case or just a
  forward-looking TODO.
- **Branch naming convention.** Lean `autoresearch/<workflow>-<target-slug>`
  (e.g. `autoresearch/transfer-qwen2.5-32b`) for grep-ability. Open to
  including a date or run-id prefix.
- **Push semantics for the bridge.** After it commits the wrapper to the new
  branch, should it `git push` automatically (after one explicit user
  confirmation), or always require the user to push themselves? Lean
  automatic-after-confirmation — fewer steps, still gated.

## Failure modes from the Qwen-32B SAE canary loop (2026-05-13)

Working through `sae_training` on Qwen-32B end-to-end exposed nine separate
failure modes plus two missing debugging affordances. Each one masked the next,
so the loop was: dispatch, fail silently, fix one thing, redispatch, hit the
next. Total: 13 canaries before canary 13 completed cleanly. Writing this up
because future pipelines will hit subsets of these.

### The bugs (in the order they surfaced)

1. **`pandas==3.0.2` clashes with `transformer-lens` on Python 3.11.** The
   project's `requirements.txt` was produced by `uv pip freeze` on a Mac, where
   nothing exercised transformer-lens's version constraint. On the pod's Linux
   Python 3.11, transformer-lens 2.18 requires `pandas<2.1` and pip's resolver
   gave up. Fix: relax pin to `pandas<2.1`. **Lesson:** a Mac-side
   `uv pip freeze` is not a portable manifest — transitive deps can pin to
   conflicting upper bounds the host venv doesn't trigger.

2. **`torch==2.11.0` upgraded without matching `torchvision` / `torchaudio`.**
   The runpod/pytorch base image ships torchvision 0.19.1 pinned to torch 2.4.1.
   `pip install -r requirements.txt` upgraded torch to 2.11 but left
   torchvision alone, producing an ABI mismatch (`operator torchvision::nms
   does not exist`) that crashed `import sae_lens` at startup. Fix: add
   `torchvision` and `torchaudio` (unpinned) so pip picks compatible versions.
   **Lesson:** if the project pins torch above the base image's version,
   torchvision and torchaudio must move in lockstep — they can't be left to
   the base image.

3. **`LoggingConfig(wandb_run_name=...)` wrong kwarg.** sae-lens 6.39 renamed
   it to `run_name`. The script hit a clean TypeError at config build. Fix:
   rename kwarg. **Lesson:** sae-lens release notes are sparse on rename
   commits; verify config kwargs against the installed version, not docs.

4. **`WORKSPACE_DIR` not injected into pod env.** `cli.py` reads
   `os.environ.get("WORKSPACE_DIR")` and falls back to `./workspace-<run_id>`
   relative to cwd. cwd inside the container was `/app`, so workspace landed on
   the ephemeral container disk and every artifact (`training.log` included)
   died with the pod. Fix: `secrets.env_for_run` now emits
   `WORKSPACE_DIR=/workspace`. **Lesson:** anything the runner reads from env
   has to be explicitly injected — relying on cwd for default-workspace
   resolution is brittle.

5. **RunPod auto-restarted exited containers, producing fast loops.** Once the
   pod's `autoresearch run` exited non-zero, RunPod's default container behavior
   restarted it every ~15 seconds. The runner loaded the same Run, re-entered
   the failed phase, wrote another identical error finding, exited, restarted.
   Burned $1.75 in 35 minutes invisibly. Fix: `cli.py` checks
   `run.status in (FAILED, COMPLETED)` at the top and exits 0, so RunPod
   doesn't see a crash to recover from. **Lesson:** clean exit on terminal
   state — non-zero exit is reserved for "something the supervisor should
   actually retry."

6. **Marker-gated pip install left fresh containers without deps.** The
   entrypoint cached "pip install ran on this volume" in
   `/workspace/.cache/requirements-installed.marker`. But the marker lives on
   the persistent volume while `/usr/local/lib/python3.11/site-packages` is
   ephemeral. New pod with a marker from a previous pod → skip install → no
   sae_lens → ModuleNotFoundError at module import. Fix: drop the marker,
   always pip install. The on-volume pip cache keeps re-installs to ~30-60s
   when wheels are cached. **Lesson:** install caches and dep markers must
   live on the *same* lifecycle layer. Pip cache on the volume is OK because
   pip looks for it; install marker on the volume is NOT OK because Python
   doesn't.

7. **sae-lens needs `zstandard` for `monology/pile-uncopyrighted`.** The
   pile-uncopyrighted shards are zstd-compressed JSONL; fsspec's compression
   detection needs the `zstandard` package installed or it crashes with
   `ValueError: Compression type zstd not supported`. Fix: add `zstandard` to
   requirements. **Lesson:** streaming HF datasets sometimes have implicit
   compression-codec deps that aren't transitively pulled by `datasets`.

8. **Qwen-32B + SAE Adam state exceeds 80GB.** On H100 80GB, training fits the
   model (~64GB bf16) and forward activations comfortably, then OOMs on the
   *first backward* — the SAE optimizer state (Adam moments for
   d_in=5120 × d_sae=102400 ≈ 1B params × 4 stats × 4 bytes = 16GB) plus
   gradients pushes total over 80GB. Fix: dispatch on B200 (180GB) at ~2× the
   hourly rate. **Lesson:** "model fits on H100 80GB" is necessary but not
   sufficient; downstream training state can double the working set.

9. **`n_checkpoints=10` × 12GB each filled the 200GB volume.** sae-lens dumps
   the SAE plus activation-buffer state on each checkpoint. For Qwen-32B
   activations, each dump was ~12GB. The default `n_checkpoints=10` produces
   120GB of intermediate state per run; two failed runs in a row exhausted the
   200GB volume and crashed canary 12 mid-final-save with `EDQUOT`. Fix:
   default `n_checkpoints=1` for canary-style smoke runs (only the final
   weights); also resized the volume to 500GB for headroom. **Lesson:**
   sae-lens's default checkpoint cadence is tuned for academic use where you
   want to resume; canary smoke tests should override it.

### Debugging affordances added in response

10. **Training-log snapshot on runner failure.** When `autoresearch run` exits
    non-zero, the entrypoint walks `/workspace` for `training.log` files
    modified in the last hour and uploads the tail (12KB cap) to R2 as an
    ERROR finding. This is what surfaced bug #6 above — without it, we needed
    a shell pod (and RunPod inventory) every time to read the actual error.
    `docker/entrypoint.sh`. **CRITICAL gotcha caught during deployment:** the
    `set -euo pipefail` at the top of the script killed the entrypoint *before*
    reaching the snapshot block when the runner exited non-zero. Fixed by
    wrapping the runner invocation in `set +e` / `set -e`. Always verify that
    your error-handling code path actually executes on the error path.

11. **`sshd` started in pod-mode entrypoint.** Our custom ENTRYPOINT replaced
    the base image's sshd-starting init, so `ssh -p <port> root@<ip>` was
    connection-refused on every canary even though RunPod exposes port 22 and
    injects `$PUBLIC_KEY`. Now the entrypoint runs `ssh-keygen -A` + drops
    `$PUBLIC_KEY` into authorized_keys + spawns sshd in the background before
    starting the runner. Cost: ~50ms at boot, no functional change for
    success path. **Lesson:** if you override a base image's ENTRYPOINT, you
    inherit responsibility for whatever the base entrypoint did.

### Operational notes from the loop

- **RunPod inventory in US-CA-2 is volatile.** H100 / H200 / A100-80GB all
  went dry at least once during this session. B200 was the only >80GB option
  for hours. Bound-to-DC network volumes magnify this — can't fall back to
  another DC.
- **Stuck pods block all dispatches.** When canary 10's runner exited 0
  cleanly (via the bug #5 fix), the pod stayed in RunPod's `RUNNING` state
  holding the volume attachment. Subsequent `start_transfer` calls returned
  500 from RunPod with no error context. Followup TODO: dispatcher should
  auto-terminate pods when their run reaches a terminal status.
- **`railway redeploy` reuses the cached source snapshot.** It doesn't always
  re-pull from main. `railway up` from local source is more reliable when
  fixing-then-deploying in a tight loop.
- **`runpod_default_image = :latest`** also surprised us. RunPod caches the
  `:latest` digest on the host between pod creations, so even after a fresh
  image push, new pods reuse the cached older digest. Fix: pin
  `runpod_default_image` to a SHA-tagged image (`sha-<short>`) via the
  Railway env override `AUTORESEARCH_RUNPOD_DEFAULT_IMAGE`. Forces RunPod to
  pull the new digest.
- **Always-install adds ~30-60s per pod boot** (with cache hits) compared to
  marker-gated. Worth the correctness gain.

### Cost notes — Qwen-32B SAE training, post-mortem

Wandb timestamps for canary 13 show pure training (post norm-scaling)
throughput is **~1,820 tokens/sec on B200** for d_sae=102400, k=64. The naive
`elapsed_seconds / training_tokens` rate (406 tok/s reported in the result
dict) mixes one-time setup with training; for short canaries it underestimates
steady-state by ~4.5×.

Real extrapolation for a 200M-token full Nura-budget run:
- Pure training: ~30.5 hours
- Setup overhead (model load + ActivationsStore init + norm scaling): ~16 min
- B200 @ $5.98/hr → **~$184 per hookpoint**
- Multi-layer sweep (e.g. 14 hookpoints) → ~$2.6k

The pipeline's `estimated_200M_token_run_cost_usd` field is misleading because
(a) it uses the H100 default rate, not the actual GPU, and (b) it
divides-elapsed-by-tokens. TODO: either pass actual `cost_per_hour` from
settings or compute throughput from the post-norm-scaling phase only.

## Notes from the fra_proj exploration

For the bridge skill design, useful concrete observations from
`/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj` (dmitry-em-repl
branch):

- Entry point is an argparse script (`phase1_fra_orchestrator.py`) — not a
  class. Bridge needs to wrap, not just import.
- Model name is hardcoded in two places (`phase1_fra_orchestrator.py:37-61`,
  `phase1_additive_orchestrator.py:54-78`). Bridge needs to find all sites,
  not just one.
- Dependency manifest is `uv.lock`, not `requirements.txt`. Bridge or pod
  entrypoint needs to handle both formats (or generate `requirements.txt`
  from `uv export`).
- `OPENAI_API_KEY_MATS` is a project-specific env var name. Bridge should
  detect non-standard names and either re-export or note the mapping in
  `autoresearch.toml`.
- Memory: 32B model + LoRA merge peaks ~130GB during load. Even on H100 80GB
  this OOMs without `device_map="auto"` or sequential CPU-merge. The bridge
  may need to recommend (or auto-apply) this fix in the wrapped pipeline.
- The Phase 1 result requires an SAE trained on the target model (`Nura-J/Qwen2.5-14B_SAE_ln1.normalised` is 14B-specific). For 32B, either a 32B
  SAE must exist or the experiment shape changes. This is a *research*
  prerequisite, not a code one — autoresearch can't synthesize an SAE.
