# Mundane autoresearch

A walkthrough of the pipeline, from "you type `/transfer` in Claude Code"
to "the experiment summary shows up on a branch in your project's git." This
doc is the foundation for building cleaner abstractions on top — once you
understand which pieces own what, the right APIs become obvious.

## Elevator pitch

You hit a promising result on Qwen-14B. You want to know if it replicates
on Qwen-32B. Or on the Nanda EM setup. Or under 10 hyperparameter variants.

You type `/transfer` in Claude Code, describe the variation in plain
English ("do the same FRA measurement on Qwen-32B as we did on -14B"),
chat with Claude for 5 minutes about hardware + budget + any judgement
calls only you can make. Then you close your laptop and move on with
your day.

Server-side:

- A **prep agent** quietly fixes the obvious bits of your code that
  wouldn't survive an off-laptop dispatch — hardcoded `/root/` paths,
  mistuned dep pins, the things you'd otherwise notice mid-failure and
  have to come back to.
- Your **training runs** on the GPU you agreed on.
- A **postflight agent** writes the summary — spend, throughput, the
  result, sanity checks, links to wandb / HF — and commits it to your
  project repo as `autoresearch/runs/<run_id>/experiment_summary.md` on
  a `autoresearch/results-<run_id>` branch.

You see the markdown next time you `git pull` on your laptop, or open
the branch on GitHub from your phone. The headline is the spend total
and the outcome. If you want details, the body has them. The result
gets reported wherever it would have been reported — bookkeeping in
the project's results dir, or a figure in the paper if it pans out,
or a "huh, didn't replicate" note if it doesn't. If something looks
suspicious, you've still only spent the agreed budget — and the
findings + log snapshots + git diff of any prep edits are all there
to diagnose from.

The user-disengaged loop is the core promise: "I've got a promising
result; help me batch-test it across the variations I'd otherwise spend
a weekend running by hand."

## The four layers

```
┌───────────────────────────────────────────────────────────────────────────┐
│ LAYER 1 — Interface  (your laptop, ephemeral)                             │
│                                                                           │
│   Claude Code, with the autoresearch MCP server in ~/.claude.json         │
│   - /transfer skill prompts you, calls MCP tools                          │
│   - 12 tools: start_transfer, list_runs, tail_log, summarize_run,         │
│               list_findings, get/set_budget, takeover, release, …         │
└──────────────────────────────────┬────────────────────────────────────────┘
                                   │  MCP-over-HTTP (JSON-RPC)
                                   ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ LAYER 2 — Orchestration  (Railway, always-on)                             │
│                                                                           │
│   Controller process: FastAPI app, MODE=serve                             │
│   ┌──────────────────┬─────────────────────────────────────────────────┐  │
│   │ mcp_surface.py   │ 12 MCP tools; closures over storage/compute     │  │
│   │ dispatcher.py    │ build SessionSpec → compute.create_session      │  │
│   │ supervisor.py    │ async loop; if heartbeat stale → redispatch     │  │
│   │ healthz.py       │ /healthz + /version for Railway probes          │  │
│   │ server.py        │ FastAPI factory + lifespan; mounts MCP at /mcp/ │  │
│   └──────────────────┴─────────────────────────────────────────────────┘  │
│                                                                           │
│   Backends (selected by autoresearch.toml + env):                         │
│     storage  = S3-compatible (R2)                                         │
│     compute  = RunPod                                                     │
│     models   = Anthropic  (for the optional preflight/postflight checks)  │
└─────────┬──────────────────────────────┬────────────────────────┬─────────┘
          │ RunPod REST API              │ S3 API (boto3)         │ Anthropic
          ▼                              ▼                        ▼  API
┌───────────────────────┐  ┌─────────────────────────────┐
│ LAYER 3 — Compute     │  │ LAYER 4 — State (durable)   │
│  (RunPod GPU pod,     │  │  (Cloudflare R2)            │
│   ephemeral)          │  │                             │
│                       │  │  runs/<id>/                 │
│  /workspace  ←────────┼──┤    run.json                 │
│  (network volume,     │  │    checkpoint.json          │
│   durable across pods)│  │    heartbeat.json           │
│                       │  │    result.json              │
│  - HF model cache     │  │    findings/<key>.json   ⤴  │
│  - user pipelines     │  │    logs/<key>.txt        ⤴  │
│  - autoresearch       │  │                             │
│    package            │  │  ⤴ append-only, unique keys │
│                       │  └──────────────▲──────────────┘
│  Container entry:     │                 │
│  autoresearch run     │                 │ reads
│    --run-id $RUN_ID   │                 │ writes
│    --heartbeat   ─────┼─────────────────┘
└────────────┬──────────┘
             │ on cache miss
             ▼
┌─────────────────────────────────────┐
│ LAYER 4b — Inputs  (HuggingFace Hub)│
│   model weights, datasets           │
│   pulled on first use → cached on   │
│   the network volume forever        │
└─────────────────────────────────────┘
```

**Why four layers and not one?** Each layer has a different lifetime and
trust boundary:

- **Layer 1 (laptop)** is *ephemeral and human-driven*. You close your laptop,
  it goes away. It should never hold experiment state.
- **Layer 2 (controller)** is *persistent and machine-driven*. It survives your
  laptop closing. But it's cheap (no GPU) and should never run pipelines itself.
- **Layer 3 (pods)** is *ephemeral but heavy* (GPU). Pods get evicted, OOM-killed,
  preempted. State here is throwaway — except for what's on the persistent volume.
- **Layer 4 (storage)** is *durable*. Runs, findings, checkpoints. If a pod dies
  mid-pipeline, this is what makes recovery possible.

## Key abstractions

### `Pipeline` — what you write

A pipeline is the unit of measurement. You write one as a Python class
conforming to a small Protocol:

```python
# pipelines/fra_measurement.py (in YOUR project)

class FRAMeasurement:
    name = "fra_measurement"
    required_gpu = "H100 80GB"
    estimated_minutes = 90

    def run(self, *, params, workspace, storage):
        model_name = params["target_model"]           # e.g. "Qwen/Qwen2.5-32B"
        # ... your measurement code; uses HF cache via HF_HOME ...
        return {"fra_score": ..., "per_layer": [...], "metadata": {...}}
```

That's the whole contract. Three class attributes (name, required_gpu,
estimated_minutes) plus a `run()` that returns a JSON-serializable dict.

autoresearch's job is to:
- Run `run()` on a pod with `params["target_model"]` set to whatever target you
  asked for
- Persist the return value to storage as the run's "result"
- Optionally pre/post-validate with an LLM
- Survive the pod dying mid-`run()` if you've made `run()` resumable from its
  own on-disk state (more on this below)

See `pipelines/fra_example.py` for a fake pipeline that's used for smoke
testing — it sleeps briefly and returns deterministic numbers.

### `Run` — the central data object

Every dispatch creates a `Run`. It's a small Pydantic model that lives at
`runs/<id>/run.json` in storage:

```
Run {
  id:                 # 12-char hex
  workflow:           # "transfer" (for v1)
  pipeline_name:      # the user pipeline's `name`
  params:             # {"target_model": ..., "source_model": ...}
  status:             # queued | loading | running | validating | completed | failed | paused
  pod_handle:         # RunPod pod ID
  budget_cap_usd:     # spending limit (advisory)
  budget_spent_usd:   # accumulated spend
  created_at:
  last_heartbeat_at:
  last_error:         # populated on failure
}
```

The Run is the source of truth across pod incarnations. If pod A crashes
mid-pipeline and pod B picks up, pod B reads the same Run and continues. Code
in `autoresearch/core/run.py`.

### Pipeline runner FSM

The runner that drives `Pipeline.run()` is a finite state machine, ~130 lines
in `autoresearch/core/pipeline_runner.py`. Each transition writes a checkpoint
so resume-from-anywhere works:

```
                ┌────────┐
                │ queued │      (no checkpoint yet)
                └───┬────┘
       preflight    │  (optional LLM call: "is this model name sane?")
                    ▼
                ┌────────┐
                │loading │      ← checkpoint written
                └───┬────┘
                    │
                    ▼
                ┌────────┐
                │running │      ← checkpoint written; in_long_call=True
                └───┬────┘      (this is where pipeline.run() executes)
        success     │     failure
       ┌────────────┴─────────┐
       ▼                      ▼
  ┌─────────┐              (raise)
  │validating│             checkpoint stays at "running" — resume retries
  └────┬────┘              run.status = FAILED, last_error set
       │  postflight       (next call to run_pipeline picks up at running)
       ▼  (optional LLM)
  ┌─────────┐
  │completed│
  └─────────┘
```

**Key invariant**: on failure, the checkpoint is NOT overwritten with "failed."
It stays at whatever phase was in progress. This is what makes
resume-on-pod-death trivial — the next pod calls `run_pipeline` again and the
FSM re-enters the same phase. No special "retry" code path.

**Idempotency**: preflight and postflight LLM hooks may run multiple times if
pods die during them. Findings they write are duplicate-safe (append-only with
unique keys). The pipeline's `run()` may also be called multiple times — the
pipeline author is responsible for making it resumable (typically by writing
intermediate state to `workspace` and checking for it on entry).

### Heartbeat & supervisor — survive pod death

The runner spawns a thread (`autoresearch/core/heartbeat.py`) that writes
`runs/<id>/heartbeat.json` every 30 seconds:

```json
{"step": "running", "timestamp": "2026-05-12T19:09:34Z", "in_long_pipeline_call": true}
```

The controller's supervisor loop (every 30 seconds, async) reads heartbeats
for active runs. If a heartbeat is stale by more than:

- **5 minutes** when `in_long_pipeline_call=false`
- **2 hours** when `in_long_pipeline_call=true` (long measurements are normal,
  the runner flips this flag around `pipeline.run()`)

…then the supervisor terminates the old pod and dispatches a new one to the
**same network volume**. The new pod's runner reads the checkpoint from R2 and
resumes at the recorded step. No coordination required.

Code: `autoresearch/controller/supervisor.py`,
`autoresearch/controller/dispatcher.py:redispatch()`.

### Storage tiers — three places things live

| Tier | Where | What | Size | Cost |
|---|---|---|---|---|
| **1. Structured state** | R2 (Cloudflare) | runs, findings, checkpoints, heartbeats, logs, result dicts | KBs–MBs per run | Free up to 10GB |
| **2. Pod-local large blobs** | RunPod network volume (`/workspace`) | HF model cache, datasets, pipeline intermediate state | tens–hundreds of GB | ~$0.07/GB-mo |
| **3. Raw inputs** | HuggingFace Hub | model weights, datasets | as published | free |

**Convention for pipeline authors (important):**

- Use `storage.write(...)` (R2, tier 1) **only for small structured outputs
  you want to compare across runs.** Per-layer scores, summaries, comparison
  artifacts. Never multi-MB tensors — that path leads to surprise R2 bills.
- Use `workspace` (tier 2, the persistent volume) for **everything heavy**.
  HF caches happen automatically because `HF_HOME=/workspace/.huggingface` is
  set in the pod env. Your pipeline can also write intermediate artifacts here
  and check for them on resume.
- Use HF Hub (tier 3) for **raw inputs only**. `from_pretrained()` honors
  HF_HOME, so first miss downloads to your volume; subsequent runs are warm.

**Volume reuse is what makes restart cheap.** When the supervisor spawns a
replacement pod, it attaches the same volume id, so the HF cache (which may be
30GB+ for a model already used) is reused. Spawning a fresh-volume pod would
redownload everything.

## Lifecycle of one `/transfer`

End-to-end trace, by file and function. (Reflects the agent chain — see
"Agent chain" section above for the conceptual picture.)

```
[1] You, in Claude Code
    > /autoresearch:transfer  (full input — see skills/transfer/SKILL.md)

[2] Claude (the model in Claude Code)
    Phase 0: intent discovery
    Phase 1: recommend_hardware(...)             [MCP call]
    Phase 1.5: walk skills/transfer/PREP_CHECKLIST.md
    Phase 2: start_prepare(...)                  [MCP — returns prep.run_id]
    Phase 3: local file-existence checks (NO code grep)
    Phase 4: start_transfer(
               ...,
               params={..., auto_postflight=True, wait_for_prep_run_id=prep.run_id}
             )                                   [MCP]

    Tells you "dispatched; chain is server-side from here." You disengage.

[3] CONTROLLER (Railway, autoresearch/controller/server.py)
    mcp_surface.start_transfer(pipeline_name, target_model, budget_usd):
      ↓
    dispatcher.dispatch_new(workflow="transfer", pipeline_name, params, budget):
      • run = Run(...); run.save(storage)          # writes runs/<id>/run.json to R2
      • spec = _build_spec(run, settings)
          → env = secrets.env_for_run(run, settings)    # AWS_*, HF_HOME, RUN_ID, ...
          → SessionSpec(gpu, image, network_volume_id, env, container_registry_auth_id, ...)
      • compute.create_session(spec)              # backends/compute/runpod.py
          → POST https://rest.runpod.io/v1/pods
      • run.pod_handle = handle.id; run.status = QUEUED; run.save()
      ↓ return {run_id, status, pod_handle}

[4] RUNPOD
    Pulls ghcr.io/chainik1125/autoresearch:latest using the registered
    GIT_PAT registry credential. Attaches volume l7wyka84iy. Boots container
    with the env vars from step [3].

[5] CONTAINER ENTRYPOINT  (docker/entrypoint.sh, MODE=pod)
    • start sshd in background (debug-access affordance; uses $PUBLIC_KEY)
    • ensure /workspace/.huggingface
    • if PROJECT_REPO_URL set and pipelines/ not on volume → git clone
      (uses PROJECT_REPO_BRANCH if set; basic-auth via PROJECT_REPO_TOKEN
       for private repos, redacted from logs)
    • if pipelines/../requirements.txt → pip install ALWAYS (not marker-gated;
      the on-volume pip cache at /workspace/.cache/pip makes warm installs
      ~30-60s with cache hits)
    • run (NOT exec) autoresearch run --run-id $RUN_ID --heartbeat
    • on non-zero exit: walk /workspace for training.log files modified in
      the last hour, append the tail (12KB cap) to R2 as an ERROR finding
      (see "Debug-on-failure affordances" below)

[6] POD-SIDE RUNNER  (autoresearch/cli.py:cmd_run → workflows/transfer.py)
    transfer(run, pipeline, storage, workspace, heartbeat=True):
      heartbeat = HeartbeatWriter(storage, run); heartbeat.start()
      ctx = RunnerContext(storage, workspace, hooks=...)
      pipeline_runner.run_pipeline(run, pipeline, ctx):
        # FSM, see diagram above
        Phase 1 — preflight (optional LLM call)
        Phase 2 — pipeline.run()  ← YOUR CODE
        Phase 3 — postflight (optional LLM call)
        Phase 4 — done; return result
      heartbeat.stop()

[7] RUNNER WRITES TO R2 (incrementally)
    • findings/...  one per type=observation/result/error/etc.
    • result.json   the pipeline's return value
    • checkpoint.json   single-writer, atomic-overwrite, each FSM transition
    • heartbeat.json   single-writer, every 30s

[8] POD EXITS
    autoresearch run returns; container exits.
    *Supervisor (next tick within ≤30s) sees a terminal-status Run with a
    still-RUNNING pod and calls compute.terminate_session(). All workflows
    (prep, transfer, postflight) reap uniformly — see controller/supervisor.py.

[8b] AUTO-POSTFLIGHT (only for transfer Runs)
    Same supervisor tick that reaps the transfer pod ALSO checks:
      run.workflow == "transfer" AND run.status terminal
      AND params.auto_postflight AND not params.postflight_run_id
    → dispatcher.dispatch_new(workflow="postflight", parent_run_id=run.id, ...)
      writes the new pf.run_id into params.postflight_run_id (sentinel).
    Postflight pod boots, reads target Run via .autoresearch_postflight_context.md,
    headless Claude Code writes experiment_summary.md, runs `git checkout/add/
    commit/push` to autoresearch/results-<transfer.run_id>. Same supervisor
    cleans up the postflight pod when it's done.

[9] LAYER 1 INSPECTION (any time, doesn't require pod to be alive)
    Claude Code → MCP tools:
      list_runs()           → reads R2
      get_run(id)           → reads runs/<id>/run.json
      list_findings(id)     → reads runs/<id>/findings/*
      tail_log(id)          → reads runs/<id>/logs/*
      summarize_run(id)     → LLM call summarizing findings (costs tokens)
      takeover(id)          → returns SSH command to the pod (if alive)
      cancel(id)            → terminates pod, marks run failed
```

## Configuration surface

Three places config comes from. Precedence: **init args > env vars > toml > defaults.**

### `autoresearch.toml` (per-project)

The opinionated values. Lives at the project root. Committed to git.

```toml
controller_url = "https://controller-production-3e79.up.railway.app"

# Tier 1 — R2
storage = "s3"
storage_bucket = "fra-proj"
storage_endpoint_url = "https://<account-id>.r2.cloudflarestorage.com"
storage_region = "auto"

# Tier 2 — RunPod
compute = "runpod"
default_gpu = "H100 80GB"
runpod_data_center = "US-CA-2"
runpod_network_volume_id = "vol_..."
runpod_default_image = "ghcr.io/<you>/autoresearch:latest"
runpod_container_registry_auth_id = "cmp..."

# Tier 3 — HF — no toml entries; HF_HOME handled by pod env
project_repo_url = "https://github.com/<you>/your-project.git"
pipeline_module_path = "pipelines"

# Workflow defaults
default_budget_usd = 30
preflight = true
postflight = true
summarize_errors = true
validation_model = "claude-haiku-4-5-20251001"
```

### Env vars (secrets + per-environment overrides)

`AUTORESEARCH_` prefix; overrides toml. Secrets only here, never in toml.

```
AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY
AUTORESEARCH_RUNPOD_API_KEY
ANTHROPIC_API_KEY                  # only if validators are enabled
HF_TOKEN                           # only for gated models
```

The controller process on Railway and the pod process at runtime each
construct their own `Settings` from this layered config (see
`autoresearch/config.py`).

## What's NOT in v1 (and why)

The protocol surface is wider than v1's implementation — that's deliberate
forcing-function design. Each stub has an outline of what would go in it.

| Stub | File | Why deferred |
|---|---|---|
| Modal compute backend | `backends/compute/modal.py` | Ephemeral parallel agents not needed yet |
| OpenAI model client | `backends/models/openai.py` | Validators only use Anthropic in v1 |
| WandB tracker | `backends/tracking/wandb.py` | JSON-in-storage covers v1 |
| `replicate` workflow | `workflows/replicate.py` | Needs Claude Agent SDK loop (deferred risk) |
| `multi_model_review` | `workflows/multi_model_review.py` | Needs OpenAI/Gemini clients + Modal |
| `sweep` | `workflows/sweep.py` | Pressure-tests Pipeline protocol at N>1 |
| `night_run` | `workflows/night_run.py` | Probably a flag on transfer/sweep, not its own |

## Extension points (the right abstractions to build off)

If you're going to write the FRA pipeline next, these are the seams worth
understanding:

### 1. `Pipeline` protocol (`core/pipeline.py`)
The only thing user code touches. Three attributes + one method. Future
refactors of the protocol should preserve the "just a function over (params,
workspace, storage)" shape, not bolt on lifecycle hooks.

### 2. `WorkflowHooks` (`core/pipeline_runner.py`)
How the runner integrates with workflow-specific LLM calls + heartbeat. Adding
a new workflow type means: write a function that builds the right hooks +
calls `run_pipeline`. See `workflows/transfer.py` for the template — it's ~50
lines.

### 3. `ComputeBackend` (`backends/compute/base.py`)
Two-shape protocol. v1 uses `create_session` (long-lived pod) only. The
`run_ephemeral` half is reserved for when Modal lands. Adding a new backend
(Lambda Labs, Together, Anyscale) means implementing two methods.

### 4. `StorageBackend` (`backends/storage/base.py`)
Five-method protocol. Already has S3 + Local impls; protocol is well-tested
because both share parametrized tests.

### 5. `ModelClient` (`backends/models/base.py`)
Single-method protocol (`complete()`). v1 uses it only for the bounded
preflight/postflight/error LLM calls. Agent loops (v2 REPLICATE) will bypass
this and use the Claude Agent SDK directly.

### 6. MCP tools (`controller/mcp_surface.py`)
The interface Layer 1 sees. Adding a new tool means: write the closure in
`build_mcp()`, add a test in `tests/test_mcp_surface.py`. Tools are the
primary place to add operator features (e.g. `compare_runs`, `export_findings`).

## Agent chain (prep → transfer → postflight)

A `/transfer` dispatch is no longer a single pod. It's a three-stage pipeline
of pods, each running a different workflow. The user disengages after the
initial dispatch; the chain self-orchestrates server-side.

```
[Local Claude / laptop]
  /autoresearch:transfer  Phase 0:    intent discovery (conversational)
                          Phase 1:    recommend_hardware (MCP)
                          Phase 1.5:  PREP_CHECKLIST.md (deterministic, FAST)
                          Phase 2:    start_prepare → prep.run_id
                          Phase 3:    local readiness check (file-existence only)
                          Phase 4:    start_transfer(..., wait_for_prep_run_id=prep.run_id,
                                                      auto_postflight=True)
[user closes laptop]
[Server side]
  Prep pod      (small GPU; required_vram_gb=8)
    │  headless Claude Code via claude-agent-sdk
    │  cwd=/workspace/project; tools=Read,Edit,Write,Bash,Glob,Grep
    │  default: "no edits"; applies MECHANICAL fixes; halts on [USER-INPUT-NEEDED]
    │  if edits: commit + push to autoresearch/prepared-<prep.run_id>
    ▼  supervisor reaps the pod when this Run hits terminal status

  Transfer pod  (real GPU per pipeline.required_vram_gb)
    │  workflows/transfer.py:_wait_for_prep_gate(prep.run_id)
    │    - poll prep until terminal (15s interval, 20-min timeout)
    │    - if [USER-INPUT-NEEDED] in prep findings → ABORT, status FAILED
    │    - if prepared_branch in prep result → git fetch + checkout that branch
    │  Then the existing pipeline.run() chain (preflight, pipeline, postflight LLM hooks)
    │  Writes its own experiment_summary.md as a template fallback at terminal state
    ▼  supervisor reaps the pod when terminal

  [Supervisor tick]: sees transfer run is terminal + auto_postflight=true +
  no postflight_run_id yet → dispatches a postflight Run with parent_run_id
  set to the transfer's id. Sentinel `postflight_run_id` written into the
  transfer run's params prevents double-dispatch on the next tick.

  Postflight pod  (small GPU; required_vram_gb=8)
    │  headless Claude Code; cwd=/workspace/project
    │  tools=Read,Write,Bash
    │  reads target run's findings/result via injected
    │    .autoresearch_postflight_context.md
    │  Writes autoresearch/runs/<transfer.run_id>/experiment_summary.md
    │  with spend headline; runs git checkout/add/commit/push to
    │  autoresearch/results-<transfer.run_id>
    ▼  supervisor reaps the pod
```

### What ties it together

- **`Run.parent_run_id`** records the spawning Run (prep is parent of nothing;
  transfer's parent is None since it's user-initiated; postflight's parent is
  the transfer Run). Walk the tree by `[r for r in list_runs() if r.parent_run_id == X]`.
- **`Run.params`** carries the chain wiring: `wait_for_prep_run_id` gates
  transfer on prep; `auto_postflight` flips on the supervisor's chain step;
  `postflight_run_id` is the supervisor's sentinel against double-dispatch.
- **Pods reap themselves** via the supervisor's terminal-status cleanup pass
  (`controller/supervisor.py:tick`). The same mechanism that closes the
  "who kills the postflight pod" bootstrap question handles all three stages
  uniformly. No leaked compute on completed work.

### What each agent's tool surface is

| Agent | Workflow | Tools | Max turns | Per-agent budget cap |
|---|---|---|---|---|
| prep | `workflows/prepare.py` | Read, Edit, Write, Bash, Glob, Grep | 30 | $10 |
| transfer | `workflows/transfer.py` (deterministic FSM, not an agent) | n/a | n/a | covered by `Run.budget_cap_usd` |
| mechanic (opt-in) | `workflows/mechanic.py` (single-shot, no tools) | n/a | n/a | $10 envelope |
| postflight | `workflows/postflight.py` | Read, Write, Bash | 20 | $10 |

Per-agent caps are enforced by the SDK itself via `ClaudeAgentOptions.max_budget_usd`.
The run-level `Run.budget_cap_usd` (default $30) is the outer envelope for the
whole workflow, and worst-case LLM spend per `/transfer` is now 3×$10 = $30 —
fits inside the default.

### `core/agent_runner.py` — the SDK wrapper

Two entry points:

- `run_agent_with_tools(...)` — headless Claude Code via `claude-agent-sdk.query()`.
  Used by prep and postflight. Async-to-sync via `asyncio.run`. Captures the
  final assistant text, total cost from `ResultMessage`, and the list of
  tool calls. Charges budget; emits one OBSERVATION finding per agent run.
  Requires the `claude` CLI on PATH (the Dockerfile installs it via
  `npm install -g @anthropic-ai/claude-code`).
- `run_agent_single_shot(...)` — one `ModelClient.complete()` call. Used by
  mechanic (no tool surface needed; just judges from R2 data).

The SDK uses `ANTHROPIC_API_KEY` from the pod env, which `secrets.env_for_run`
already injects. No new auth wiring.

## Plugin distribution

The skills + MCP server config are now shipped as a Claude Code plugin. Users
install once:

```
/plugin marketplace add chainik1125/autoresearch
/plugin install autoresearch@autoresearch
```

When prompted, paste the controller URL. Plugin lays down `/autoresearch:transfer`
and `/autoresearch:make-compatible` + registers the `autoresearch` MCP server
pointing at the user's controller. Updates flow through `/plugin update` — no
per-project `autoresearch init` copies of the skills.

Layout (in this repo):

```
.claude-plugin/
  plugin.json        — manifest (name, version, mcpServers, userConfig)
  marketplace.json   — single-plugin marketplace stub so this repo IS the marketplace
mcp-config.json      — HTTP MCP wired to ${user_config.controller_url}
skills/transfer/SKILL.md            — the /transfer skill prompt
skills/transfer/PREP_CHECKLIST.md   — the local user-input-only checklist
skills/make-compatible/SKILL.md     — for adapting arbitrary projects
```

## Hardware selection module

`autoresearch/core/hardware.py` — picks which GPU type(s) to ask the compute
backend for on each dispatch. **This is one of the iteration-targets.** The
module is intentionally small and self-contained so heuristics, prompts, and
selection strategies can evolve without touching the dispatcher.

### Why a module at all

Early dispatches hardcoded a single GPU type ("H100 80GB"). When inventory
went dry in the volume's DC, the create-pod API 500'd and we had to manually
re-issue with a different GPU type. RunPod's `gpuTypeIds` accepts a list and
picks any available — letting us pass a preference-ordered fallback ladder.
But what goes on that ladder is non-trivial: it depends on the pipeline's
working-set memory, the volume's DC inventory, the user's intent (canary vs
production), and which model the pipeline is loading.

So we split the concern out:

```
core/hardware.py
├── GpuOffer                                    # data class: id, mem_gb, $, in_stock
├── select_gpu_offers(...)                      # pure-Python deterministic ranker
└── advise_gpu_offers(..., intent, client)      # LLM-advised wrapper, falls back
                                                #   to select_gpu_offers on failure
```

And one new method on the compute backend:

```
ComputeBackend.list_gpu_offers(data_center_id) -> list[GpuOffer]
```

### Selection flow

```
dispatcher.dispatch_new(..., gpu=None, required_vram_gb=N, intent="..."):
  1. If `gpu` is explicit (str or list)  → use as-is, bypass module.
  2. Else if `required_vram_gb` is set   → compute.list_gpu_offers()
                                          → hardware.recommend(...)
                                          → return rec.picks
  3. Else                                 → fall back to settings.default_gpu.
```

Step 2 is the path `/transfer` takes. The skill calls the
`recommend_hardware` MCP tool first to surface the choice + confidence to
the user, then calls `start_transfer` with either the auto-picks (high
confidence) or the user-confirmed override (needs_review).

### Three layers, three iteration targets

**Layer 1 — deterministic `select_gpu_offers`.** Pure function: filters and
sorts offers by a named heuristic. Heuristics:

- `"fastest_least_complicated"` (default): smallest VRAM bucket that has
  1.2× headroom over the floor; within bucket, newest-gen first (price as
  proxy for generation). Picks L40S for a 30GB job over a 180GB B200 you'd
  be overpaying for, but picks B200 over an H100 80GB that's borderline
  for Qwen-32B SAE training (no headroom).
- `"cheapest"`: lowest price-per-hour first. Used for budget-sensitive
  sweeps where wall-clock matters less than total cost.
- `"biggest_memory_first"`, `"fastest_first"`: niche; mostly future-facing
  knobs once we have observed tokens-per-sec data on real pipelines.

Used by the supervisor's restart path (no LLM in the loop) and as fallback
when the LLM advisor is unavailable. Unit-tested in
`tests/test_hardware.py`.

**Layer 2 — `recommend()` with deterministic confidence.** Top-level
function that wraps layer 1 and produces a `HardwareRecommendation`
dataclass: `{picks, confidence, rationale, alternatives}`.
Confidence rule (deterministic path):

- `"high"` — top pick has 1.2× VRAM headroom AND is < $3/hr AND there are
  ≥2 picks (RunPod has fallback inventory).
- `"needs_review"` — otherwise. Rationale explains why ("steep rate",
  "thin inventory", "VRAM at the floor — no headroom").

The /transfer skill consumes the confidence signal: `high` → auto-dispatch
and narrate; `needs_review` → ask the user via Phase 0 conversation with
the rationale as framing and `alternatives` as choices.

**Layer 3 — LLM-advised `recommend()`.** Same function, but when called
with `client` (a `ModelClient`) and `intent` (free-form user text), it
queries the LLM with `_ADVISOR_PROMPT`. The prompt asks the model to:

1. Apply the fastest-least-complicated default.
2. Reason about the operator's intent (e.g. "quick canary" → favor speed
   + battle-tested cards; "exploratory sweep, 10 hookpoints" → favor cost).
3. Output structured JSON with picks + its own confidence rating.

The model's picks are validated against the actual offers list before
returning (hallucinated GPU names are dropped). Any failure — invalid
JSON, no valid picks, network error — falls back silently to layer 2
(deterministic). Dispatches never block on advisor failure.

The prompt lives in `_ADVISOR_PROMPT` at module scope — single grep target
to iterate on.

### MCP surface

Two tools cover the hardware-selection UX:

- `recommend_hardware(required_vram_gb, intent?, pipeline_name?, ...)` —
  returns the recommendation dict so a skill can branch on confidence
  before dispatching. The /transfer skill calls this in Phase 0.
- `start_transfer(..., required_vram_gb?, intent?, gpu?)` — accepts both
  the auto-selection params and a manual override. Forward-compatible
  with the skill's recommend-then-dispatch flow.

### What to iterate on

The user has flagged this module for active development. Likely changes:

- **Better heuristics**: a `"best_perf_per_dollar"` once we have observed
  tokens/sec per (gpu, pipeline) stored somewhere — measure once, reuse.
- **Spot-pod fallthrough**: pass `interruptible=true` to the backend when
  the user signals tolerance for preemption.
- **Cross-DC search**: today we filter to the volume's DC. Pipelines that
  don't need volume reuse could open the search globally.
- **Prompt richness**: feed historical run data ("this pipeline averaged
  N hours on H100 last week") into the advisor for calibrated answers.
- **Streaming inventory probes**: today `list_gpu_offers` is one snapshot.
  An advisor that re-queries mid-decision could avoid picking something
  that goes dry in the next 30s.
- **A "VRAM headroom" knob**: today required_vram_gb is a hard floor; a
  preference for "1.5× headroom" would prevent edge OOMs without locking
  in a specific GPU.

The hardware module is *not* the place to put RunPod-specific quirks. Those
go in `backends/compute/runpod.py:list_gpu_offers`. The hardware module
just consumes the abstract `GpuOffer` stream.

## Debug-on-failure affordances

Round of canary debugging on Qwen-32B SAE training exposed that "the runner
exited 1; see /workspace/.../training.log" is useless if you can't read the
volume. Two affordances ship now (`docker/entrypoint.sh`):

1. **Training-log snapshot to R2.** After `autoresearch run` exits non-zero,
   walk `/workspace` for `training.log` files modified in the last hour;
   upload the tail (12KB cap) as an ERROR finding using the existing R2
   creds in the pod env. `list_findings(run_id)` then includes the actual
   pipeline subprocess traceback — no shell pod, no SSH. Gotcha: `set -euo
   pipefail` at the top of the script killed the entrypoint on non-zero
   exit before reaching the snapshot block; wrap the runner call in
   `set +e` / `set -e`.
2. **sshd in pod-mode.** Our custom ENTRYPOINT replaced the base
   image's sshd init. Now pod-mode runs `ssh-keygen -A`, writes
   `$PUBLIC_KEY` to `authorized_keys`, starts `/usr/sbin/sshd` in the
   background, then continues to the runner. ~50ms cost, makes every pod
   shell-accessible — useful for hangs (where the snapshot affordance
   can't help because the runner never returned).

## RunPod image-pin gotcha

`runpod_default_image = ".../autoresearch:latest"` is ergonomic but RunPod
caches the `:latest` digest on hosts between pod creations, so new pods can
reuse a stale digest after fresh image pushes. Pin to a SHA-tagged image via
the Railway env override `AUTORESEARCH_RUNPOD_DEFAULT_IMAGE=...:sha-<short>`
to force a fresh pull. Operational loop:
1. Push code to main → GHA builds `sha-<short>` + `:latest`
2. `railway variables --set AUTORESEARCH_RUNPOD_DEFAULT_IMAGE=...:sha-<short>`
3. Railway rebuilds (Dockerfile builder, ~5min) → new dispatches use new image

## Cost model

The pipeline result's `estimated_200M_token_run_cost_usd` mixes setup overhead
into the rate and uses an H100 default — both wrong if you have to run on
B200 (Qwen-32B's SAE training needs >80GB). Use wandb's `_runtime` between
first and last logged step for steady-state tok/s, then multiply by the
actual GPU's hourly rate.

Empirical (canary 13, Qwen-32B, layer 16, d_sae=102400, k=64, B200):
- Steady-state: 1,820 tok/s
- Setup overhead: ~16 min fixed
- Full 200M-token run: ~30.5h × $5.98/hr ≈ **$184/hookpoint**

## What we'd do differently

- **Pipeline discovery is fragile.** Walk + `name` attribute lookup; an
  entry-points plugin registry would be cleaner.
- **`autoresearch.toml` baked into the controller image** is a project-local
  hack. Env-var-only config is the right model for a multi-project controller.
- **No SQLite index.** `list_runs` scans R2 prefixes. Fine at v1 scale.
- **No central pod-stdout-to-R2 streamer.** The training-log snapshot covers
  the failure case but only for files named `training.log`.
- **`Pipeline.run()` `params` is untyped.** Pydantic-typed `params` generic
  would catch input errors earlier.
- **`cost_per_hour` should come from the dispatched GPU type**, not the
  hardcoded H100 default in the pipeline result.
- **Mechanic agent is opt-in and single-shot.** Real value would come from a
  Claude Agent SDK loop polling the live run; right now it's a one-shot
  judgement at a single moment in time.
- **Prep agent has no apply-confirm mode.** It either edits + pushes or
  halts on USER-INPUT-NEEDED. A "plan mode → show diff → user confirms →
  apply" flow would be safer for substantive refactors. Today's mitigation
  is plan-mode default + `git diff` on the pushed branch.
- **Pod lifecycle for prep + postflight is sub-optimal.** Both are small
  agents that don't need GPU; the dispatcher still picks a min-VRAM GPU pod
  (~$0.20/hr). A CPU-only backend (or Modal) would be more honest about the
  cost shape.

### Resolved since last revision

- ~~Pod lifecycle on terminal runs (stuck-RUNNING pods)~~ — supervisor's
  cleanup pass now reaps them on each tick.
- ~~Auto-select GPU from VRAM requirement~~ — `core/hardware.py:recommend`
  does this.
- ~~Plugin distribution~~ — autoresearch ships as a Claude Code plugin.

## Real-world example: Qwen-32B SAE training

First pipeline taken end-to-end was `sae_training` in `fra_proj`
(`autoresearch/transfer-qwen32b-sae` branch). It wraps an existing argparse
script (`fra/train_sae_at_hookpoint.py`, built on sae-lens 6.39) by
subprocess-spawning it with `--output-dir` pointing into `workspace/saes/…`.
The script writes its own training.log there; the pipeline returns a small
result dict with throughput, cost extrapolation, wandb URLs, and (when
enabled) HF Hub URL.

Wrap-don't-restructure was the right call: zero changes to the existing
CLI's argparse surface; all adapter logic lives in the user's repo's
`pipelines/sae_training.py`. The autoresearch image only needs to know how
to clone-and-pip-install.

The first canary surfaced nine independent bugs (pandas pin, torchvision
ABI, sae-lens kwarg rename, missing `WORKSPACE_DIR` injection, RunPod
restart loop, marker-skip-pip-install, missing zstandard, OOM on H100,
n_checkpoints disk blowup) before canary 13 completed cleanly. See
`notes/working_notes.md` → "Failure modes from the Qwen-32B SAE canary
loop" for the writeup of each.
