# How autoresearch works

A walkthrough of the v1 pipeline, from "you type `/transfer` in Claude Code" to
"findings show up in R2." This doc is the foundation for building cleaner
abstractions on top — once you understand which pieces own what, the right APIs
become obvious.

## Elevator pitch

You write a `Pipeline` (a Python class that does some measurement on a model).
autoresearch dispatches it onto a GPU pod, persists its findings somewhere
durable, and lets you check in from your laptop while it runs. Your laptop can
disconnect; the experiment keeps going.

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

End-to-end trace, by file and function:

```
[1] You, in Claude Code
    > /transfer fra_measurement Qwen/Qwen2.5-32B --budget 30

[2] Claude (the model in Claude Code)
    Picks the `start_transfer` MCP tool
      ↓ POST https://<controller>/mcp/  (JSON-RPC)

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
    • ensure /workspace/.huggingface
    • if PROJECT_REPO_URL set and pipelines/ not on volume → git clone
    • if pipelines/../requirements.txt → pip install (marker-gated, once per volume)
    • exec autoresearch run --run-id $RUN_ID --heartbeat

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
    autoresearch run returns; container exits; RunPod marks pod EXITED.
    The volume remains attached to nothing, ready for the next dispatch.

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

## What we'd do differently

- **Pipeline discovery is fragile.** Today the runner walks `pipeline_module_path`
  and looks for classes with a `name` attribute. A proper plugin registry
  (`entry_points` in pyproject) would be cleaner.
- **`autoresearch.toml` baked into the controller image is a project-local
  hack.** For a multi-project controller, env-var-only config is the right model.
- **No Postgres / SQLite.** Fine at v1 scale, but listing runs requires scanning
  R2 prefixes (one S3 list per call). A small SQLite-backed index would make
  `list_runs` instant.
- **Logging.** Each FSM transition writes a finding, but there's no central
  "log everything the pipeline printed" mechanism. Pipeline authors who want
  stdout captured have to write it themselves.
- **`Pipeline.run()` signature.** `params` is a free-form dict, which means
  every pipeline has to validate its own inputs. A Pydantic-typed `params`
  generic would catch mistakes earlier.

These are the targets for the next iteration once the FRA pipeline is real.

## What's next: real FRA on Qwen-32B

The smoke pipeline (`pipelines/fra_example.py`) sleeps half a second and
returns fake numbers. The real test is to replace it with the actual FRA
measurement code from your fra_proj.

Concretely, that means:
1. **Move** the FRA measurement into a `pipelines/fra_measurement.py` (in the
   fra_proj repo, not this one), conforming to the `Pipeline` protocol.
2. Set `project_repo_url` in `autoresearch.toml` to point at fra_proj instead
   of this repo.
3. If FRA needs special deps (e.g. specific transformers version), add a
   `requirements.txt` next to `pipelines/` so the entrypoint installs it once
   per volume.
4. `/transfer fra_measurement Qwen/Qwen2.5-32B --source Qwen/Qwen2.5-14B`
   from Claude Code.
5. First run will be slow — the pod has to download Qwen-32B (~60GB) from HF
   into the volume. Subsequent runs on the same volume are warm.

After that works once, the extension story is clear and we can talk about the
real abstractions to bake in (sweep over models, parallel agents, etc.).
