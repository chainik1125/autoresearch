# Ideas for automated research pipeline


## Current working projects

### Same pipeline, new model

**Deliverable**: A skill (?) or MCP that reliably has Claude replicate a tentative result across different settings

Paradigmatic Use case: We found that the FRA gets a very big effect in Qwen-14B. It sure would be nice to know if it replicates across all other 

Paradigmatic counter-example: I want to know if this effect generalizes across all layers. Can we make that measurement? 
Paradigmatic counter-example: Spinning up an agent to launch a command on the command line.

Does this need to exist?
Similar to many other automating proposals, it's worth asking whether a tool _needs_ to exist for this, or whether this is a symptom of an underlying process problem.




### Replication

**Deliverable** A skill (?) or MCP that reliably has Claude replicate an existing result

Use case: "I want to see if our techniques works on EM. Can you first replicate their finding, and then launch an agent to do X?"


### Night run/Take-over

Paradigmatic use case: Claude can you keep running this experiment overnight, fix any bugs and let me know how it goes?
Paradigmatic use case: I'm hopping out for lunch, can you takeover and keep going on your own?


Notes:
Lots of important things here: 
1. Hardware:
    - Can the Claude's decide how much to accelerate? I think yes, but you should set a default spend limit, and have an override that lets you increase it. Probably good to have a hard limit you can't exceed. This would be nice because then the Claude's could spin up, say 20x A40's and 
    - Storage is important. I think it would be good to have on persistent bucket where you can chuck everything into. Not sure if AWS or HF is the solution.
    - 
### Multi-Model Review

Not sure what the best way to do this, but it would be awesome if you had a skill that let's Claude call other models.

Paradigmatic use case: Here is my abstract. Can you spin up a debate protocol between three different models to refine and improve it?


### Dashboard (special case)

In the W2S automated researcher, the human-operator surface is a web dashboard (Flask server at `localhost:8000`): queue runs, watch a leaderboard, read a shared "findings forum" where parallel agents post intermediate results. We don't want this as the *default* interface — local Claude Code + MCP is — but we do want it as a **special-case opt-in**: a single `/dashboard` skill/module that stands up the analogue (web UI over the same controller state) for projects where a glanceable dashboard beats prompting. Treat it as one consumer of the controller's API, not the primary one.



## General abstraction idea

1. Input task description
    - "Pick up an idea off the idea shelf"
    - We will want some agent interaction. At this stage, however, you are just aiming for _conceptual clarity_ not yet questions of how much money, etc... the bean counting will come in stage 2.
    - Ah, this makes it a little unclear conceptually if the "Pre-register predictions" stage or, more generally, the Pre-theorizing stage should go here, in part 2, or have its own stage. Probably cleanest if it is its own stage

2. Pre-theorizing stage

    - Here you want to have as a minimum a pre-registered prediction of how the experiment will go, and what it will tell you.
    - This is also an opportune time to have MMR. In particular, the zeroth order workflow of making a call to GPT-Pro (including, if need be the workaround of having computer use sign into my ChatGPT and copy and paste the input, then copy and paste the response.)
    - Probably here, you want to check if there is a metric in the input task description, and if not each pre-registered prediction should at the least cache it out in a prediction for what this value will be.
    


3. Prepare experiment


4. Run experiments

    - Get hardware going
    - Optionally launch 'mechanic' agents that troubleshoot/babysit the RunPods.
    - 

5. Post-processing

    - Bring together all the results 


Note! You will want some meta-optimization agents going: I.e. you may want a pre-experiment layer which determines what the minimum 'time to insight' experiment is. Maybe this is its own stage, maybe a parrallel stage (figuring that out whilst you're running the core experiment), maybe a mid-point-check-in stage.


## Problems

1. De-risking runs: although technically not an issue for an experiment where you already have a promising result, still seems good practice. Shuld probably make this optional.


## v2 TODOs

### Auth / billing

- **Claude Code Max subscription for pod-side agents.** Today every
  off-laptop Claude call goes through the Anthropic API key (in Railway
  env), pay-per-token. The local Claude on the laptop runs under your
  Max subscription. If/when there's a clean way to authenticate the
  Claude Agent SDK against a Max sub instead of an API key, the prep /
  mechanic / postflight agents could run on the same flat-rate plan.
  Current Max auth is OAuth-style tied to interactive sessions; shelling
  out to `claude -p ...` headless on a pod technically works but is
  fragile and arguably outside intended use. Track Anthropic's roadmap
  here.

### List CPU offers dynamically (instead of hardcoding flavor list)

Today the dispatcher uses a hardcoded preference order `["cpu3c", "cpu3g",
"cpu5c", "cpu5g"]`. RunPod picks the first one with stock. Verified
2026-05-13 against US-CA-2: cpu3c + cpu3g had stock, cpu5c + cpu5g did
not. This will go stale as RunPod adds/removes flavors. v2: add
`compute.list_cpu_offers(data_center_id)` analogous to `list_gpu_offers`,
then thread it through `recommend_hardware` so the LLM advisor sees the
real available CPU catalog (currently the advisor only handles GPU).

### Pipeline-side footguns surfaced by /transfer attempts

Captured from the local Claude during a smoke test (2026-05-13):

- **`d_in` defaults to 5120 for sae_lens path** (`pipelines/sae_training.py:193`),
  which is the Qwen-14B/32B hidden size. Trying the same pipeline on
  Qwen-7B requires overriding to 3584 manually. Should auto-derive
  `d_in` from `target_model` via the model's HF config — `AutoConfig
  .from_pretrained(target_model).hidden_size` returns the right value.
- **`arditi` backend hardcodes `hook_resid_post`** (`pipelines/sae_training.py:265`).
  Arditi's underlying `run_from_config.py` is fully hookpoint-parameterizable
  — they trained at multiple layers and hookpoints in the paper. Our
  wrapper picks `f"blocks.{layer}.hook_resid_post"` purely for ergonomic
  reasons. Loosening it to accept any hook_name (defaulting to resid_post)
  is ~5 lines and gives Arditi-recipe SAEs at non-resid_post hookpoints.
- **Phase-0 of /transfer skill doesn't catch recipe mismatches.** When a
  user asks for "ln1 and resid_post SAEs at L15 in the Arditi recipe",
  the skill should notice that arditi backend only does resid_post and
  ask the user whether to (a) extend the wrapper or (b) accept that ln1
  will be a different recipe (sae_lens defaults). Today the skill
  silently dispatches both, producing non-comparable SAEs. Worth a
  Phase-0 intent-discovery rule.

### Prior-failures-as-context for the hardware advisor

The LLM advisor currently sees `offers`, `intent`, `pipeline_name`, and
`workflow` — but **not** which SKUs have failed before for this pipeline.
That context would unlock judgment calls like "A4000 OOM'd on the last
two preps with sae_lens deps; bump to A5000 even though it's $0.09/hr
more." Today the advisor can apply that reasoning only if the caller
hand-encodes it into `intent`.

The minimal wiring: in dispatcher's `_resolve_gpu`, query the last N
FAILED runs with the same `workflow + pipeline_name` and pass their
`pod_handle`-derived SKU + `last_error` snippet as a new `prior_failures`
field on the advisor prompt. Add a small index on R2 so the lookup is
O(1) — today it'd be a full prefix scan of `runs/`.

Why v2 not v1: the advisor handles intent strings well enough that
manually adding "don't pick A4000, it OOM'd last time" to the intent
covers it for now. Pull this forward when the same SKU bites us twice
in a row in the wild.

### CPU-only pods for agent workflows ✅ landed (2026-05-13)

Prep / mechanic / postflight now default to CPU-only RunPod pods
(~$0.05/hr). RunPod's REST `/v1/pods` accepts `computeType: "CPU"` +
`cpuFlavorIds`. The original assumption that "RunPod requires a GPU" was
wrong; documenting here so we don't lose track of why agent pods are
cheap. Caller can opt back into GPU with `compute_type="GPU"` for special
cases (e.g. a future prep variant that does import-smoke-tests on real
hardware).

RunPod supports CPU pods through a separate API path (`/v1/cpu-pods` or
similar gRPC). Wiring it in:

- `backends/compute/runpod.py`: add `create_cpu_session(spec)` that hits
  the CPU endpoint; switch which one based on `spec.gpu is None`.
- `SessionSpec.gpu: str | list[str] | None`. None ⇒ CPU pod.
- `mcp_surface.py`: pass `gpu=None` for prepare/mechanic/postflight; the
  `prefer="cheapest"` selector becomes a fallback for when caller passes
  required_vram_gb.

Tracked here because v1 took the pragmatic route (cheapest GPU, never
fall through to default_gpu) after a real incident: prep landed on H200
SXM ($5.49/hr) for a 30-second LLM call — ~5000x cost overrun, root cause
in dispatcher.py's fallback path. Fixed (2026-05-13) by adding
`fallback_to_default=False` for agent workflows. CPU pods are the next
mile of savings.

### Storage cost — RunPod network volume is the priciest piece

RunPod network volumes are ~$0.07/GB·month. We sized the catch-all
volume at 1TB (~$70/mo) on 2026-05-13 because debugging ENOSPC mid-run
was eating more time than the storage is worth. For v1 that's fine.

For v2, the genuinely cheaper architecture is **R2 (cold) + ephemeral
container disk (warm)**:

- Weights live in Cloudflare R2 (~$0.015/GB·mo, free egress).
- `entrypoint.sh` pre-stages the model for this Run from R2 to the pod's
  ephemeral container disk on boot.
- HF cache on the persistent volume becomes optional — only used if
  the user explicitly wants it for repeated runs of the same model.
- Trade-off: 5-10 min cold-boot per model vs. ~$50/mo storage savings
  at 1TB scale. Pays back fast if the iteration loop is "long-running
  pods, infrequent boots."

Honest comparison (researched 2026-05-13): cheap object-storage rates
(GCS ~$0.020/GB·mo, R2 ~$0.015/GB·mo) **do not** translate to cheap
mountable filesystems. PD-SSD on GCP is $0.17/GB·mo, Filestore
$0.20-0.30. The only way to beat RunPod here is the hybrid pattern
above — accept the cold-boot latency in exchange for object-storage
pricing. Not worth the architectural work for v1; revisit when (a) the
catch-all volume crosses 2TB or (b) we routinely cache 5+ huge models.

### Budget accounting

- **Hardware-advisor spend** isn't currently added to
  `Run.budget_spent_usd`. Tiny ($0.001-$0.01 per call) but should be
  tracked for honesty. `core.hardware.recommend()` returns the picks +
  rationale; needs to also surface the LLM cost so the dispatcher can
  call `budget.add_spend`.
- **Compute pod-hours not tracked in-flight.** Pipeline result
  extrapolates a cost from `cost_per_hour × elapsed`, but the
  in-flight pod-hour spend is never added to `budget_spent_usd`. So
  `check()` can't hard-stop a Run that's blowing budget on compute,
  only on LLM. Supervisor would need to poll RunPod billable seconds
  and call `add_spend` periodically.
- **Per-Run agent-tree budget rollup.** With `parent_run_id` landed,
  the natural "budget for this whole tree" is `sum(budget_spent_usd for
  r in subtree(root))`. Currently each Run has its own cap. A `--tree-
  budget` flag (or settings.default_tree_budget_usd) would express the
  user's actual intent: "spend at most $30 across everything spawned
  by this /transfer."

## Wishlist

1. 'Debug cycles' in the pre-launch phase. In general, it would be good to have a number of debug cycles (potentially with another model!) before you launch the experiment.

2. Data-tracking.

Would be really cool to track the data of how often different model choices did best. For example, for the debug cycles, you could imagine at the most basic level having a model injection to look over the code and give you a response (probably CC implements although that model could implement itself if I understand how that works - i.e. call Codex to do it vs. CC and see who does better).
