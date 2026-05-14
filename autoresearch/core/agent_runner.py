"""Agent runner — drive a real headless Claude Code loop on the pod.

This is the wrapper the prep / postflight workflows use to run a real
`claude-agent-sdk` query — i.e. a *headless Claude Code session* running on
the RunPod pod, with the user's project repo as cwd. The agent has the same
Read/Edit/Write/Bash/Glob/Grep tools you'd get from interactive Claude Code
on your laptop. Edits land in the cloned project on the volume, and git
sees them through `git diff` like any other change.

Why not just one `client.complete()`? Because the limitations show up
immediately for the prep case:
  - multi-file edits don't fit a single verbatim find/replace
  - file creation (adding a missing dep file) isn't expressible
  - import-smoke-tests (`python -c "import sae_lens"`) need Bash
  - the model needs to read several files before deciding what to fix

The SDK fixes all of those. The cost is ~$0.10-1 per agent run (vs
~$0.01-0.05 single-shot) and an async-to-sync bridge.

### Public API

  - `run_agent_with_tools(...)` — real CC. Use this for prep, postflight,
    or anything that needs filesystem tools.
  - `run_agent_single_shot(...)` — one `ModelClient.complete()` call. Use
    for mechanic or anything that's just "read findings + judge."

Both charge the Run's budget, emit an OBSERVATION finding with the agent's
final answer, and return an `AgentResult`.

### Auth

The SDK uses `ANTHROPIC_API_KEY` from env, which `secrets.env_for_run`
already injects into every pod. No new auth surface to wire.

### Permission mode

We use `bypassPermissions` because the pod is the sandbox — we trust the
agent to do whatever it needs in /workspace, and there's no interactive
human to grant per-tool-call permissions.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from autoresearch.backends.storage import StorageBackend
from autoresearch.core import budget, findings
from autoresearch.core.findings import FindingType
from autoresearch.core.run import Run

if TYPE_CHECKING:
    from autoresearch.backends.models.base import ModelClient


@dataclass
class AgentResult:
    text: str
    cost_usd: float
    num_turns: int = 0
    tool_calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Real CC loop via claude-agent-sdk
# ---------------------------------------------------------------------------


async def _query_with_tools(
    *,
    system: str,
    user: str,
    cwd: Path,
    allowed_tools: list[str],
    max_turns: int,
    max_budget_usd: float,
) -> tuple[str, float, int, list[tuple[str, dict[str, Any]]]]:
    # Import inside the function so the SDK isn't required for tests that
    # only exercise the single-shot path.
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ToolUseBlock,
        query,
    )

    options = ClaudeAgentOptions(
        system_prompt=system,
        cwd=str(cwd),
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        permission_mode="bypassPermissions",
    )

    final_text = ""
    cost_usd = 0.0
    num_turns = 0
    tool_calls: list[tuple[str, dict[str, Any]]] = []

    async for msg in query(prompt=user, options=options):
        if isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, TextBlock):
                    # Keep the most recent assistant text — the SDK delivers
                    # one final text turn after tool use is done.
                    final_text = block.text
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append((block.name, block.input))
        elif isinstance(msg, ResultMessage):
            cost_usd = msg.total_cost_usd or 0.0
            num_turns = msg.num_turns
            if msg.result:
                final_text = msg.result

    return final_text, cost_usd, num_turns, tool_calls


def run_agent_with_tools(
    *,
    run: Run,
    storage: StorageBackend,
    system: str,
    user: str,
    cwd: Path,
    allowed_tools: list[str],
    max_turns: int = 30,
    max_budget_usd: float = 10.0,
    label: str = "agent",
) -> AgentResult:
    """Run a headless CC loop and emit findings + charge the budget.

    `max_budget_usd` is a per-invocation hard cap enforced by the SDK
    itself — when this single agent invocation crosses the threshold it
    stops mid-loop. Defaults to $10/agent; workflows override if they need
    a different ceiling. This is the safety belt against a runaway agent;
    the Run-level `budget_cap_usd` is the outer envelope for the whole
    workflow (LLM + future compute).

    The agent's filesystem effects are NOT captured here — they're whatever
    landed in `cwd` (and visible via `git diff` if it's a repo). Callers
    that want to act on those effects (commit, push, etc.) inspect the
    working directory after this returns.
    """
    # Pre-flight: probe the `claude` CLI directly so failures from inside
    # the SDK's subprocess transport (which swallows stderr) are at least
    # surfaced before we delegate. Writes an OBSERVATION finding with the
    # captured stdout+stderr.
    _probe_claude_cli(run=run, storage=storage, label=label)

    final_text, cost_usd, num_turns, tool_calls = asyncio.run(
        _query_with_tools(
            system=system, user=user, cwd=cwd,
            allowed_tools=allowed_tools, max_turns=max_turns,
            max_budget_usd=max_budget_usd,
        )
    )
    if cost_usd > 0:
        budget.add_spend(storage, run, cost_usd)
    tool_summary = ", ".join(name for name, _ in tool_calls[:20]) or "no tool use"
    body = (
        f"=== {label} === ({num_turns} turns, {len(tool_calls)} tool calls, "
        f"${cost_usd:.4f})\n"
        f"tools used: {tool_summary}\n\n"
        f"{final_text}"
    )
    findings.append(storage, run, FindingType.OBSERVATION, body)
    return AgentResult(
        text=final_text, cost_usd=cost_usd,
        num_turns=num_turns, tool_calls=tool_calls,
    )


def _probe_claude_cli(*, run: Run, storage: StorageBackend, label: str) -> None:
    """Run `claude` CLI directly + capture stdout+stderr to a finding.

    claude-agent-sdk's subprocess transport swallows stderr on non-zero
    exit, leaving us with only "ProcessError: exit code 1". This probe
    runs the same binary directly with a minimal headless prompt and
    captures everything to an OBSERVATION finding, so when the SDK
    subsequently fails we have the actual error message to diagnose.

    Best-effort. Never raises. Adds ~1s to agent startup.
    """
    import os
    import shutil
    import subprocess

    parts = []

    parts.append("=== which claude ===")
    parts.append(shutil.which("claude") or "(claude not in PATH)")

    parts.append("\n=== claude --version ===")
    try:
        out = subprocess.run(
            ["claude", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        parts.append(f"exit={out.returncode}\nstdout: {out.stdout.strip()}\nstderr: {out.stderr.strip()}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"(--version failed: {e})")

    parts.append("\n=== LLM-related env keys (names only) ===")
    parts.append("\n".join(sorted(
        k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_"))
    )))

    parts.append("\n=== minimal headless probe: echo 'reply ok' | claude --print ===")
    try:
        out = subprocess.run(
            ["claude", "--print"],
            input="reply with the single word ok and nothing else",
            capture_output=True, text=True, timeout=30,
        )
        parts.append(f"exit={out.returncode}")
        parts.append(f"stdout (head):\n{out.stdout[:2000]}")
        parts.append(f"stderr (head):\n{out.stderr[:2000]}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"(probe failed: {type(e).__name__}: {e})")

    parts.append("\n=== claude --help (flag inventory) ===")
    try:
        out = subprocess.run(
            ["claude", "--help"], capture_output=True, text=True, timeout=10,
        )
        # Filter to flag names + check the ones the SDK uses
        relevant = [
            ln for ln in (out.stdout + out.stderr).splitlines()
            if any(f in ln for f in [
                "--output-format", "--verbose", "--system-prompt",
                "--allowedTools", "--max-turns", "--max-budget-usd",
                "--permission-mode", "--print",
            ])
        ]
        parts.append("\n".join(relevant) or "(no relevant flags in --help)")
    except Exception as e:  # noqa: BLE001
        parts.append(f"(--help failed: {e})")

    parts.append("\n=== SDK-style probe: full flag set used by claude_agent_sdk ===")
    try:
        out = subprocess.run(
            [
                "claude",
                "--output-format", "stream-json",
                "--verbose",
                "--system-prompt", "you are a test",
                "--allowedTools", "",
                "--max-turns", "1",
                "--max-budget-usd", "0.01",
                "--permission-mode", "bypassPermissions",
                "--print",
            ],
            input="say ok",
            capture_output=True, text=True, timeout=20,
        )
        parts.append(f"exit={out.returncode}")
        parts.append(f"stdout (head):\n{out.stdout[:2000]}")
        parts.append(f"stderr (head):\n{out.stderr[:2000]}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"(SDK-style probe failed: {type(e).__name__}: {e})")

    body = "[claude probe pre-" + label + "]\n```\n" + "\n".join(parts) + "\n```"
    try:
        findings.append(storage, run, FindingType.OBSERVATION, body)
    except Exception:  # noqa: BLE001
        pass  # probe is best-effort; never block the agent on it


# ---------------------------------------------------------------------------
# Single-shot LLM call (mechanic still uses this — no tools needed)
# ---------------------------------------------------------------------------


def run_agent_single_shot(
    *,
    run: Run,
    storage: StorageBackend,
    client: "ModelClient",
    system: str,
    user: str,
    max_tokens: int = 2000,
    label: str = "agent",
) -> AgentResult:
    """One `ModelClient.complete()` call; no tool use."""
    resp = client.complete(system=system, user=user, max_tokens=max_tokens)
    if resp.cost_usd > 0:
        budget.add_spend(storage, run, resp.cost_usd)
    findings.append(
        storage, run, FindingType.OBSERVATION,
        f"=== {label} ===\n{resp.text}",
    )
    return AgentResult(text=resp.text, cost_usd=resp.cost_usd)


# Backwards-compat alias for the original `run_agent` name used elsewhere.
# Resolves to single-shot since that was the v0 behavior.
run_agent = run_agent_single_shot
