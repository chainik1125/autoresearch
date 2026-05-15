"""Agent runner — drive a tool-using LLM agent loop on the pod.

Two implementations sit behind a common `AgentRunner` protocol:

  - `ClaudeCodeRunner` (primary): wraps `claude-agent-sdk` → `claude` CLI.
    Real headless Claude Code with multi-turn + native tool use. Most
    mature path; designed for exactly our use case.

  - `CodexRunner` (fallback): hits OpenAI Chat Completions API directly
    via httpx, runs a tool-use loop locally. Implements the same
    Read/Edit/Write/Bash/Glob/Grep tool set the workflows use. No new
    Python dep (we already have httpx).

`run_agent_with_tools` tries the primary; on certain recoverable
failures (rate limit, API down, SDK init error) it falls back to the
next runner in the chain. Each runner attempt writes a finding so we
can see which one fired.

Adding a third runner (Gemini etc.) is straightforward: implement the
`AgentRunner` protocol, add an instance to `DEFAULT_RUNNER_CHAIN`.

### Public API (unchanged)

  - `run_agent_with_tools(...)` — tool-using agent (prep, postflight).
  - `run_agent_single_shot(...)` — one `ModelClient.complete()` call
    (mechanic). Single-provider; doesn't go through the chain.

### Auth

  - `ANTHROPIC_API_KEY` → ClaudeCodeRunner (via SDK / claude CLI).
  - `OPENAI_API_KEY` → CodexRunner (via direct httpx).
  Both are forwarded by `core/secrets.py:env_for_run` if the controller
  has them in env.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import httpx

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
    runner_name: str = ""   # which runner produced this result


class RunnerError(Exception):
    """Runner-side failure that's potentially recoverable.

    Caught by `run_agent_with_tools` to fall back to the next runner.
    Distinct from regular exceptions (TypeError, ImportError, etc.)
    which indicate programming bugs and should bubble up immediately.
    """


# ---------------------------------------------------------------------------
# AgentRunner protocol
# ---------------------------------------------------------------------------


class AgentRunner(Protocol):
    """Protocol for tool-using agent backends.

    Each implementation owns its own LLM provider auth, tool-execution
    loop, and cost accounting. `run_agent_with_tools` is the orchestrator
    that picks one and falls back to the next on RunnerError.
    """

    name: str  # e.g. "claude-code", "openai-codex"

    def run(
        self,
        *,
        run: Run,
        storage: StorageBackend,
        system: str,
        user: str,
        cwd: Path,
        allowed_tools: list[str],
        max_turns: int,
        max_budget_usd: float,
        label: str,
    ) -> AgentResult: ...


# ---------------------------------------------------------------------------
# ClaudeCodeRunner — primary path, wraps claude-agent-sdk
# ---------------------------------------------------------------------------


class ClaudeCodeRunner:
    name = "claude-code"

    def run(
        self, *, run, storage, system, user, cwd, allowed_tools,
        max_turns, max_budget_usd, label,
    ) -> AgentResult:
        # Pre-flight probe — captures stderr from the claude CLI in case
        # the SDK's subprocess transport swallows it.
        _probe_claude_cli(run=run, storage=storage, label=label)

        try:
            final_text, cost_usd, num_turns, tool_calls = asyncio.run(
                _claude_query_with_tools(
                    system=system, user=user, cwd=cwd,
                    allowed_tools=allowed_tools, max_turns=max_turns,
                    max_budget_usd=max_budget_usd,
                    run=run, storage=storage, label=label,
                )
            )
        except ImportError as e:
            raise RunnerError(f"claude-agent-sdk not installed: {e}") from e
        except Exception as e:  # noqa: BLE001
            # The SDK raises ProcessError for CLI failures; httpx exceptions
            # for transport issues; CLINotFoundError when claude isn't in
            # PATH. All of these are recoverable — fall back to the next
            # runner.
            err_name = type(e).__name__
            if err_name in (
                "ProcessError", "CLINotFoundError", "CLIConnectionError",
                "ConnectError", "ReadTimeout", "ConnectTimeout",
            ) or "rate_limit" in str(e).lower() or "overloaded" in str(e).lower():
                raise RunnerError(f"claude-agent-sdk {err_name}: {str(e)[:200]}") from e
            raise

        return _finalize_agent_result(
            run=run, storage=storage, label=label,
            text=final_text, cost_usd=cost_usd,
            num_turns=num_turns, tool_calls=tool_calls,
            runner_name=self.name,
        )


async def _claude_query_with_tools(
    *,
    system: str, user: str, cwd: Path,
    allowed_tools: list[str], max_turns: int, max_budget_usd: float,
    run: Run | None = None, storage: StorageBackend | None = None,
    label: str = "agent",
) -> tuple[str, float, int, list[tuple[str, dict[str, Any]]]]:
    """Drive the claude-agent-sdk query loop, streaming turn findings."""
    from claude_agent_sdk import (
        AssistantMessage, ClaudeAgentOptions, ResultMessage,
        TextBlock, ToolUseBlock, query,
    )

    options = ClaudeAgentOptions(
        system_prompt=system,
        cwd=str(cwd),
        allowed_tools=allowed_tools,
        max_turns=max_turns,
        max_budget_usd=max_budget_usd,
        # bypassPermissions is refused by claude CLI when running as root
        # (which the pod is). acceptEdits is the closest viable alternative
        # for headless mode. Switch to bypassPermissions if/when we run
        # claude as a non-root user.
        permission_mode="acceptEdits",
    )

    final_text = ""
    cost_usd = 0.0
    num_turns = 0
    tool_calls: list[tuple[str, dict[str, Any]]] = []
    turn_idx = 0

    def _stream(body: str) -> None:
        if run is None or storage is None:
            return
        try:
            findings.append(storage, run, FindingType.OBSERVATION, body)
        except Exception:  # noqa: BLE001
            pass

    _stream(f"[{label}] claude-code: sdk loop started")

    async for msg in query(prompt=user, options=options):
        if isinstance(msg, AssistantMessage):
            turn_idx += 1
            turn_text_parts: list[str] = []
            turn_tool_parts: list[str] = []
            for block in msg.content:
                if isinstance(block, TextBlock):
                    final_text = block.text
                    turn_text_parts.append(block.text[:1500])
                elif isinstance(block, ToolUseBlock):
                    tool_calls.append((block.name, block.input))
                    input_repr = str(block.input)[:300]
                    turn_tool_parts.append(f"{block.name}({input_repr})")
            lines = [f"[{label}] claude-code: turn {turn_idx}"]
            if turn_text_parts:
                lines.append("text: " + " | ".join(turn_text_parts)[:2000])
            if turn_tool_parts:
                lines.append("tools: " + "; ".join(turn_tool_parts)[:2000])
            _stream("\n".join(lines))
        elif isinstance(msg, ResultMessage):
            cost_usd = msg.total_cost_usd or 0.0
            num_turns = msg.num_turns
            if msg.result:
                final_text = msg.result
            _stream(
                f"[{label}] claude-code: result {num_turns} turns, "
                f"${cost_usd:.4f}, stop_reason={getattr(msg, 'stop_reason', '?')}"
            )

    _stream(f"[{label}] claude-code: loop exited ({turn_idx} assistant turns)")
    return final_text, cost_usd, num_turns, tool_calls


def _probe_claude_cli(*, run: Run, storage: StorageBackend, label: str) -> None:
    """Capture claude CLI state to a finding so we can diagnose SDK failures
    where stderr is swallowed by the subprocess transport. Best-effort.
    """
    parts: list[str] = []
    parts.append("=== which claude ===")
    parts.append(shutil.which("claude") or "(claude not in PATH)")
    parts.append("\n=== claude --version ===")
    try:
        out = subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10,
        )
        parts.append(f"exit={out.returncode}\nstdout: {out.stdout.strip()}\nstderr: {out.stderr.strip()}")
    except Exception as e:  # noqa: BLE001
        parts.append(f"(--version failed: {e})")
    parts.append("\n=== LLM-related env keys (names only) ===")
    parts.append("\n".join(sorted(
        k for k in os.environ if k.startswith(("ANTHROPIC_", "CLAUDE_", "OPENAI_"))
    )))
    body = f"[{label}] claude probe\n```\n" + "\n".join(parts) + "\n```"
    try:
        findings.append(storage, run, FindingType.OBSERVATION, body)
    except Exception:  # noqa: BLE001
        pass


# ---------------------------------------------------------------------------
# CodexRunner — fallback path, OpenAI Chat Completions API + local tool loop
# ---------------------------------------------------------------------------


# Default model for the Codex fallback. OpenAI's coding-tuned model.
# The "gpt-5-codex" name is the public alias as of 2026-05.
_CODEX_MODEL = os.environ.get("AUTORESEARCH_CODEX_MODEL", "gpt-5-codex")

# Rough USD per million tokens for cost tracking. Approximate; refine when
# OpenAI publishes more precise rates.
_CODEX_INPUT_PRICE_PER_1M = float(os.environ.get("AUTORESEARCH_CODEX_INPUT_PRICE", "1.25"))
_CODEX_OUTPUT_PRICE_PER_1M = float(os.environ.get("AUTORESEARCH_CODEX_OUTPUT_PRICE", "10.0"))


class CodexRunner:
    """OpenAI-backed fallback. Implements a tool-use loop against the Chat
    Completions API using only httpx (no openai-python dep).

    Limitations vs ClaudeCodeRunner:
      - Tool list is hardcoded to Read/Edit/Write/Bash/Glob/Grep. Adding new
        tools requires updating `_codex_tools_schema()` here.
      - No advanced features (parallel tool calls, partial messages).
      - Tool execution runs in this Python process, not in a sandbox.
        Fine on the pod (the pod IS the sandbox); risky if you ever run
        this runner on the controller.
    """

    name = "openai-codex"

    def run(
        self, *, run, storage, system, user, cwd, allowed_tools,
        max_turns, max_budget_usd, label,
    ) -> AgentResult:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RunnerError("OPENAI_API_KEY not in env; cannot use Codex fallback")

        def _stream(body: str) -> None:
            try:
                findings.append(storage, run, FindingType.OBSERVATION, body)
            except Exception:  # noqa: BLE001
                pass

        _stream(f"[{label}] openai-codex: starting (model={_CODEX_MODEL})")

        tools_schema = _codex_tools_schema(allowed_tools)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        cost_usd = 0.0
        tool_calls: list[tuple[str, dict[str, Any]]] = []
        final_text = ""
        turn_idx = 0

        client = httpx.Client(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            timeout=120.0,
        )
        try:
            while turn_idx < max_turns:
                turn_idx += 1
                try:
                    resp = client.post("/chat/completions", json={
                        "model": _CODEX_MODEL,
                        "messages": messages,
                        "tools": tools_schema,
                        "tool_choice": "auto",
                    })
                except httpx.RequestError as e:
                    raise RunnerError(f"openai request error: {e}") from e

                if resp.status_code in (429, 500, 502, 503):
                    raise RunnerError(f"openai {resp.status_code}: {resp.text[:200]}")
                if resp.status_code >= 400:
                    # 4xx other than rate-limit = real error (bad request,
                    # auth, etc.) — don't fall back further.
                    raise RuntimeError(f"openai {resp.status_code}: {resp.text[:500]}")

                data = resp.json()
                usage = data.get("usage", {})
                turn_cost = (
                    usage.get("prompt_tokens", 0) * _CODEX_INPUT_PRICE_PER_1M / 1e6
                    + usage.get("completion_tokens", 0) * _CODEX_OUTPUT_PRICE_PER_1M / 1e6
                )
                cost_usd += turn_cost
                if cost_usd > max_budget_usd:
                    _stream(f"[{label}] openai-codex: budget cap hit at ${cost_usd:.4f}")
                    break

                choice = data["choices"][0]
                msg = choice["message"]
                content = msg.get("content") or ""
                tcalls = msg.get("tool_calls") or []

                turn_summary = [f"[{label}] openai-codex: turn {turn_idx}"]
                if content:
                    final_text = content
                    turn_summary.append("text: " + content[:1500])

                if not tcalls:
                    # Conversation done — model answered without tools.
                    turn_summary.append(f"(stop, cost so far ${cost_usd:.4f})")
                    _stream("\n".join(turn_summary))
                    break

                # Append the assistant message with tool_calls so the next
                # round can correlate tool_call_id → tool result.
                messages.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": tcalls,
                })

                tool_call_summaries = []
                for tc in tcalls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "?")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except json.JSONDecodeError:
                        args = {"_raw": fn.get("arguments", "")}
                    tool_calls.append((name, args))
                    tool_call_summaries.append(f"{name}({str(args)[:300]})")
                    result = _execute_codex_tool(name, args, cwd=cwd)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": result[:8000],  # cap tool output going back to model
                    })

                turn_summary.append("tools: " + "; ".join(tool_call_summaries)[:2000])
                _stream("\n".join(turn_summary))
        finally:
            client.close()

        _stream(
            f"[{label}] openai-codex: loop exited ({turn_idx} turns, "
            f"${cost_usd:.4f})"
        )

        return _finalize_agent_result(
            run=run, storage=storage, label=label,
            text=final_text, cost_usd=cost_usd,
            num_turns=turn_idx, tool_calls=tool_calls,
            runner_name=self.name,
        )


def _codex_tools_schema(allowed_tools: list[str]) -> list[dict[str, Any]]:
    """OpenAI Chat Completions API expects tools as JSON-Schema function
    definitions. Map the workflow's allow-list to the schema."""
    all_tools = {
        "Read": {
            "name": "Read",
            "description": "Read a file from disk. Returns the file contents as text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path to the file."},
                },
                "required": ["file_path"],
            },
        },
        "Write": {
            "name": "Write",
            "description": "Write text content to a file. Overwrites if exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
            },
        },
        "Edit": {
            "name": "Edit",
            "description": "Edit a file by replacing one substring with another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "description": "Replace all occurrences. Default false."},
                },
                "required": ["file_path", "old_string", "new_string"],
            },
        },
        "Bash": {
            "name": "Bash",
            "description": "Run a shell command and return stdout+stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string", "description": "What this command does (for logs)."},
                },
                "required": ["command"],
            },
        },
        "Glob": {
            "name": "Glob",
            "description": "List files matching a glob pattern relative to cwd.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"],
            },
        },
        "Grep": {
            "name": "Grep",
            "description": "Search file contents using a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory or file to search (defaults to cwd)."},
                },
                "required": ["pattern"],
            },
        },
    }
    return [
        {"type": "function", "function": all_tools[t]}
        for t in allowed_tools if t in all_tools
    ]


def _execute_codex_tool(name: str, args: dict[str, Any], *, cwd: Path) -> str:
    """Execute one tool locally and return a string result suitable for the
    model. Best-effort; returns the error message on failure (not raise)
    so the model can decide what to do next."""
    try:
        if name == "Read":
            return Path(args["file_path"]).read_text()
        if name == "Write":
            p = Path(args["file_path"])
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {len(args['content'])} chars to {p}"
        if name == "Edit":
            p = Path(args["file_path"])
            body = p.read_text()
            old, new = args["old_string"], args["new_string"]
            if args.get("replace_all"):
                body2 = body.replace(old, new)
                count = body.count(old)
            else:
                if body.count(old) != 1:
                    return f"ERROR: old_string occurs {body.count(old)} times (need exactly 1, or set replace_all=true)"
                body2 = body.replace(old, new, 1)
                count = 1
            p.write_text(body2)
            return f"edited {p}: replaced {count} occurrence(s)"
        if name == "Bash":
            proc = subprocess.run(
                args["command"], shell=True, capture_output=True, text=True,
                cwd=str(cwd), timeout=120,
            )
            return f"exit={proc.returncode}\nstdout:\n{proc.stdout[:4000]}\nstderr:\n{proc.stderr[:2000]}"
        if name == "Glob":
            matches = sorted(str(p) for p in cwd.glob(args["pattern"]))
            return "\n".join(matches[:200]) or "(no matches)"
        if name == "Grep":
            path = args.get("path") or str(cwd)
            proc = subprocess.run(
                ["grep", "-rn", args["pattern"], path],
                capture_output=True, text=True, timeout=30,
            )
            return proc.stdout[:8000] or "(no matches)"
        return f"ERROR: unknown tool {name}"
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {type(e).__name__}: {str(e)[:500]}"


# ---------------------------------------------------------------------------
# Default runner chain + public orchestrator
# ---------------------------------------------------------------------------


def default_runner_chain() -> list[AgentRunner]:
    """Build the default chain. Codex is omitted when OPENAI_API_KEY isn't
    set — no point trying a fallback we can't authenticate."""
    chain: list[AgentRunner] = [ClaudeCodeRunner()]
    if os.environ.get("OPENAI_API_KEY"):
        chain.append(CodexRunner())
    return chain


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
    runners: list[AgentRunner] | None = None,
) -> AgentResult:
    """Run a tool-using agent, trying each runner in order until one
    succeeds. Default chain: [ClaudeCodeRunner, CodexRunner (if key set)].

    On `RunnerError` (rate limit, API down, transport failure) the next
    runner gets a shot. Programming bugs (TypeError, ImportError of our
    own code, etc.) bubble up immediately — those aren't recoverable by
    swapping providers.
    """
    chain = runners or default_runner_chain()
    last_exc: Exception | None = None
    for runner in chain:
        findings.append(
            storage, run, FindingType.OBSERVATION,
            f"[{label}] trying runner={runner.name}",
        )
        try:
            return runner.run(
                run=run, storage=storage,
                system=system, user=user, cwd=cwd,
                allowed_tools=allowed_tools,
                max_turns=max_turns, max_budget_usd=max_budget_usd,
                label=label,
            )
        except RunnerError as e:
            findings.append(
                storage, run, FindingType.OBSERVATION,
                f"[{label}] runner={runner.name} raised RunnerError: {e}; "
                f"trying next in chain",
            )
            last_exc = e
            continue
    raise RuntimeError(
        f"all {len(chain)} runners failed for {label}; last error: {last_exc}"
    ) from last_exc


def _finalize_agent_result(
    *,
    run: Run, storage: StorageBackend, label: str,
    text: str, cost_usd: float, num_turns: int,
    tool_calls: list[tuple[str, dict[str, Any]]],
    runner_name: str,
) -> AgentResult:
    """Common post-run bookkeeping: charge budget + write summary finding."""
    if cost_usd > 0:
        budget.add_spend(storage, run, cost_usd)
    tool_summary = ", ".join(name for name, _ in tool_calls[:20]) or "no tool use"
    body = (
        f"=== {label} ({runner_name}) === ({num_turns} turns, "
        f"{len(tool_calls)} tool calls, ${cost_usd:.4f})\n"
        f"tools used: {tool_summary}\n\n"
        f"{text}"
    )
    findings.append(storage, run, FindingType.OBSERVATION, body)
    return AgentResult(
        text=text, cost_usd=cost_usd,
        num_turns=num_turns, tool_calls=tool_calls,
        runner_name=runner_name,
    )


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
    """One `ModelClient.complete()` call; no tool use. Single-provider —
    doesn't go through the runner chain because mechanic's use case is
    well-served by a one-shot LLM read."""
    resp = client.complete(system=system, user=user, max_tokens=max_tokens)
    if resp.cost_usd > 0:
        budget.add_spend(storage, run, resp.cost_usd)
    findings.append(
        storage, run, FindingType.OBSERVATION,
        f"=== {label} ===\n{resp.text}",
    )
    return AgentResult(text=resp.text, cost_usd=resp.cost_usd)


# Backwards-compat alias for the original `run_agent` name used elsewhere.
run_agent = run_agent_single_shot
