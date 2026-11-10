"""Claude Code as an LLM provider: run the agents on your own Claude subscription.

No API key. Each completion is one call to the `claude` command you already have
installed and signed in to, so the usage is counted against your Claude plan the
same way as a chat you typed yourself.

Why shell out and not call the API with your login token: Anthropic's terms allow a
Claude plan to be used through the unmodified Claude Code app, and do not allow other
software to collect or reuse the plan's OAuth token. This module never reads, stores
or forwards any credential. It only starts the `claude` binary and reads its output.

The call is deliberately stripped down (`--safe-mode`, no tools, no session saved):
- `--safe-mode` keeps the login and model choice but skips CLAUDE.md files, plugins,
  hooks and MCP servers. Without it a one-word reply costs ~18k input tokens on a
  typical setup (measured); with it, ~1k.
- `--bare` would be even smaller, but it ignores the subscription login, so it is
  never used here.
- `ANTHROPIC_API_KEY` & co. are removed from the child's environment. Claude Code
  prefers an API key over the subscription login when one is set, and this provider
  exists to avoid billing the API.

`python -m hedge_fund.llm.claude_code` runs a one-call self-test.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from hedge_fund.llm.client import LLMCallError, TokenListener

PROVIDER = "Claude Code"
MODEL_PREFIX = "claude-code/"

# Variables that make Claude Code bill an API account (or another cloud) instead of
# using the subscription login.
_BILLING_ENV = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_BASE_URL",
    "CLAUDE_CODE_USE_BEDROCK",
    "CLAUDE_CODE_USE_VERTEX",
    "CLAUDE_CODE_USE_FOUNDRY",
)

_FALLBACK_SYSTEM = "You are a helpful assistant."


def _max_parallel() -> int:
    """How many `claude` processes may run at once. Agents run in parallel; a
    plan's rate limit is shared by all of them, so keep the fan-out modest."""
    try:
        return max(1, int(os.environ.get("HEDGE_FUND_CLAUDE_CONCURRENCY", "3")))
    except ValueError:
        return 3


_SLOTS = threading.BoundedSemaphore(_max_parallel())


def find_claude_binary() -> str:
    """Path to the `claude` executable, or ValueError saying how to fix it."""
    override = os.environ.get("CLAUDE_CODE_BIN")
    if override:
        if Path(override).is_file():
            return override
        raise ValueError(f"CLAUDE_CODE_BIN={override!r} is not a file.")
    found = shutil.which("claude")
    if found:
        return found
    raise ValueError("The `claude` command was not found. Install Claude Code, run `claude` once to sign in with your Claude account, or set CLAUDE_CODE_BIN to its full path.")


def cli_model(model_id: str) -> str:
    """The id `claude --model` expects: the registry id without our routing prefix."""
    return model_id[len(MODEL_PREFIX):] if model_id.startswith(MODEL_PREFIX) else model_id


def _child_env() -> dict[str, str]:
    return {key: value for key, value in os.environ.items() if key not in _BILLING_ENV}


def _workdir() -> str:
    """A neutral, empty folder to run in, so nothing from the caller's project is
    picked up as context."""
    path = Path(tempfile.gettempdir()) / "hedge-fund-claude-code"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


class ClaudeCodeLLM:
    """The LLMClient protocol, served by the local `claude` command.

    With an `on_token` listener the call streams: the listener sees text as it
    arrives, and complete() still returns the whole response (like ChatLLM).
    """

    def __init__(self, model: str, timeout: float = 300.0, on_token: TokenListener = None, binary: str | None = None) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        self.model = model
        self._timeout = timeout
        self._on_token = on_token
        self._binary = binary or find_claude_binary()

    # -- LLMClient ----------------------------------------------------------

    def complete(self, system: str, user: str) -> str:
        prompt_file = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8")
        try:
            with prompt_file:
                prompt_file.write(system if system.strip() else _FALLBACK_SYSTEM)
            command = self._command(prompt_file.name, stream=self._on_token is not None)
            with _SLOTS:
                if self._on_token is None:
                    return self._run(command, user)
                return self._stream(command, user)
        finally:
            try:
                os.unlink(prompt_file.name)
            except OSError:
                pass

    # -- internals ----------------------------------------------------------

    def _command(self, system_file: str, stream: bool) -> list[str]:
        command = [
            self._binary,
            "-p",
            "--safe-mode",
            "--model",
            cli_model(self.model),
            "--system-prompt-file",
            system_file,
            "--tools",
            "",
            "--no-session-persistence",
            "--disable-slash-commands",
        ]
        if stream:
            command += ["--output-format", "stream-json", "--verbose", "--include-partial-messages"]
        else:
            command += ["--output-format", "json"]
        return command

    def _run(self, command: list[str], user: str) -> str:
        try:
            done = subprocess.run(
                command,
                input=user,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=_workdir(),
                env=_child_env(),
                timeout=self._timeout,
            )
        except subprocess.TimeoutExpired:
            raise LLMCallError(f"claude timed out after {self._timeout:g}s") from None
        except OSError as exc:
            raise LLMCallError(f"could not start claude: {exc}") from None
        return _result_text(done.stdout, done.returncode, done.stderr)

    def _stream(self, command: list[str], user: str) -> str:
        try:
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=_workdir(),
                env=_child_env(),
            )
        except OSError as exc:
            raise LLMCallError(f"could not start claude: {exc}") from None

        timed_out = threading.Event()

        def _kill() -> None:
            timed_out.set()
            proc.kill()

        timer = threading.Timer(self._timeout, _kill)
        timer.start()
        stderr_chunks: list[str] = []
        # Drain stderr on its own thread so a chatty child can never fill the pipe
        # and stall stdout.
        drain = threading.Thread(target=lambda: stderr_chunks.append(proc.stderr.read()), daemon=True)
        drain.start()
        final: dict | None = None
        try:
            try:
                proc.stdin.write(user)
                proc.stdin.close()
            except OSError:
                pass  # the child died early; its exit code and stderr explain why
            for line in proc.stdout:
                event = _parse_json(line)
                if not isinstance(event, dict):
                    continue
                if event.get("type") == "stream_event":
                    delta = (event.get("event") or {}).get("delta") or {}
                    if delta.get("type") == "text_delta" and delta.get("text"):
                        self._on_token(delta["text"])
                elif event.get("type") == "result":
                    final = event
            proc.wait()
        finally:
            timer.cancel()
            if proc.poll() is None:
                proc.kill()
                proc.wait()
            drain.join(timeout=2)
        if timed_out.is_set():
            raise LLMCallError(f"claude timed out after {self._timeout:g}s")
        return _result_text(json.dumps(final) if final else "", proc.returncode, "".join(stderr_chunks))


def _parse_json(text: str):
    try:
        return json.loads(text)
    except ValueError:
        return None


def _result_text(stdout: str, returncode: int, stderr: str) -> str:
    """The model's text from `claude --output-format json`, or LLMCallError."""
    data = _parse_json(stdout.strip()) if stdout.strip() else None
    if isinstance(data, list):  # some versions emit the event list
        data = next((e for e in reversed(data) if isinstance(e, dict) and e.get("type") == "result"), None)
    if isinstance(data, dict) and data.get("type") == "result":
        text = data.get("result")
        if data.get("is_error") or not isinstance(text, str):
            raise LLMCallError(f"claude reported an error: {str(text or data.get('subtype') or 'unknown')[:500]}")
        return text
    detail = (stderr or stdout or "").strip()[:500]
    raise LLMCallError(f"claude exited with code {returncode} and no result" + (f": {detail}" if detail else ""))


def _selftest() -> int:
    """One real call, so you can see the subscription route work end to end."""
    model = sys.argv[1] if len(sys.argv) > 1 else MODEL_PREFIX + "claude-sonnet-5-5"
    seen: list[str] = []
    llm = ClaudeCodeLLM(model, on_token=seen.append)
    print(f"binary : {llm._binary}")
    print(f"model  : {llm.model} (claude --model {cli_model(llm.model)})")
    reply = llm.complete("You are a terse test responder.", "Reply with only the single word PONG.")
    print(f"reply  : {reply!r}  ({len(seen)} streamed chunk(s))")
    return 0 if "PONG" in reply.upper() else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
