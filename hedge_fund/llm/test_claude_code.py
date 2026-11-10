"""The Claude Code provider: runs prompts through the local `claude` command.

Nothing here starts a real `claude` process. `subprocess` is replaced, so these tests
pin down what we ask the command to do and how its answer is read: the call must
use the subscription login (no API key, no --bare), stay small (--safe-mode, no
tools), and turn every failure into LLMCallError so an agent abstains cleanly.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from hedge_fund.llm import ClaudeCodeLLM, LLMCallError, make_llm, provider_for
from hedge_fund.llm import claude_code
from hedge_fund.llm.claude_code import cli_model, find_claude_binary
from hedge_fund.llm.registry import env_var_for, is_supported, load_api_models

MODEL = "claude-code/claude-sonnet-5-5"


@pytest.fixture(autouse=True)
def fake_binary(monkeypatch):
    monkeypatch.setenv("CLAUDE_CODE_BIN", sys.executable)


def _result(text="hello", is_error=False):
    return json.dumps({"type": "result", "subtype": "success", "is_error": is_error, "result": text})


class _Run:
    """Stands in for subprocess.run and records the call."""

    def __init__(self, stdout=None, returncode=0, stderr="", raises=None):
        self.stdout = _result() if stdout is None else stdout
        self.returncode = returncode
        self.stderr = stderr
        self.raises = raises
        self.command = None
        self.kwargs = None
        self.system_file_text = None

    def __call__(self, command, **kwargs):
        self.command, self.kwargs = command, kwargs
        # The system prompt lives in a temp file that is deleted after the call,
        # so read it while the "process" is running.
        self.system_file_text = Path(command[command.index("--system-prompt-file") + 1]).read_text(encoding="utf-8")
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(command, self.returncode, self.stdout, self.stderr)


def test_registry_knows_claude_code_without_a_key():
    assert provider_for(MODEL) == "Claude Code"
    assert provider_for("claude-code/opus") == "Claude Code"  # unlisted ids still route
    assert is_supported("Claude Code")
    assert env_var_for("Claude Code") is None  # the picker then says "no key is needed"
    assert {p for _, _, p in load_api_models()} >= {"Claude Code"}


def test_make_llm_builds_a_claude_code_client_with_no_keys(monkeypatch):
    for variable in ("ANTHROPIC_API_KEY", "HEDGE_FUND_LLM_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    llm = make_llm(MODEL)
    assert isinstance(llm, ClaudeCodeLLM)
    assert llm.model == MODEL
    assert llm._timeout >= 300  # the CLI is slower than an API call


def test_cli_model_strips_only_our_routing_prefix():
    assert cli_model("claude-code/opus") == "opus"
    assert cli_model(MODEL) == "claude-sonnet-5-5"
    assert cli_model("claude-opus-5-5") == "claude-opus-5-5"


def test_complete_uses_subscription_flags_and_returns_the_text(monkeypatch):
    run = _Run(stdout=_result("the answer"))
    monkeypatch.setattr(subprocess, "run", run)
    reply = ClaudeCodeLLM(MODEL).complete("SYSTEM PROMPT", "USER PROMPT")

    assert reply == "the answer"
    command = run.command
    assert command[0] == sys.executable
    assert "-p" in command
    assert "--safe-mode" in command  # small, login kept
    assert "--bare" not in command  # --bare would ignore the subscription login
    assert command[command.index("--model") + 1] == "claude-sonnet-5-5"
    assert command[command.index("--tools") + 1] == ""  # pure text, no tools
    assert "--no-session-persistence" in command
    assert command[command.index("--output-format") + 1] == "json"
    assert run.kwargs["input"] == "USER PROMPT"  # user text goes over stdin, not argv
    assert run.system_file_text == "SYSTEM PROMPT"
    assert not Path(command[command.index("--system-prompt-file") + 1]).exists()  # temp file cleaned up


def test_api_billing_variables_never_reach_the_child(monkeypatch):
    for variable in claude_code._BILLING_ENV:
        monkeypatch.setenv(variable, "set-by-a-test")
    monkeypatch.setenv("KEEP_ME", "yes")
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    ClaudeCodeLLM(MODEL).complete("s", "u")
    env = run.kwargs["env"]
    assert not set(claude_code._BILLING_ENV) & set(env), "an API key in the child would bill the API instead of the plan"
    assert env["KEEP_ME"] == "yes"


def test_blank_system_prompt_gets_a_fallback(monkeypatch):
    run = _Run()
    monkeypatch.setattr(subprocess, "run", run)
    ClaudeCodeLLM(MODEL).complete("   ", "u")
    assert run.system_file_text == claude_code._FALLBACK_SYSTEM


@pytest.mark.parametrize(
    "run, message",
    [
        (_Run(stdout=_result("Not logged in", is_error=True)), "Not logged in"),
        (_Run(stdout="", returncode=1, stderr="boom"), "boom"),
        (_Run(stdout="not json at all", returncode=0), "no result"),
        (_Run(raises=subprocess.TimeoutExpired("claude", 5)), "timed out"),
        (_Run(raises=OSError("no such file")), "could not start"),
    ],
)
def test_every_failure_is_an_LLMCallError(monkeypatch, run, message):
    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(LLMCallError, match=message):
        ClaudeCodeLLM(MODEL).complete("s", "u")


def test_missing_binary_says_how_to_fix_it(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_BIN")
    monkeypatch.setattr(claude_code.shutil, "which", lambda _name: None)
    with pytest.raises(ValueError, match="CLAUDE_CODE_BIN"):
        find_claude_binary()
    monkeypatch.setenv("CLAUDE_CODE_BIN", str(Path(sys.executable).with_name("definitely-not-here.exe")))
    with pytest.raises(ValueError, match="not a file"):
        find_claude_binary()


class _FakeProc:
    """Stands in for subprocess.Popen on the streaming path."""

    def __init__(self, stdout_lines, returncode=0, stderr=""):
        import io

        self.stdin = io.StringIO()
        self.stdin.close = lambda: None  # keep the written text readable
        self.stdout = iter(stdout_lines)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode

    def poll(self):
        return self.returncode

    def wait(self):
        return self.returncode

    def kill(self):
        pass


def _delta(text):
    return json.dumps({"type": "stream_event", "event": {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}}}) + "\n"


def test_streaming_feeds_the_listener_and_returns_the_whole_text(monkeypatch):
    lines = [
        json.dumps({"type": "system", "subtype": "init"}) + "\n",
        _delta("1, "),
        _delta("2, 3"),
        json.dumps({"type": "assistant"}) + "\n",
        _result("1, 2, 3") + "\n",
    ]
    proc = _FakeProc(lines)
    seen = []
    captured = {}

    def popen(command, **kwargs):
        captured["command"] = command
        return proc

    monkeypatch.setattr(subprocess, "Popen", popen)
    reply = ClaudeCodeLLM(MODEL, on_token=seen.append).complete("s", "count")

    assert seen == ["1, ", "2, 3"]
    assert reply == "1, 2, 3"
    assert proc.stdin.getvalue() == "count"
    command = captured["command"]
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--include-partial-messages" in command and "--verbose" in command


def test_streaming_without_a_result_is_an_error(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: _FakeProc([_delta("partial")], returncode=1, stderr="died"))
    with pytest.raises(LLMCallError, match="died"):
        ClaudeCodeLLM(MODEL, on_token=lambda _t: None).complete("s", "u")
