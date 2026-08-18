"""R3 C3.1 — embedding backend registry, live base-URL routing, and
excluded-backend errors (RT1, RT2, RT4).

RT1: ``EMBEDDING_BACKENDS`` exposes exactly ``{"ollama", "openai"}`` (C5).
RT2: ``_call_embeddings`` fails fast for excluded backends
(``claude-cli``/``bedrock``/``claude``) with a ``ValueError`` naming the
supported set — no route through, no fall-through 404.
RT4: ``resolve_embedding_backend("ollama")`` resolves the configured local
gateway when ``OPENAI_BASE_URL`` is set (the Bifrost route the stack uses),
else the direct local Ollama ``http://localhost:11434/v1`` — never a paid
``https://api.*`` cloud endpoint in either arm, resolved LIVE at call time.
"""
from __future__ import annotations

import pytest

from graphify.embed import _call_embeddings
from graphify.embeddings import EMBEDDING_BACKENDS, resolve_embedding_backend


def test_embedding_backends_exposes_exactly_ollama_and_openai():
    """RT1 — the embedding registry is exactly ``{"ollama", "openai"}``,
    independent of the chat ``BACKENDS`` shape (C5)."""
    assert set(EMBEDDING_BACKENDS) == {"ollama", "openai"}


def test_excluded_backends_raise_valueerror_naming_supported_set():
    """RT2 — ``claude-cli``/``bedrock``/``claude`` are excluded from the
    embedding seam and must fail fast with a ``ValueError`` naming the
    supported set (both discriminators appear: the excluded backend's
    "claude" and the supported "openai"), never touching the network."""
    for backend in ("claude-cli", "bedrock", "claude"):
        with pytest.raises(ValueError) as excinfo:
            _call_embeddings(backend, "nomic-embed-text", ["some text"])
        message = str(excinfo.value)
        assert "claude" in message
        assert "openai" in message


def test_ollama_resolves_to_openai_base_url_gateway_when_set(monkeypatch):
    """RT4 arm 1 — with ``OPENAI_BASE_URL`` set, ``ollama`` resolves to it
    verbatim (the Bifrost gateway route the stack already uses)."""
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:9765/openai/v1")
    assert resolve_embedding_backend("ollama") == "http://localhost:9765/openai/v1"


def test_ollama_resolves_local_when_no_gateway_env_set(monkeypatch):
    """RT4 arm 2 — with ``OPENAI_BASE_URL``, ``OLLAMA_BASE_URL`` and
    ``OLLAMA_HOST`` all unset, ``ollama`` resolves to the direct local
    ``http://localhost:11434/v1`` (never a cloud endpoint)."""
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert resolve_embedding_backend("ollama") == "http://localhost:11434/v1"


def _install_openai_recorder(monkeypatch):
    """RT3/RT5 seam — monkeypatch ``openai.OpenAI`` (the repo's established
    pattern, test_ollama_retry_cap.py monkeypatches the same string target) with
    a recording stub: every ``embeddings.create`` call appends its input list to
    ``call_log``, every client construction records its kwargs, and responses
    carry one vector per input keyed to the input's GLOBAL position (so a
    dropped/reordered concatenation is visible). Returns ``(call_log,
    client_kwargs)``."""

    class _Item:
        def __init__(self, index, embedding):
            self.index = index
            self.embedding = embedding

        def model_dump(self):
            return {"index": self.index, "embedding": self.embedding}

    class _Response:
        def __init__(self, model, data):
            self.model = model
            self.data = data

        def model_dump(self):
            return {"model": self.model, "data": [d.model_dump() for d in self.data]}

    class _Embeddings:
        def __init__(self):
            self._pos = 0

        def create(self, model, input):
            batch = list(input)
            call_log.append(batch)
            data = [
                _Item(
                    self._pos + i,
                    [float(self._pos + i + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                )
                for i in range(len(batch))
            ]
            self._pos += len(batch)
            return _Response(model=model, data=data)

    class _Client:
        def __init__(self, **kwargs):
            client_kwargs.append(kwargs)
            self.embeddings = _Embeddings()

    call_log: list[list[str]] = []
    client_kwargs: list[dict] = []
    monkeypatch.setattr("openai.OpenAI", _Client)
    return call_log, client_kwargs


def test_ollama_embedding_client_retries_zero_default_override_wins(monkeypatch):
    """RT3 — the embedding seam applies the SAME 0-retry-ollama rule as the chat
    path (llm.py:1198): with GRAPHIFY_MAX_RETRIES unset, ``_call_embeddings(
    "ollama", ...)`` constructs its OpenAI-compat client with ``max_retries ==
    0`` (a local server does not rate-limit, #1686); with
    GRAPHIFY_MAX_RETRIES="3" the same call constructs ``max_retries == 3``.
    Either way the two inputs issue exactly ONE ``embeddings.create`` call.
    RED today: the stub returns deterministic vectors without ever constructing
    a client, so the client_kwargs assertion fails first."""
    call_log, client_kwargs = _install_openai_recorder(monkeypatch)

    monkeypatch.delenv("GRAPHIFY_MAX_RETRIES", raising=False)
    out = _call_embeddings("ollama", "nomic-embed-text", ["a", "b"])
    assert client_kwargs, (
        "_call_embeddings('ollama', ...) must construct an OpenAI client; the "
        "current stub never calls the network"
    )
    assert client_kwargs[0]["max_retries"] == 0, (
        "ollama must default to 0 SDK retries when GRAPHIFY_MAX_RETRIES is unset"
    )
    assert call_log == [["a", "b"]], (
        f"two inputs must issue exactly one embeddings.create call, got {call_log!r}"
    )
    assert len(out) == 2, "one vector per input"

    call_log.clear()
    client_kwargs.clear()
    monkeypatch.setenv("GRAPHIFY_MAX_RETRIES", "3")
    out = _call_embeddings("ollama", "nomic-embed-text", ["a", "b"])
    assert client_kwargs, "no OpenAI client constructed for the override arm"
    assert client_kwargs[0]["max_retries"] == 3, (
        "an explicit GRAPHIFY_MAX_RETRIES must override the ollama 0-retry default"
    )
    assert call_log == [["a", "b"]], (
        f"two inputs must issue exactly one embeddings.create call, got {call_log!r}"
    )
    assert len(out) == 2, "one vector per input"


def test_embeddings_batch_250_into_three_calls_in_input_order(monkeypatch):
    """RT5 — batching: with the module batch size of 100, 250 inputs issue
    exactly THREE ``embeddings.create`` calls — the first two carry 100 inputs,
    the last 50 — and the concatenated return has exactly one vector per input,
    in input order. RED today: the stub issues no create calls at all, so the
    call-count assertion fails first."""
    call_log, _ = _install_openai_recorder(monkeypatch)
    inputs = [f"text-{i:03d}" for i in range(250)]

    out = _call_embeddings("ollama", "nomic-embed-text", inputs)

    assert len(call_log) == 3, (
        f"250 inputs at batch size 100 must issue exactly 3 create calls, got {len(call_log)}"
    )
    assert [len(batch) for batch in call_log] == [100, 100, 50], (
        f"batch sizes must be 100/100/50, got {[len(b) for b in call_log]!r}"
    )
    assert call_log[0] == inputs[:100], "first batch must be inputs[0:100], in input order"
    assert call_log[1] == inputs[100:200], "second batch must be inputs[100:200], in input order"
    assert call_log[2] == inputs[200:250], "last batch must be inputs[200:250], in input order"
    expected = [
        [float(i + 1), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] for i in range(len(inputs))
    ]
    assert out == expected, (
        "the concatenated return must have exactly one vector per input, in input "
        "order — vector i belongs to input i, never dropped or reordered"
    )
