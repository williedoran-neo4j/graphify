"""T7 — semantic_search cosine path: deterministic ranking + pinned render.

C2.2 introduces ``graphify/search.py`` (``load_sidecar`` / ``_stub_query_embed`` /
``search_vectors``) and the ``_tool_semantic_search`` render surface in serve.py.
The fixture ``.npz`` follows R1's sidecar layout (``embed.py``): ``text_ids`` in
``sorted(graph.nodes)`` id order, ``text_vecs`` rows L2-normalized float32, plus
``text_meta``. Every row is a hand-laid identity-basis unit vector, so cosine is
a plain dot product (R1 I3) *and* rows are mutually distinguishable. (R1's real
all-positive ``[i+1]*dim`` rows are mutually parallel once L2-normalized — every
cosine ties — so they cannot drive a ranking fixture; that is why T7's fixture
uses hand-laid orthonormal vectors instead.)

Deterministic ranking contract (the sole-reason / false-lock fix)
----------------------------------------------------------------
Fixture rows are stored in sorted-id order (n-a-01 -> e0, n-b-02 -> e1,
n-c-03 -> e2). The C4 query-embed stub ``_stub_query_embed`` is deterministic
per query; for the literal query ``"query"`` it must dominate the SECOND row's
coordinate, with the first row next and the third last:

    q[1] > q[0] > q[2] > 0     (canonical: [1.0, 2.0, 0.5, 0, 0, 0, 0, 0])

score_j = dot(q, e_j) = q[j], so the fixture dictates scores
b=2.0 > a=1.0 > c=0.5 and the locked ranking:

    ["n-b-02", "n-a-01", "n-c-03"]

which is a real PERMUTATION of the stored sorted-id order. A no-rank
implementation that returns rows in stored/lexicographic order
["n-a-01", "n-b-02", "n-c-03"] VISIBLY fails the id-order assertion.
(The earlier expectation, which EQUALED stored row order, could not
distinguish rank-by-cosine from "return as stored" — hence this restructure.)
"""
from __future__ import annotations

import asyncio
import json
import re
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp.types import CallToolRequest, CallToolRequestParams  # noqa: E402

from graphify import serve as serve_mod  # noqa: E402
from graphify.embed import _STUB_DIM  # noqa: E402
from graphify.search import load_sidecar, search_vectors  # noqa: E402
import graphify.search as search_mod  # noqa: E402

# Sorted-id order — the R1 sidecar rows live in this order (embed.py sorts).
IDS = ("n-a-01", "n-b-02", "n-c-03")
LABELS = ("Alpha embed", "Beta embed", "Gamma embed")
FILE_TYPES = ("document", "concept", "document")
SOURCE_FILES = ("docs/a.md", "docs/b.md", "docs/c.md")
# Hand-laid ORTHONORMAL identity-basis unit rows, in IDS order. L2 norms come
# out exactly 1.0, mirroring R1's normalize-on-write (I3), and the rows are
# mutually orthogonal so score_j == q[j] for any query embed q.
_ORTHO = np.array(
    [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)
assert _ORTHO.shape == (len(IDS), _STUB_DIM), "ortho rows must be _STUB_DIM-wide"


def _write_sidecar(tmp_path: Path) -> Path:
    """Write a hand-built embeddings.npz next to a minimal graph.json in tmp_path."""
    (tmp_path / "graph.json").write_text(json.dumps({"directed": True, "nodes": [], "edges": []}))
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        text_ids=np.array(IDS, dtype=str),
        text_vecs=_ORTHO,
        text_meta=json.dumps(meta),
    )
    return path


def _graph_file(tmp_path: Path, labels: tuple[str, str, str] = LABELS) -> Path:
    """graph.json whose node labels are ``labels`` (default: the clean LABELS)."""
    g = {
        "directed": True,
        "nodes": [
            {"id": IDS[0], "label": labels[0], "source_file": SOURCE_FILES[0],
             "file_type": FILE_TYPES[0], "community": 0},
            {"id": IDS[1], "label": labels[1], "source_file": SOURCE_FILES[1],
             "file_type": FILE_TYPES[1], "community": 0},
            {"id": IDS[2], "label": labels[2], "source_file": SOURCE_FILES[2],
             "file_type": FILE_TYPES[2], "community": 0},
        ],
        "edges": [],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(g), encoding="utf-8")
    return path


def test_load_sidecar_roundtrips_arrays(tmp_path):
    """T7 direct — load_sidecar returns the stored text_* arrays: the unicode
    ids and the float32 vectors, plus a parseable meta dict; a missing .npz
    yields None."""
    sidecar = _write_sidecar(tmp_path)
    loaded = load_sidecar(sidecar)
    assert loaded is not None
    assert list(loaded["text_ids"]) == list(IDS)
    assert loaded["text_vecs"].dtype == np.float32
    assert loaded["text_vecs"].shape == (len(IDS), _STUB_DIM)
    assert json.loads(str(loaded["text_meta"]))["dim"] == _STUB_DIM
    assert load_sidecar(tmp_path / "missing.npz") is None


def test_search_vectors_ranks_descending_with_real_query_embed(tmp_path):
    """T7 direct — search_vectors ranks by cosine (dot product over unit rows),
    NEVER by stored row order. With the C4 stub embed satisfying q[1] > q[0] >
    q[2] (canonical [1.0, 2.0, 0.5, ...]), score_j = q[j] and the locked ranking
    is ["n-b-02", "n-a-01", "n-c-03"] — a real permutation of the stored
    sorted-id order ["n-a-01", "n-b-02", "n-c-03"]. A stored-order/no-sort
    implementation fails the first id assertion (sole-reason lock)."""
    sidecar = _write_sidecar(tmp_path)
    rows = search_vectors(sidecar, "query", space="text")
    assert [r["id"] for r in rows] == ["n-b-02", "n-a-01", "n-c-03"]
    scores = [r["score"] for r in rows]
    assert scores[0] > scores[-1], "scores must be strictly ranking, not tied"
    assert scores == sorted(scores, reverse=True), "rows must be score-descending"


def test_semantic_search_renders_pinned_text_surface(tmp_path):
    """T7 render — _tool_semantic_search renders the pinned surface
    f"{score:.3f}  [text]  {nid}  {label}  ({file_type})" in descending score.

    C2.1's handler still returns the placeholder "no semantic matches yet", so
    the RED failure is the missing pins (no [text] token). Once green lands,
    against this fixture and the canonical stub embed the exact rendering is:

        2.000  [text]  n-b-02  Beta embed  (concept)
        1.000  [text]  n-a-01  Alpha embed  (document)
        0.500  [text]  n-c-03  Gamma embed  (document)

    The assertions pin the format and the DESCENDING order without over-pinning
    the stub embed's magnitude (any q with q[1] > q[0] > q[2] > 0.3 — the tool's
    default min_score — yields the same order; the canonical q[2]=0.5 clears it).
    """
    _write_sidecar(tmp_path)  # the green compute path reads the sidecar
    server = serve_mod._build_server(str(_graph_file(tmp_path)))
    result_future = server.request_handlers[CallToolRequest](
        CallToolRequest(params=CallToolRequestParams(name="semantic_search", arguments={"query": "query"}))
    )
    if asyncio.iscoroutine(result_future):
        result = asyncio.run(result_future)
    else:  # newer mcp majors may return an awaitable/wrapped result directly
        result = result_future
    result_text = result.root.content[0].text
    assert "[text]" in result_text, (
        "render lacks the [text] space literal (is it still the C2.1 placeholder?)"
    )
    assert "(document)" in result_text, "render lacks the (file_type) suffix"
    lines = result_text.splitlines()
    assert len(lines) == 3, f"expected 3 ranked rows, got {lines!r}"
    rendered_ids: list[str] = []
    rendered_scores: list[float] = []
    for line in lines:
        m = re.fullmatch(r"(\d+\.\d{3})  \[text\]  (\S+)  (.+)  \((\w+)\)", line)
        assert m, (
            f"{line!r} does not match the pinned "
            "`{score:.3f}  [text]  {nid}  {label}  ({file_type})` surface"
        )
        rendered_scores.append(float(m.group(1)))
        rendered_ids.append(m.group(2))
    assert rendered_ids == ["n-b-02", "n-a-01", "n-c-03"], (
        "rows not rendered in the fixture-dictated descending order"
    )
    assert rendered_scores == sorted(rendered_scores, reverse=True), (
        "scores must be rendered score-descending"
    )


def test_search_vectors_min_score_drops_strictly_below_keeps_exact(tmp_path):
    """T8 — min_score excludes rows whose cosine is < min_score, and KEEPS a
    row at exactly == min_score (strict `<` drop). Sole-reason lock: against
    the identity fixture + canonical stub embed, score_j == q[j] and the exact
    float32-representable scores are b=2.0, a=1.0, c=0.5.
      * min_score=3.0 (above the top row)  -> empty result;
      * min_score=0.75 (between rows)      -> only [n-b-02, n-a-01], in order;
      * min_score=1.0 (== row a's score)   -> n-a-01 SURVIVES, n-c-03 dropped.
    A `>` (strict-greater) drop fails the last assertion (a would vanish); a
    missing filter leaves n-c-03 present and fails the first three.
    """
    sidecar = _write_sidecar(tmp_path)
    assert search_vectors(sidecar, "query", space="text", min_score=3.0) == [], (
        "min_score above the top row must yield an empty result"
    )
    rows = search_vectors(sidecar, "query", space="text", min_score=0.75)
    assert [r["id"] for r in rows] == ["n-b-02", "n-a-01"], (
        "rows with cosine < 0.75 must be excluded, survivors in score-desc order"
    )
    rows_at_boundary = search_vectors(sidecar, "query", space="text", min_score=1.0)
    assert [r["id"] for r in rows_at_boundary] == ["n-b-02", "n-a-01"], (
        "a row at exactly == min_score (n-a-01, score 1.0) must be KEPT; "
        "n-c-03 (0.5 < 1.0) must be dropped"
    )


def test_search_vectors_top_k_cuts_after_ranking(tmp_path):
    """T7 tail (C2.5) — the top_k cut happens AFTER ranking, so the survivors are
    the top-SCORING rows (never the first-stored ones), and top_k=0 yields no
    rows while the default returns everything.

    Sole-reason lock: fixture rows are stored in sorted-id order
    n-a-01(e0, score 1.0), n-b-02(e1, score 2.0), n-c-03(e2, score 0.5), so a
    "cut the first N stored rows BEFORE ranking" implementation returns
    [n-a-01, n-b-02] for top_k=2 — the WRONG ids (scores 1.0/2.0, not desc).
    The top-two-scoring ids are [n-b-02, n-a-01] (scores 2.0/1.0), so asserting
    exactly those ids (a real permutation of the first-two-stored) distinguishes
    cut-after-ranking from cut-before-ranking. top_k=0 -> [] locks the empty-cut
    boundary; the default (top_k=10) -> all 3 rows score-descending.
    """
    sidecar = _write_sidecar(tmp_path)
    assert [r["id"] for r in search_vectors(sidecar, "query", space="text", top_k=2)] == [
        "n-b-02",
        "n-a-01",
    ], "top_k must keep the top TWO SCORING ids, not the first two stored rows"
    # top_k=1 pins the discriminator: the correct top-1 is n-b-02 (second-stored,
    # q[1]=2.0 > q[0]=1.0), so a cut-then-resort implementation
    # (sorted(rows[:top_k], reverse=True)) returns n-a-01 here and FAILS — the
    # top_k=2 arm above cannot discriminate, because the top-2-scoring set
    # happens to equal the first-2-stored set.
    assert [r["id"] for r in search_vectors(sidecar, "query", space="text", top_k=1)] == [
        "n-b-02",
    ], "top_k=1 must keep the SINGLE top-scoring id (n-b-02), not the first stored row (n-a-01)"
    assert search_vectors(sidecar, "query", space="text", top_k=0) == [], (
        "top_k=0 must cut every row"
    )
    all_rows = search_vectors(sidecar, "query", space="text", top_k=10)
    assert [r["id"] for r in all_rows] == ["n-b-02", "n-a-01", "n-c-03"], (
        "top_k=10 must keep all rows in score-desc order"
    )
    assert [r["id"] for r in search_vectors(sidecar, "query", space="text")] == [
        "n-b-02",
        "n-a-01",
        "n-c-03",
    ], "the default top_k must keep all rows in score-desc order"


def test_search_vectors_file_type_filter_reads_injected_lookup(tmp_path):
    """T9 — the file_type allow-set must be resolved through the INJECTED per-id
    lookup, never a hardcoded table and never `""`-for-everything.

    The seam design: ``search_vectors(..., file_type_lookup=node_file_type)`` and
    the serve handler passes a closure over the live graph. ``node_file_type``
    (search.py:87-90) currently returns ``""`` for EVERY id, so on today's code
    any allow-set silently drops ALL rows — that is the red this test locks.
    ``search_vectors`` does not accept a ``file_type_lookup`` keyword yet, so the
    call itself fails (or, if green only adds an ignored/empty keyword, the
    per-lookup assertions fail). The lookup must be injected because the sole
    fixture file_type values are bound to the fixture ids; a filter hardcoded to
    those ids would pass the T9 arms and wrongly survive. The guard below maps
    EVERY id to ``"video"`` and asserts ``file_type=["video"]`` keeps all three
    rows AND that the score-desc order is preserved — only a filter that consults
    the actual injected lookup can pass while a hardcoded/inert filter cannot.

    Fixture file_types (n-a-01 -> document, n-b-02 -> concept, n-c-03 ->
    document) + identity basis + canonical stub embed (q[1] > q[0] > q[2]):
    score_document = 1.0 > 0.5, so the survivors render score-descending.
    """
    sidecar = _write_sidecar(tmp_path)

    fixture_types = dict(zip(IDS, FILE_TYPES, strict=True))
    real_lookup = lambda nid: fixture_types.get(nid, "")  # noqa: E731

    # Guard arm: every id resolves to "video"; the filter must consult the
    # INJECTED lookup, not a fixture-bound hardcoded table. On today's code the
    # keyword doesn't exist, so this call errors out — the red.
    all_rows = search_vectors(
        sidecar,
        "query",
        space="text",
        file_type=["video"],
        file_type_lookup=lambda nid: "video",
    )
    assert [r["id"] for r in all_rows] == ["n-b-02", "n-a-01", "n-c-03"], (
        "file_type filter must resolve types through the injected file_type_lookup; "
        "an inert or ""-for-everything lookup wrongly drops ALL rows"
    )

    # T9 arms against the REAL fixture types.
    docs = search_vectors(sidecar, "query", space="text", file_type=["document"], file_type_lookup=real_lookup)
    assert [r["id"] for r in docs] == ["n-a-01", "n-c-03"], (
        "document rows must survive in score-desc order: n-a-01 (1.0) above n-c-03 (0.5); "
        "a missing or ""-returning file_type lookup drops ALL rows"
    )
    concept_rows = search_vectors(sidecar, "query", space="text", file_type=["concept"], file_type_lookup=real_lookup)
    assert [r["id"] for r in concept_rows] == ["n-b-02"]
    assert search_vectors(sidecar, "query", space="text", file_type=["audio"], file_type_lookup=real_lookup) == [], (
        "a file_type matching nothing must yield []"
    )


def _invoke_semantic_search(server, query="query", **extra):
    """Invoke the semantic_search tool on the built server and return its text."""
    arguments = {"query": query, **extra}
    result_future = server.request_handlers[CallToolRequest](
        CallToolRequest(
            params=CallToolRequestParams(name="semantic_search", arguments=arguments)
        )
    )
    if asyncio.iscoroutine(result_future):
        result = asyncio.run(result_future)
    else:  # newer mcp majors may return an awaitable/wrapped result directly
        result = result_future
    return result.root.content[0].text


def test_building_server_never_opens_embeddings_npz(tmp_path, monkeypatch):
    """T14 guard — building a server (or merely listing tools) must NOT open the
    embeddings sidecar; only an actual semantic_search invocation may read it.

    Sole-reason premise: the only code path that may open the .npz is
    search_vectors -> load_sidecar, and that runs exclusively inside the
    semantic_search handler. If ANY of that path ran at build/list time, the
    spy below records an open and the assertion fires. The observation is an
    exact per-path open count, NOT an absent-from-result coarse check: nothing
    else in the build intersects this path, so a single spurious open is
    visible. Expected GREEN (the cache is C2.6's build — no build-time open
    exists today); teeth are proven by a mutator adding an eager build-time
    sidecar load.
    """
    _write_sidecar(tmp_path)
    calls: list[str] = []
    real_load_sidecar = search_mod.load_sidecar

    def spy(path):
        calls.append(str(path))
        return real_load_sidecar(path)

    monkeypatch.setattr(search_mod, "load_sidecar", spy)

    serve_mod._build_server(str(_graph_file(tmp_path)))

    assert calls == [], (
        "building the server must never open embeddings.npz (lazy load); "
        f"got {len(calls)} open(s): {calls}"
    )


def test_semantic_search_sidecar_loaded_once_across_repeated_calls(tmp_path, monkeypatch):
    """C2.6 — the loaded sidecar is memoized ON THE GRAPH OBJECT, so two
    semantic_search calls on the same server open the .npz exactly ONCE.

    Sole-reason premise: the graph-object cache is the ONLY mechanism that can
    dedupe the second call. The file is never rewritten between the calls, so
    the (st_mtime_ns, st_size) key is identical for both — there is no disk
    change for a per-call staleness re-read to notice. Today's handler calls
    search_vectors fresh every invocation, load_sidecar fires on BOTH calls,
    and this assertion fails — the red. If green were to dedupe in
    graph-agnostic search.py's module scope instead, the key would not travel
    with a hot-reloaded G (C2.11's constraint) and a module-level cache would
    wrongly survive a fresh graph object — the graph-object placement is part
    of the behavior under test.
    """
    _write_sidecar(tmp_path)
    calls: list[str] = []
    real_load_sidecar = search_mod.load_sidecar

    def spy(path):
        calls.append(str(path))
        return real_load_sidecar(path)

    monkeypatch.setattr(search_mod, "load_sidecar", spy)

    server = serve_mod._build_server(str(_graph_file(tmp_path)))
    first = _invoke_semantic_search(server)
    second = _invoke_semantic_search(server)

    assert "[text]" in first and "[text]" in second, "render must come from a real search"
    assert len(calls) == 1, (
        "the sidecar must be loaded ONCE and cached on the graph object; "
        f"two semantic_search calls opened it {len(calls)} time(s): {calls}"
    )


def test_semantic_search_neutralizes_control_char_in_rendered_label(tmp_path):
    """C2.7/T11 (F1) — every LLM-derived value the render emits must be routed
    through sanitize_label, so an injected control-char sentinel in a node label
    NEVER reaches the tool's returned text.

    Fixture: n-a-01's label carries "\x1f" (a CHARMATCHER for security.py's
    ``_CONTROL_CHAR_RE`` = ``[\x00-\x1f\x7f]``, security.py:390). sanitize_label
    substitutes an EMPTY STRING for control chars, so the NEUTRALIZED label is
    ``"Alphaembed"`` (``"\x1f"`` deleted, NOT replaced with a space — security.py
    line 402). Rows are otherwise identical to the C2.2 fixture, so the ranking
    stays ["n-b-02" (2.0), "n-a-01" (1.0), "n-c-03" (0.5)] and every OTHER row
    must still match the pinned C2.2 render regex — no regression to the normal
    render surface.

    Sole-reason lock: today the render emits label VERBATIM (serve.py:1956,
    ``G.nodes[r['id']].get('label', r['id'])``), so the raw "\x1f" byte lands in
    the returned text and the sentinel-absence assert below fails RED.

    Guarded variants must ALL go red with today's code:
      * strip control chars from only the id/score but pass the label through
        verbatim — the "\x1f" still leaks (sentinel-absence assert fails);
      * strip NULLs or a "\n"-specific pattern only — "\x1f" survives;
      * drop the poisoned row entirely — the n-a-01 presence assert bars the
        cheap "filter out rows with a dirty label" escape (n-a-01's label would
        not be RENDERED, so a raw-sentinel scan would pass vacuously).
    """
    sentinel = "\x1f"
    assert sentinel in "Alpha\x1fembed", "sentinel must be a byte in the fixture label"
    poisoned_labels = ("Alpha\x1fembed", LABELS[1], LABELS[2])
    _write_sidecar(tmp_path)  # ids/vectors unchanged — only graph.json carries the sentinel
    server = serve_mod._build_server(str(_graph_file(tmp_path, labels=poisoned_labels)))
    result_future = server.request_handlers[CallToolRequest](
        CallToolRequest(params=CallToolRequestParams(name="semantic_search", arguments={"query": "query"}))
    )
    if asyncio.iscoroutine(result_future):
        result = asyncio.run(result_future)
    else:  # newer mcp majors may return an awaitable/wrapped result directly
        result = result_future
    result_text = result.root.content[0].text
    assert sentinel not in result_text, (
        "rendered text must NOT contain the raw control-char sentinel — "
        "serve.py must route the rendered label through sanitize_label"
    )
    # Guard: the poisoned ROW must still render (not "empty-or-dropped").
    assert "n-a-01" in result_text, (
        "the poisoned row must itself render (id present) — dropping the row "
        "would be an invalid fix this lock forbids"
    )
    # The neutralized label is the exact sanitize_label output (control char
    # DELETED — no space inserted — and not truncated).
    assert "Alphaembed" in result_text, (
        f"label must render NEUTRALIZED as sanitize_label's exact output; got "
        f"{result_text!r}"
    )
    # Every OTHER row must still match the pinned C2.2 render surface.
    assert "[text]" in result_text, "render lacks the [text] space literal"
    lines = result_text.splitlines()
    assert len(lines) == 3, f"expected 3 rendered rows, got {lines!r}"
    assert re.fullmatch(
        r"(\d+\.\d{3})  \[text\]  (\S+)  (.+)  \((\w+)\)", lines[0]
    ), f"{lines[0]!r} no longer matches the pinned render surface"
    assert re.fullmatch(
        r"(\d+\.\d{3})  \[text\]  (\S+)  (.+)  \((\w+)\)", lines[1]
    ), f"{lines[1]!r} no longer matches the pinned render surface"
    assert re.fullmatch(
        r"(\d+\.\d{3})  \[text\]  (\S+)  (.+)  \((\w+)\)", lines[2]
    ), f"{lines[2]!r} no longer matches the pinned render surface"


def test_semantic_search_absent_sidecar_returns_embed_instruction(tmp_path):
    """T12 (C2.8) — a server pointed at a graph dir with NO embeddings.npz must
    answer semantic_search with the run-`graphify embed .` instruction (C4:
    "returns the literal instruction text to run ``graphify embed .`` ... NOT an
    exception", spec lines 159-162) — never ``"No semantic matches."`` and never
    an "Error executing ..." alias.

    Sole-reason lock (the absent-vs-empty split): today the handler's single
    ``if not rows:`` branch (serve.py:1954-1955) returns the EXACT
    ``"No semantic matches."`` string for BOTH the absent-sidecar case
    (``search_vectors`` returns ``None``) and the filtered-empty case
    (``rows == []``). So under today's code:
      * the literal-`graphify embed .` assertion FAILS (absent == the empty text);
      * the NOT-``"No semantic matches."`` assertion FAILS (absent == that text);
      * a raised/aliased exception is ALSO a failure now and after green: the
        call_tool alias (serve.py:2060) rewrites every exception into
        ``"Error executing semantic_search: ..."``, which contains no literal —
        so only a successful tool result carrying the run-instruction can pass.
    A blanket "handle absent == empty" implementation keeps absent equal to the
    empty text and fails both asserts; only the None-seam split passes.
    """
    _graph_file(tmp_path)  # graph.json only — embeddings.npz ABSENT
    server = serve_mod._build_server(str(Path(tmp_path).resolve() / "graph.json"))

    # Direct handler-level red FIRST (no args besides the query): today this
    # renders "No semantic matches." — the red is unpolluted by any exception
    # alias, so this is the clean unimplemented-split failure.
    assert "Error executing" not in _invoke_semantic_search(server, query="query"), (
        "the absent-sidecar path must never produce the call_tool error alias — "
        "but today the handler returns the empty text, so the 'Error executing' "
        "absence alone cannot discriminate; the literal asserts below do"
    )
    # The load-bearing discriminators (the absent-vs-empty split is sole-reason):
    assert "graphify embed ." in _invoke_semantic_search(server, query="query"), (
        "absent sidecar must return the run-`graphify embed .` instruction; the "
        "blanket 'No semantic matches.' fallback (today's code) fails this"
    )
    assert "No semantic matches." not in _invoke_semantic_search(server, query="query"), (
        "the absent case must NOT render the empty-result text — today it does, "
        "and a green that keeps absent == empty fails this"
    )

    # Absent-vs-empty discrimination through the ACTIVE MCP shape (call_tool's
    # alias + project_path handling): the filter arm is C2.3's own min_score — a
    # sidecar whose every score lies BELOW it renders the exact empty text, so
    # the absent-instruction result cannot be "empty in pyjamas". (The fixture
    # scores under the canonical stub embed are 2.0/1.0/0.5, so min_score=3.0
    # filters ALL of them — same boundary value T8 uses for its empty arm.)
    _write_sidecar(tmp_path)
    filtered_empty = _invoke_semantic_search(server, query="query", min_score=3.0)
    assert filtered_empty == "No semantic matches.", (
        "present-but-filtered-empty must STILL render the EXACT empty text "
        "(the split is symmetric — a green returning the instruction here fails)"
    )
    # Populated arm: the same server, same graph dir, a request that CLEARS the
    # filter renders real rows — proving the sidecar is alive and the empty text
    # above came from genuine filtering, not absence.
    rows_text = _invoke_semantic_search(server, query="query", min_score=0.0)
    assert "n-b-02" in rows_text and "[text]" in rows_text, (
        f"with min_score cleared the sidecar must render real rows, got {rows_text!r}"
    )


def _write_sidecar_with_ghost(tmp_path: Path) -> Path:
    """Sidecar carrying the 3 fixture ids PLUS a 4th row ``z-ghost`` whose id is
    ABSENT from the graph — the C2.11 stale-row fixture. ``z-ghost``'s vector
    reuses e0 (n-a-01's identity row), so under the canonical stub embed
    (score_j == q[j]) it scores EXACTLY 1.0 — above the tool's default
    min_score=0.3 and inside the default top_k=10 (4 rows < 10). It must
    actually SURVIVE top_k/min_score or the row is filtered before the render
    and the C2.11 lock could only pass vacuously."""
    (tmp_path / "graph.json").write_text(json.dumps({"directed": True, "nodes": [], "edges": []}))
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    vecs = np.vstack([_ORTHO, _ORTHO[0]])  # the ghost reuses e0 -> score exactly 1.0
    assert vecs.shape == (len(IDS) + 1, _STUB_DIM)
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        text_ids=np.array([*IDS, "z-ghost"], dtype=str),
        text_vecs=vecs,
        text_meta=json.dumps(meta),
    )
    return path


def test_semantic_search_stale_sidecar_row_renders_not_errors(tmp_path):
    """C2.11/F2 (TEST-ONLY lock) — a sidecar row whose id is ABSENT from the
    graph renders id-as-label with an empty ``()`` file_type suffix instead of
    raising KeyError (which call_tool aliases into an
    "Error executing semantic_search: ..." string).

    The ``G.nodes.get(r['id'], {})`` join ALREADY landed with C2.8
    (serve.py:1962-1963) — it is unchanged on this cycle, so the test passes on
    first run (GREEN-ON-ARRIVAL) and the MUTATOR proves teeth by reverting the
    join to ``G.nodes[r['id']]``: the stale ``z-ghost`` row then raises
    ``KeyError``, call_tool's alias (serve.py:2064) renders
    "Error executing semantic_search: 'z-ghost'", and BOTH the no-error assert
    and the exact stale-line assert go red.

    Sole-reason premise (no vacuous pass): ``z-ghost`` scores EXACTLY 1.0 (e0
    reuse, score_j == q[j]) — above the tool's default min_score=0.3 and within
    the default top_k=10 — so it genuinely reaches the render. The discriminator
    substrings are exact: the aliased error text
    "Error executing semantic_search: 'z-ghost'" contains the bare "z-ghost" but
    NOT the doubled-space render "  z-ghost  z-ghost  ()"; and a broken
    implementation that filtered the stale row pre-render (min_score/top_k)
    fails the exact stale-line assert. The pinned render regex ``\\(\\w+\\)``
    cannot match the empty ``()`` suffix, so the stale line is asserted with its
    own exact literal while the 3 real rows stay on the pinned surface.
    """
    _write_sidecar_with_ghost(tmp_path)  # ids = 3 fixture ids + z-ghost
    server = serve_mod._build_server(str(_graph_file(tmp_path)))  # graph keeps only the 3 real nodes
    result_text = _invoke_semantic_search(server, query="query")

    assert "Error executing" not in result_text, (
        f"the stale row must never surface call_tool's exception alias; got {result_text!r}"
    )
    lines = result_text.splitlines()
    assert len(lines) == 4, (
        f"3 real rows + the stale z-ghost row must all render, got {lines!r}"
    )

    # The 3 real rows must still match the pinned C2.2 render surface (no
    # regression); the stale row is the ONE line the pinned regex rejects
    # (`\\(\\w+\\)` needs a word char, `()` has none).
    pinned = r"(\d+\.\d{3})  \[text\]  (\S+)  (.+)  \((\w+)\)"
    real_ids: list[str] = []
    real_scores: list[float] = []
    unmatched: list[str] = []
    for line in lines:
        m = re.fullmatch(pinned, line)
        if m:
            real_ids.append(m.group(2))
            real_scores.append(float(m.group(1)))
        else:
            unmatched.append(line)
    assert real_ids == ["n-b-02", "n-a-01", "n-c-03"], (
        f"the 3 real rows must still render in the pinned descending order; "
        f"got {real_ids!r}"
    )
    assert real_scores == sorted(real_scores, reverse=True), (
        f"real-row scores must stay score-descending, got {real_scores!r}"
    )
    assert len(unmatched) == 1, (
        f"exactly the stale row must fail the pinned regex, got {unmatched!r}"
    )
    stale_text = "1.000  [text]  z-ghost  z-ghost  ()"
    assert unmatched[0] == stale_text, (
        f"the stale row must render id-as-label with an empty () type, got "
        f"{unmatched[0]!r}; expected the exact {stale_text!r}"
    )


def test_semantic_search_every_line_carries_space_text(tmp_path):
    """T10 (C2.10) — every rendered result line carries its `[text]` space
    token; the space literal is PER-LINE, not a one-occurrence-in-the-output
    accident.

    Guard invariant I4: each rendered row names its own space (`"text"` today —
    R2 has exactly one space; R10 later adds `code_*`). Sole-reason lock: a
    render that emits the `[text]` token only on the first line (e.g. a header
    carrying the space while data rows skip it) still satisfies a whole-output
    `"[text]" in text` check, but FAILS this per-line walk — which is the hole
    the C2.10 entry calls out.

    The line count assert (== 3) is not decorative: it fixes the number of
    fixture rows, so the walk genuinely touches every rendered row rather than
    shrinking to a vacuous single-line pass. Dropping `[{space}]` from the
    render (mutator) fires the per-line assert red on every row.
    """
    _write_sidecar(tmp_path)
    server = serve_mod._build_server(str(_graph_file(tmp_path)))
    result_text = _invoke_semantic_search(server, query="query")
    lines = result_text.splitlines()
    assert len(lines) == 3, (
        f"the fixture must render exactly 3 rows for the per-line walk to be "
        f"meaningful, got {len(lines)}: {lines!r}"
    )
    for line in lines:
        assert "[text]" in line, (
            f"every rendered result line must carry its [text] space literal; "
            f"{line!r} does not"
        )


def test_load_sidecar_validates_meta_dim_against_stored_rows(tmp_path):
    """RT7 — load_sidecar is the SINGLE load chokepoint and must enforce
    invariant I5 at load time: a sidecar whose ``text_meta["dim"]`` DIFFERS from
    ``text_vecs.shape[1]`` raises ``ValueError``; a matching sidecar loads fine.
    A malformed sidecar must never flow onward into search_vectors/serve.py —
    I5's "never pass a dimension mix onward" (the write-side guard is the
    seam-side half; this is the load-side half).

    Sole-reason lock: the mismatch fixture differs from the matching one ONLY in
    the stored ``text_meta["dim"]`` value (``_STUB_DIM + 1`` vs ``_STUB_DIM``) —
    rows, ids, and every other meta key are identical. The matching arm is a
    control proving the check is a real comparison and not a blanket reject.

    RED (guaranteed) today: load_sidecar returns the raw arrays unchanged with
    no meta parse and no dim check, so the mismatch arm FAILS on
    ``pytest.raises(ValueError)``.
    """
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }

    mismatched = tmp_path / "embeddings.npz"
    np.savez(
        mismatched,
        text_ids=np.array(IDS, dtype=str),
        text_vecs=_ORTHO,
        text_meta=json.dumps({**meta, "dim": _STUB_DIM + 1}),
    )
    with pytest.raises(ValueError) as excinfo:
        load_sidecar(mismatched)
    assert "dim" in str(excinfo.value), (
        "the ValueError must name the dim mismatch; got "
        f"{str(excinfo.value)!r}"
    )

    matching = tmp_path / "matching.npz"
    np.savez(
        matching,
        text_ids=np.array(IDS, dtype=str),
        text_vecs=_ORTHO,
        text_meta=json.dumps(meta),
    )
    loaded = load_sidecar(matching)
    assert loaded is not None
    assert loaded["text_vecs"].shape == (len(IDS), _STUB_DIM)


def _write_sidecar_model(tmp_path: Path, model: str) -> Path:
    """A sidecar that differs from ``_write_sidecar`` ONLY in ``text_meta.model``.

    Identical ids and orthonormal rows (so RT7's dim check passes), written into
    its own subdirectory so ``_write_sidecar``'s ``embeddings.npz`` is untouched.
    """
    d = tmp_path / model
    d.mkdir()
    meta = {
        "model": model,
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    path = d / "embeddings.npz"
    np.savez(
        path,
        text_ids=np.array(IDS, dtype=str),
        text_vecs=_ORTHO,
        text_meta=json.dumps(meta),
    )
    return path


def test_search_vectors_query_embed_lru_keyed_model_and_text(tmp_path):
    """RT6 — the query-embed LRU is keyed ``(model, text)`` and is deduped
    THROUGH ``search_vectors``: the same ``(model, text)`` embeds exactly ONCE
    across repeated calls; a changed query text re-calls; the same text under a
    DIFFERENT ``meta["model"]`` re-calls.

    Sole-reason lock: today ``search_vectors`` invokes the injected
    ``query_embed`` unconditionally on every call (search.py:99), so two calls
    with the same text record TWO embed calls and the exact-payload assertion
    fails — the red. The exact-payload sequence pins the KEY: a text-only cache
    passes the two-call-one-call arm but FAILS the different-model arm (the
    model2 "same phrase" would hit the model1 entry); a model-only cache FAILS
    the changed-text arm. Only a cache keyed on ``(meta["model"], query)`` over
    the injected seam reproduces the three payloads below.
    """
    sidecar = _write_sidecar(tmp_path)  # meta.model == "nomic-embed-text"
    calls: list[tuple[str, str]] = []

    def counting_embed(query, *, space, meta):
        del space
        calls.append((query, str(meta["model"])))
        return [1.0, 2.0, 0.5, 0, 0, 0, 0, 0]

    # Same (model, text) twice -> exactly ONE embedding call total.
    search_vectors(sidecar, "same phrase", space="text", query_embed=counting_embed)
    search_vectors(sidecar, "same phrase", space="text", query_embed=counting_embed)
    assert len(calls) == 1, (
        "the same (model, text) must embed exactly ONCE across repeated "
        f"search_vectors calls; got {len(calls)} embed call(s): {calls}"
    )

    # A changed query text breaks the LRU entry -> re-call.
    search_vectors(sidecar, "different phrase", space="text", query_embed=counting_embed)
    assert len(calls) == 2, (
        "a changed query text must re-call the embed; "
        f"got {len(calls)} embed call(s): {calls}"
    )

    # The same text under a different recorded model must NOT hit the model1 entry.
    other_sidecar = _write_sidecar_model(tmp_path, "text-embedding-3-small")
    search_vectors(other_sidecar, "same phrase", space="text", query_embed=counting_embed)
    assert len(calls) == 3, (
        "the same text under a different meta.model must re-call the embed "
        f"(the LRU key is (model, text)); got {len(calls)} embed call(s): {calls}"
    )

    # One embed per distinct (model, text) key — the RT6 discriminator.
    assert calls == [
        ("same phrase", "nomic-embed-text"),
        ("different phrase", "nomic-embed-text"),
        ("same phrase", "text-embedding-3-small"),
    ], calls


CODE_IDS = ("c-a-01", "c-b-02")
CODE_LABELS = ("CA Code", "CB Code")
CODE_FILES = ("src/ca.py", "src/cb.py")


def _write_dual_sidecar(tmp_path: Path) -> Path:
    """embeddings.npz carrying BOTH a text group (the standard fixture rows) and
    a code group, so a search has two subspaces to rank.

    The code rows live in the (e0, e2) plane and both have unit norm, so under
    the canonical stub embed (q[0]=1.0, q[2]=0.5, score == dot(q, row)) they
    score c-a-01 = 0.6 - 0.4 = 0.2 and c-b-02 = 0.8 + 0.3 = 1.1. The code rows
    are stored ASCENDING (0.2 then 1.1) so a per-space-scored-descending order
    must come from a real sort. Every text score (2.0 / 1.0 / 0.5) stays
    distinct from the code scores, giving the merged series
    2.0 -> 1.1 -> 1.0 -> 0.5 -> 0.2 that interleaves the two spaces.
    """
    meta = {
        "model": "nomic-embed-text",
        "backend": "ollama",
        "dim": _STUB_DIM,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    code_vecs = np.array(
        [
            [0.6, 0.0, -0.8, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-a-01 -> 0.2
            [0.8, 0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0],  # c-b-02 -> 1.1
        ],
        dtype=np.float32,
    )
    path = tmp_path / "embeddings.npz"
    np.savez(
        path,
        text_ids=np.array(IDS, dtype=str),
        text_vecs=_ORTHO,
        text_meta=json.dumps(meta),
        code_ids=np.array(CODE_IDS, dtype=str),
        code_vecs=code_vecs,
        code_meta=json.dumps({**meta, "model": "qwen2.5-coder"}),
    )
    return path


def _dual_graph_file(tmp_path: Path) -> Path:
    """graph.json carrying the 3 text nodes PLUS the 2 code nodes, so the render
    can join labels and file_types for the code rows exactly as for text rows."""
    g = {
        "directed": True,
        "nodes": [
            {"id": IDS[0], "label": LABELS[0], "source_file": SOURCE_FILES[0],
             "file_type": FILE_TYPES[0], "community": 0},
            {"id": IDS[1], "label": LABELS[1], "source_file": SOURCE_FILES[1],
             "file_type": FILE_TYPES[1], "community": 0},
            {"id": IDS[2], "label": LABELS[2], "source_file": SOURCE_FILES[2],
             "file_type": FILE_TYPES[2], "community": 0},
            {"id": CODE_IDS[0], "label": CODE_LABELS[0], "source_file": CODE_FILES[0],
             "file_type": "code", "community": 0},
            {"id": CODE_IDS[1], "label": CODE_LABELS[1], "source_file": CODE_FILES[1],
             "file_type": "code", "community": 0},
        ],
        "edges": [],
    }
    path = tmp_path / "graph.json"
    path.write_text(json.dumps(g), encoding="utf-8")
    return path


def test_search_vectors_merges_code_and_text_spaces(tmp_path):
    """Semantic search ranks the UNION of the code and text subspaces, not just
    the text group. Every result row carries its own `space` label; rows are
    score-descending within each space, and both spaces' rows interleave in ONE
    global score-descending list cut whole-batch by `top_k`, with `min_score`
    applied across both groups before the cut. The serve render must then show
    `[code]` AND `[text]` lines in the same output.

    Fixture scores under the canonical stub embed: n-b-02 2.0, c-b-02 1.1,
    n-a-01 1.0, n-c-03 0.5, c-a-01 0.2. The global top-3 cut is 2.0 / 1.1 / 1.0
    (one code row interleaving two text rows), and a min_score of 0.6 keeps
    2.0 / 1.1 / 1.0 — the 0.5 text row and the 0.2 code row both drop, so both
    spaces must still be represented after the whole-batch filter. Today only
    the text group is scored, so every "both spaces present" arm is red.
    """
    sidecar = _write_dual_sidecar(tmp_path)

    rows = search_vectors(sidecar, "query", space="text")
    assert {r["space"] for r in rows} == {"text", "code"}, (
        "the merged result must carry rows from BOTH subspaces; the code group "
        f"is never scored today — got {[(r['id'], r['space']) for r in rows]!r}"
    )
    assert {r["id"] for r in rows if r["space"] == "text"} == set(IDS)
    assert {r["id"] for r in rows if r["space"] == "code"} == set(CODE_IDS)
    scores = [r["score"] for r in rows]
    assert scores == sorted(scores, reverse=True), (
        "the merged list must be ONE score-descending series (the two spaces "
        f"interleave); got {[(r['space'], r['score']) for r in rows]!r}"
    )
    # Per-space descending, pinned so a per-space stored-order walk cannot pass.
    for group in ("text", "code"):
        grouped = [r["score"] for r in rows if r["space"] == group]
        assert grouped == sorted(grouped, reverse=True), (
            f"rows within the {group} space must be score-descending, got {grouped!r}"
        )

    # Whole-batch min_score: the 0.5 text row AND the 0.2 code row drop, while
    # both spaces still keep a survivor above the threshold.
    clipped = search_vectors(sidecar, "query", space="text", min_score=0.6)
    assert {r["space"] for r in clipped} == {"text", "code"}, (
        "min_score must apply across BOTH groups after the merge — c-b-02 (1.1) "
        "and n-b-02/n-a-01 (2.0/1.0) survive while c-a-01 (0.2) and n-c-03 (0.5) "
        f"drop; got {[(r['id'], r['space']) for r in clipped]!r}"
    )
    assert all(r["score"] >= 0.6 for r in clipped)
    assert {r["id"] for r in clipped} == {"n-b-02", "c-b-02", "n-a-01"}

    # Whole-batch top_k: the global top-3 cut (2.0, 1.1, 1.0) interleaves the
    # spaces — a per-space cut would keep a different set / length.
    cut = search_vectors(sidecar, "query", space="text", top_k=3)
    assert len(cut) == 3, (
        "top_k must cut the MERGED list whole-batch, not each space separately; "
        f"got {len(cut)} rows: {[(r['id'], r['score']) for r in cut]!r}"
    )
    assert {r["space"] for r in cut} == {"text", "code"}, (
        "the global top three (2.0, 1.1, 1.0) include the c-b-02 code row; got "
        f"{[(r['id'], r['space']) for r in cut]!r}"
    )
    cut_scores = [r["score"] for r in cut]
    assert cut_scores == sorted(cut_scores, reverse=True)

    # Render surface: BOTH a [code] and a [text] line in the same output. The
    # tool's default min_score (0.3) drops only c-a-01's 0.2, so four rows
    # render: n-b-02 2.0, c-b-02 1.1, n-a-01 1.0, n-c-03 0.5.
    _write_dual_sidecar(tmp_path)
    server = serve_mod._build_server(str(_dual_graph_file(tmp_path)))
    result_text = _invoke_semantic_search(server, query="query")
    assert "[code]" in result_text and "[text]" in result_text, (
        "a dual-group search must render [code] AND [text] lines together; the "
        f"code group is never scored today — got {result_text!r}"
    )
    lines = result_text.splitlines()
    assert len(lines) == 4, (
        f"expected the 4 rows above default min_score 0.3, got {lines!r}"
    )
    pinned_render = re.compile(r"(\d+\.\d{3})  \[(code|text)\]  (\S+)  (.+)  \((\w+)\)")
    rendered_scores: list[float] = []
    rendered_ids: list[str] = []
    for line in lines:
        m = pinned_render.fullmatch(line)
        assert m, (
            f"{line!r} does not match the per-row `[{space}]` render surface "
            "(every line carries its own space literal)"
        )
        rendered_scores.append(float(m.group(1)))
    assert rendered_scores == sorted(rendered_scores, reverse=True), (
        "rendered rows must stay score-descending down the whole merged list"
    )
    assert "CB Code" in result_text, (
        f"the surviving code row (c-b-02, score 1.1) must render its graph "
        f"label; got {result_text!r}"
    )


def test_search_vectors_embeds_dual_spaces_concurrently(tmp_path):
    """A dual-space search embeds the query in the code space and the text
    space CONCURRENTLY, not one-after-the-other. The two spaces carry distinct
    models (text -> nomic-embed-text, code -> qwen2.5-coder in this fixture),
    so both embeds are genuinely required and cannot be served from a single
    cached entry.

    An injected query_embed seam sleeps briefly and records each call's
    [start, end) interval under a lock. The assertion is interleaving-based,
    never wall-clock-based: if both embeds overlap in time, each call's start
    is strictly earlier than the other call's end (max(start) < min(end)), and
    the "seen-while-active" flag is set; under the C5 serial loop the second
    embed only starts after the first returns, so the intervals touch at most
    and the flag stays clear. The flag + the total elapsed time (well under
    the two-sleep sum that a serial run needs, but allowing generous slack)
    both discriminate; either arm alone proves the point, and neither can
    flake on scheduler jitter because the assertions reason about recorded
    overlap, not exact sleeps.
    """
    sidecar = _write_dual_sidecar(tmp_path)
    gate = 0.2
    lock = threading.Lock()
    seen_while_active = False
    intervals: dict[str, list[float]] = {}

    def slow_embed(query, *, space, meta):
        """Sleep `gate` seconds, recording this call's [start, end) window and
        whether any other space's embed was already in-flight at start time.
        The in-flight marker (end == None) is written BEFORE the sleep so two
        fully-overlapping embeds observe each other."""
        nonlocal seen_while_active
        del query, meta
        started = time.monotonic()
        with lock:
            intervals[space] = (started, None)
            for other, (os, oe) in intervals.items():
                if other != space and os <= started and (oe is None or started < oe):
                    seen_while_active = True
        time.sleep(gate)
        ended = time.monotonic()
        with lock:
            intervals[space] = (started, ended)
        return [1.0, 2.0, 0.5, 0, 0, 0, 0, 0]

    start = time.monotonic()
    search_vectors(
        sidecar,
        "dual query",
        space="text",
        query_embed=slow_embed,
    )
    elapsed = time.monotonic() - start

    assert set(intervals) == {"text", "code"}, (
        "both the text AND code spaces must embed the query; the code group is "
        f"not being embedded — got {sorted(intervals)!r}"
    )
    text_start, text_end = intervals["text"]
    code_start, code_end = intervals["code"]
    assert max(text_start, code_start) < min(text_end, code_end), (
        "the text and code embeds must OVERLAP in time; a serial one-then-the-"
        "other loop computes them back-to-back and their intervals only touch"
    )
    assert seen_while_active, (
        "while one embed was active the other must have been observed in-flight; "
        "serial execution never has two embeds resident at once"
    )
    assert elapsed < 2 * gate, (
        "concurrent embeds finish in ~one sleep, not ~two (serial); "
        f"elapsed {elapsed:.3f}s over a 2x{gate}s serial sum"
    )


def test_real_query_embed_resolves_backend_model_and_l2_normalizes(monkeypatch):
    """A real query embed resolves backend/model from meta, falls back to space
    defaults, and L2-normalizes the returned vector.

    Arm A: explicit backend and model in meta must reach the embeddings seam.
    Arm B: missing backend/model with space="text" falls back to ollama/nomic-embed-text.
    Arm C: space="code" falls back to ollama/nomic-embed-code.
    """
    import numpy as np

    mock_calls: list[tuple[str, str, list[str]]] = []

    def fake_call_embeddings(backend: str, model: str, inputs: list[str]) -> list[list[float]]:
        mock_calls.append((backend, model, inputs))
        return [[3.0, 4.0, 0.0]]

    monkeypatch.setattr(search_mod, "_call_embeddings", fake_call_embeddings)

    # Arm A — explicit meta
    vec = search_mod._real_query_embed("test", space="text", meta={"backend": "openai", "model": "text-embedding-3-small"})
    assert mock_calls[-1] == ("openai", "text-embedding-3-small", ["test"])
    assert isinstance(vec, list)
    assert all(isinstance(v, float) for v in vec)
    norm = np.linalg.norm(vec)
    assert abs(norm - 1.0) < 1e-6, f"expected L2 norm 1.0, got {norm}"
    assert len(vec) == 3
    assert all(abs(v - e) < 1e-5 for v, e in zip(vec, [0.6, 0.8, 0.0])), f"got {vec!r}"

    # Arm B — text fallback
    search_mod._real_query_embed("hello", space="text", meta={})
    assert mock_calls[-1] == ("ollama", "nomic-embed-text", ["hello"])

    # Arm C — code fallback
    search_mod._real_query_embed("def foo():", space="code", meta={})
    assert mock_calls[-1] == ("ollama", "nomic-embed-code", ["def foo():"])


def test_semantic_search_handler_injects_real_query_embed(tmp_path, monkeypatch):
    """The _tool_semantic_search handler must pass `query_embed=_real_query_embed`
    to `search_vectors` so the query is embedded through the real backend seam,
    not the 8-dimensional test stub. A 3-dimensional sidecar crashes under the
    stub (matmul size mismatch) but succeeds when the real helper is injected
    and monkeypatched to return a compatible 3-dimensional vector.
    """
    import json
    import numpy as np
    from collections import OrderedDict

    import graphify.search as search_mod
    from graphify import serve as serve_mod

    # Clear the global query-embed cache so the handler must call the embedder.
    monkeypatch.setattr(search_mod, "_QUERY_EMBED_CACHE", OrderedDict())

    ids = ("n-a-01", "n-b-02", "n-c-03")
    dim = 3
    meta = {
        "model": "test-model-3d",
        "backend": "ollama",
        "dim": dim,
        "graphify_version": "test",
        "created_at": "2026-08-18T00:00:00+00:00",
    }
    vecs = np.eye(dim, dtype=np.float32)
    np.savez(
        tmp_path / "embeddings.npz",
        text_ids=np.array(ids, dtype=str),
        text_vecs=vecs,
        text_meta=json.dumps(meta),
    )

    graph = {
        "directed": True,
        "nodes": [
            {"id": ids[0], "label": "Alpha", "community": 0},
            {"id": ids[1], "label": "Beta", "community": 0},
            {"id": ids[2], "label": "Gamma", "community": 0},
        ],
        "edges": [],
    }
    (tmp_path / "graph.json").write_text(json.dumps(graph), encoding="utf-8")

    spy_calls = []

    def fake_real_query_embed(query, *, space, meta):
        spy_calls.append((query, space, meta))
        # Return a 3-dim vector that scores highest on the second row.
        return [0.0, 1.0, 0.0]

    monkeypatch.setattr(search_mod, "_real_query_embed", fake_real_query_embed)

    server = serve_mod._build_server(str(tmp_path / "graph.json"))
    result_text = _invoke_semantic_search(server, query="query")

    assert "Error executing" not in result_text, (
        f"the 8-dim stub against a 3-dim sidecar crashes and produces an error alias; "
        f"got: {result_text!r}"
    )
    assert "n-b-02" in result_text, (
        f"the injected real helper must produce a ranked result; "
        f"got: {result_text!r}"
    )
    assert len(spy_calls) >= 1, (
        "the monkeypatched real helper must be invoked by the handler"
    )
