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
