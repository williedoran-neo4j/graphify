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


def _graph_file(tmp_path: Path) -> Path:
    g = {
        "directed": True,
        "nodes": [
            {"id": IDS[0], "label": LABELS[0], "source_file": SOURCE_FILES[0],
             "file_type": FILE_TYPES[0], "community": 0},
            {"id": IDS[1], "label": LABELS[1], "source_file": SOURCE_FILES[1],
             "file_type": FILE_TYPES[1], "community": 0},
            {"id": IDS[2], "label": LABELS[2], "source_file": SOURCE_FILES[2],
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
