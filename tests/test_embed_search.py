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
