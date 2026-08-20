"""R7 Neo4j/FalkorDB tier — shared prop-filter helper.

The four push sites in ``graphify/exporters/graphdb.py`` each wrote an inline
dict comprehension that kept only scalar values (``str``/``int``/``float``/
``bool``) under non-``_`` keys. Neo4j can store ``LIST<FLOAT>``, and the
embedding-vector props planned for the same push are numpy rows of int/float —
so the filter that feeds the driver must keep an all-numeric list. ``True`` is
a subclass of ``int`` in Python, so the membership test must exclude ``bool``
before it accepts ``int`` (a ``True`` member must be dropped, never coerced to
``1``). Any nested/object element (e.g. a ``dict``) must still drop the whole
list. This file is the R7 test home; the other slices live in
``test_embed_graphdb.py`` alongside this one.
"""
from __future__ import annotations

from graphify.exporters.graphdb import _pushable_props


def test_pushable_props_float_list_survives_but_bool_first_and_dict_list_dropped():
    props = _pushable_props(
        {
            "evals": [0.717, 1.0, 2.5],          # all float — must survive
            "counts": [1, 2, 3],                 # all int — must survive
            "mixed_num": [1, 2.5, 3],            # int+float mix — must survive
            "flags": [True, False, 1],           # bool member — whole list dropped
            "records": [{"role": "doc"}],        # dict element — whole list dropped
            "label": "doc",                      # scalar str — survives
            "rank": 3,                           # scalar int — survives
            "score": 0.5,                        # scalar float — survives
            "active": True,                      # scalar bool — survives
            "_internal": "secret",               # _-prefixed key — dropped
        }
    )

    # All-numeric lists survive with their members intact.
    assert props["evals"] == [0.717, 1.0, 2.5]
    assert props["counts"] == [1, 2, 3]
    assert props["mixed_num"] == [1, 2.5, 3]

    # A bool member is not a number (I9): dropped, never coerced to 1. A
    # list[dict] is nested/object data the driver cannot store: dropped.
    assert "flags" not in props
    assert "records" not in props

    # Scalars survive exactly as today; _-prefixed keys are filtered out.
    assert props["label"] == "doc"
    assert props["rank"] == 3
    assert props["score"] == 0.5
    assert props["active"] is True
    assert "_internal" not in props
