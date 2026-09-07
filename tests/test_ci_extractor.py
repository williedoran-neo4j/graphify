"""Tests for the CI extractor whitelist (file_type='ci')."""
from __future__ import annotations


def test_ci_file_type_passes_validation_and_survives_build():
    """A node with file_type='ci' must pass schema validation without a file_type
    error and must retain file_type='ci' through graph assembly, not be coerced
    to 'concept'."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "ci://x/j",
                "label": "j",
                "file_type": "ci",
                "source_file": "f.yml",
                "attributes": {},
            }
        ],
        "edges": [],
    }

    # Validation: must NOT flag "ci" as an invalid file_type
    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'ci'" in e]
    assert file_type_errors == []

    # Build: file_type must survive as "ci", not be coerced to "concept"
    G = build_from_json(extraction)
    assert G.nodes["ci://x/j"]["file_type"] == "ci"
