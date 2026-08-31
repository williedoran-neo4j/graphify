def test_build_file_type_passes_validation_and_survives_build():
    """A build node with file_type='build' must pass schema validation and
    retain its file_type through graph assembly, not be coerced to 'concept'."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "makefile://x/y",
                "label": "y",
                "file_type": "build",
                "source_file": "Makefile",
                "attributes": {},
            }
        ],
        "edges": [],
    }

    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'build'" in e]
    assert file_type_errors == []

    G = build_from_json(extraction)
    assert G.nodes["makefile://x/y"]["file_type"] == "build"
