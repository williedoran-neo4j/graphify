"""Tests for the image-reference node helper (graphify/extractors/images.py)."""
from __future__ import annotations

from graphify.extractors.images import image_ref_node


def test_image_ref_node_concrete_registry_references_and_rejects_placeholders():
    """image_ref_node returns a C1 node dict for concrete registry references and
    None for bare names, placeholders, ko:// URIs, and empty strings."""

    # 1. Full GCR-style path with tag.
    assert image_ref_node(
        "europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility:806406f5-unclean"
    ) == {
        "id": "image://europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility",
        "label": "europe-west1-docker.pkg.dev/aura-docker-images/aura/aura-operations-utility",
        "file_type": "image",
        "source_file": None,
        "attributes": {
            "registry": "europe-west1-docker.pkg.dev",
            "tags": ["806406f5-unclean"],
        },
    }

    # 2. Quay-style path with @sha256 digest (full digest string kept as tag).
    assert image_ref_node("quay.io/keycloak/keycloak@sha256:deadbeef") == {
        "id": "image://quay.io/keycloak/keycloak",
        "label": "quay.io/keycloak/keycloak",
        "file_type": "image",
        "source_file": None,
        "attributes": {
            "registry": "quay.io",
            "tags": ["sha256:deadbeef"],
        },
    }

    # 3. Port colon must survive (tag is after the LAST '/', not the first ':').
    assert image_ref_node("host:5000/ns/img:tag") == {
        "id": "image://host:5000/ns/img",
        "label": "host:5000/ns/img",
        "file_type": "image",
        "source_file": None,
        "attributes": {
            "registry": "host:5000",
            "tags": ["tag"],
        },
    }

    # 4. Bare name with no registry separator → None (would synthesise a fake node).
    assert image_ref_node("nginx") is None

    # 5. Build-time placeholder token → None.
    assert image_ref_node("_IMAGE") is None

    # 6. ko:// URI → None.
    assert image_ref_node("ko://node-scaler") is None

    # 7. GitHub Actions variable expression → None.
    assert image_ref_node("${{ vars.DEBIAN_DEV_IMAGE }}") is None

    # 8. Empty string → None.
    assert image_ref_node("") is None

    # 9. Make variable → None (I5: placeholders/vars are never images).
    assert image_ref_node("$(REGISTRY)/app:latest") is None

    # 10. Shell variable → None.
    assert image_ref_node("${REGISTRY}/app") is None

    # 11. Bare shell variable → None.
    assert image_ref_node("$REGISTRY/app") is None

    # 12. URL → None (not an image spec).
    assert image_ref_node("https://dist.neo4j.org/x/y") is None

    # 13. Relative path → None (no registry authority).
    assert image_ref_node("./cmd/nes") is None
    assert image_ref_node("../deployments/foo") is None

    # 14. Absolute path → None.
    assert image_ref_node("/usr/local/bin/x") is None

    # 15. Makefile recipe fragments with shell metacharacters → None.
    assert image_ref_node("out/bin/ensure --flag=value") is None
    assert image_ref_node("app.kubernetes.io/name=neo4j-db") is None
    assert image_ref_node("bin/build --tag *") is None


def test_image_file_type_passes_validation_and_survives_build():
    """A container-image node with file_type="image" and source_file=None must
    pass schema validation and retain its file_type through graph assembly,
    not be coerced to "concept"."""
    from graphify.validate import validate_extraction
    from graphify.build import build_from_json

    extraction = {
        "nodes": [
            {
                "id": "image://quay.io/keycloak/keycloak",
                "label": "quay.io/keycloak/keycloak",
                "file_type": "image",
                "source_file": None,
                "attributes": {"registry": "quay.io", "tags": []},
            }
        ],
        "edges": [],
    }

    # Validation: must NOT flag "image" as an invalid file_type
    errors = validate_extraction(extraction)
    file_type_errors = [e for e in errors if "file_type" in e or "'image'" in e]
    assert file_type_errors == []

    # Build: file_type must survive as "image", not be coerced to "concept"
    G = build_from_json(extraction)
    assert G.nodes["image://quay.io/keycloak/keycloak"]["file_type"] == "image"
