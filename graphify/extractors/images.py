"""Pure helper for turning raw container image references into C1 node dicts."""
from __future__ import annotations


def image_ref_node(ref: str) -> dict | None:
    """Return a node dict for a concrete registry image reference, or None."""
    if not ref or "${{" in ref or ref.startswith("ko://") or "/" not in ref:
        return None
    # Reject the unambiguous non-image forms I5 forbids. A concrete registry
    # path has no variable expansion, no URL scheme, no shell metacharacters,
    # and does not start from the filesystem root or a bare dot.
    if (
        "$" in ref
        or "://" in ref
        or ref.startswith(("./", "../", "/"))
        or any(ch in ref for ch in (" ", "=", "*", ";", "&", "|", ">", "<", "~"))
    ):
        return None

    last_slash = ref.rfind("/")
    name = ref[last_slash + 1 :]
    tag = None
    digest = None
    if "@" in name:
        name, _, digest = name.partition("@")
    if ":" in name:
        name, _, tag = name.partition(":")

    full = ref[: last_slash + 1] + name
    return {
        "id": f"image://{full}",
        "label": full,
        "file_type": "image",
        "source_file": None,
        "attributes": {
            "registry": full.split("/", 1)[0],
            "tags": [t for t in (tag, digest) if t is not None],
        },
    }
