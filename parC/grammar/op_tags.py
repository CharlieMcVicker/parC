"""
Operational tags and utility functions for stage-based composition cascade.
"""

from __future__ import annotations

import re
from loguru import logger
from parC.yaml_utils.models import (
    Marker,
    SingleStringMarker,
    StringTupleMarker,
    UnorderedMarker,
    PrincipalPartMarker,
    resolve_marker,
)

def slugify(s: str) -> str:
    """
    Convert a string into a clean, alphanumeric-and-underscore-only slug.
    """
    s = re.sub(r'[^a-zA-Z0-9_]', '_', s)
    s = re.sub(r'_+', '_', s)
    s = s.strip('_')
    return s

def get_op_tag(marker: Marker) -> str:
    """
    Returns a clean, deterministic operational tag of the form [OP=...] for a given Marker.
    """
    if isinstance(marker, SingleStringMarker):
        val_part = slugify(marker.value)
        if marker.stage:
            return f"[OP={marker.kind}_{marker.stage}_{val_part}]"
        return f"[OP={marker.kind}_{val_part}]"
    elif isinstance(marker, StringTupleMarker):
        val_part = f"{slugify(marker.value[0])}_to_{slugify(marker.value[1])}"
        if marker.stage:
            return f"[OP={marker.kind}_{marker.stage}_{val_part}]"
        return f"[OP={marker.kind}_{val_part}]"
    elif isinstance(marker, PrincipalPartMarker):
        val_part = slugify(marker.display_value)
        stage_str = marker.stage if marker.stage else "principal_part"
        return f"[OP={marker.kind}_{stage_str}_{val_part}]"
    elif isinstance(marker, UnorderedMarker):
        val_part = slugify(marker.value)
        return f"[OP={marker.kind}_{val_part}]"
    else:
        raise TypeError(f"Unknown marker type: {type(marker)}")

def extract_contingent_markers(curr) -> list[dict]:
    """
    Recursively extract all marker dicts from nested contingent markers data.
    """
    if isinstance(curr, list):
        return curr
    elif isinstance(curr, dict):
        markers = []
        for val in curr.values():
            markers.extend(extract_contingent_markers(val))
        return markers
    return []

def get_all_op_tags() -> set[str]:
    """
    Scans all paradigms, retrieves all possible markers, and collects their operational tags.
    """
    from parC.yaml_utils.yaml_server import get_yaml_kind, get_yaml_data_safe

    op_tags = set()
    paradigms_info = get_yaml_kind("Paradigm")
    if not paradigms_info:
        return op_tags

    for file_basename, paradigm_data in paradigms_info.get("valid", []):
        global_stage = paradigm_data.get("global_stage", None)

        def add_marker(m_dict):
            try:
                resolved = resolve_marker(m_dict)
                op_tags.add(get_op_tag(resolved))
                if global_stage and hasattr(resolved, "stage") and resolved.stage is None:
                    staged = resolved._replace(stage=global_stage)
                    op_tags.add(get_op_tag(staged))
                if hasattr(resolved, "kind") and resolved.kind == "principal_part":
                    # principal_part is resolved to a string_map marker at runtime
                    simulated = PrincipalPartMarker(
                        kind="string_map",
                        display_value=resolved.value,
                        value=(),
                        stage="principal_part",
                    )
                    op_tags.add(get_op_tag(simulated))
            except Exception as e:
                logger.warning(f"Failed to resolve marker dict {m_dict}: {e}")

        # 1. Global markers
        if "global_markers" in paradigm_data:
            for m_dict in paradigm_data["global_markers"]:
                add_marker(m_dict)

        # 2. Feature markers
        for feature, ref in paradigm_data.get("feature_markers", {}).items():
            if ref is not None and isinstance(ref, str) and ref.startswith("$"):
                data = get_yaml_data_safe("FeatureMarkers", ref)
                if data and "markers" in data:
                    for val, markers_list in data["markers"].items():
                        if isinstance(markers_list, list):
                            for m_dict in markers_list:
                                add_marker(m_dict)

        # 3. Contingent markers
        for ref in paradigm_data.get("contingent_markers", []):
            if isinstance(ref, str) and ref.startswith("$"):
                data = get_yaml_data_safe("ContingentFeatureMarkers", ref)
                if data and "markers" in data:
                    contingent_list = extract_contingent_markers(data["markers"])
                    for m_dict in contingent_list:
                        add_marker(m_dict)

    return op_tags
