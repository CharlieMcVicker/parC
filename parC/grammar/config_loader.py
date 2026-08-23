"""Upfront Config Loader & Typed Config Models.

Implements Ports & Adapters Architecture by reading and resolving all YAML config
models upfront into pure, typed dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from parC.yaml_utils.models import (
    Feature,
    Inventory,
    Marker,
    Pattern,
    Rule,
)
from parC.yaml_utils.yaml_server import (
    get_feature_array,
    get_feature_map,
    get_inventory_items,
    get_patterns,
    get_rules,
    get_yaml_data_safe,
    get_yaml_kind,
)
from parC.lexicon import get_roots, get_principal_part_for_all_roots
from parC.yaml_utils.models import PrincipalPartMarker, resolve_marker


@dataclass(frozen=True)
class ParadigmConfig:
    paradigm_name: str
    paradigm_data: dict
    feature_map: dict[str, tuple[str, ...]]
    sorted_markers: tuple[tuple[Marker, dict | set | frozenset | str], ...]
    contingent_by_file: dict
    contingent_features: dict
    regular_by_feature: dict
    global_markers: tuple[tuple[Marker, str], ...]
    lexical_feature_names: set[str]
    part_of_speech_data: dict


@dataclass(frozen=True)
class GrammarConfig:
    inventory: Inventory
    features: tuple[Feature, ...]
    feature_map: dict[str, tuple[str, ...]]
    patterns: dict[str, Pattern]
    rules: dict[str, Rule]
    paradigms: dict[str, ParadigmConfig]


def load_paradigm_config(
    paradigm_name: str,
    feature_map: dict[str, tuple[str, ...]] | None = None,
) -> ParadigmConfig:
    """Loads and resolves a ParadigmConfig upfront for a given paradigm_name."""
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    if feature_map is None:
        feature_map = get_feature_map()

    marker_files = []
    for ref in paradigm_data.get("feature_markers", {}).values():
        if isinstance(ref, str) and ref.startswith("$"):
            marker_files.append(ref)
    contingent_files = list(paradigm_data.get("contingent_markers", []))

    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = (
        get_yaml_data_safe(yaml_basename=part_of_speech, kind="PartOfSpeech") or {}
    )
    lexical_feature_names = set(part_of_speech_data.get("lexical_features", []))

    global_stage = paradigm_data.get("global_stage")
    stage_order = list(paradigm_data.get("stage_order", []))
    if "principal_part" not in stage_order:
        stage_order.insert(0, "principal_part")

    def process_marker(marker: Marker) -> Marker:
        marker = resolve_marker(marker)
        if (
            hasattr(marker, "stage")
            and marker.stage is None
            and global_stage is not None
        ):
            marker = marker._replace(stage=global_stage)
        if marker.kind == "principal_part":
            roots = get_roots(part_of_speech)
            pps = get_principal_part_for_all_roots(part_of_speech, marker.value)
            marker = PrincipalPartMarker(
                kind="string_map",
                display_value=marker.value,
                value=tuple(zip(roots, pps)),
            )
        return marker

    contingent_by_file = {}
    contingent_features = {}
    all_markers = []

    # 1. Process contingent markers
    for contingent_file in contingent_files:
        data = get_yaml_data_safe("ContingentFeatureMarkers", contingent_file) or {}
        features_list = data.get("features", [])
        inflectional_contingent_names = {
            f for f in features_list if f not in lexical_feature_names
        }
        contingent_features[contingent_file] = inflectional_contingent_names

        leaves = []

        def traverse(curr, path_features: dict):
            if isinstance(curr, list):
                resolved_list = []
                contingent_feature_values = {(f, v) for f, v in path_features.items()}
                for m in curr:
                    pm = process_marker(m)
                    resolved_list.append((pm, contingent_feature_values))
                    all_markers.append((pm, contingent_feature_values))
                leaves.append((frozenset(path_features.items()), resolved_list))
            elif isinstance(curr, dict):
                level = len(path_features)
                if level < len(features_list):
                    feature_name = features_list[level]
                    for val, sub in curr.items():
                        new_path = path_features.copy()
                        new_path[feature_name] = val
                        traverse(sub, new_path)

        traverse(data.get("markers", {}), {})
        contingent_by_file[contingent_file] = leaves

    # 2. Process regular markers
    regular_by_feature = {}
    for marker_file in marker_files:
        data = get_yaml_data_safe("FeatureMarkers", marker_file) or {}
        marker_feature = data.get("feature", "")
        if marker_feature not in regular_by_feature:
            regular_by_feature[marker_feature] = {}
        for val, marker_list in data.get("markers", {}).items():
            resolved_list = []
            marker_feature_set = {(marker_feature, val)}
            for m in marker_list:
                pm = process_marker(m)
                resolved_list.append((pm, marker_feature_set))
                all_markers.append((pm, marker_feature_set))
            regular_by_feature[marker_feature][val] = resolved_list

    # 3. Process global markers
    global_markers = []
    if "global_markers" in paradigm_data:
        for m in paradigm_data["global_markers"]:
            pm = process_marker(m)
            global_markers.append((pm, "global"))
            all_markers.append((pm, "global"))

    # Sort all_markers by stage order
    all_markers.sort(
        key=lambda m: (
            stage_order.index(m[0].stage) if m[0].stage in stage_order else float("inf")
        )
    )

    return ParadigmConfig(
        paradigm_name=paradigm_name,
        paradigm_data=paradigm_data,
        feature_map=feature_map,
        sorted_markers=tuple(all_markers),
        contingent_by_file=contingent_by_file,
        contingent_features=contingent_features,
        regular_by_feature=regular_by_feature,
        global_markers=tuple(global_markers),
        lexical_feature_names=lexical_feature_names,
        part_of_speech_data=part_of_speech_data,
    )


def load_grammar_config() -> GrammarConfig:
    """Loads all YAML config models upfront into a pure GrammarConfig dataclass."""
    inventory = get_inventory_items()
    features = get_feature_array()
    feature_map = get_feature_map()
    patterns = get_patterns()
    rules = get_rules()

    paradigm_kind_res = get_yaml_kind("Paradigm")
    paradigms = {}
    if paradigm_kind_res and "valid" in paradigm_kind_res:
        for file_path, data in paradigm_kind_res["valid"]:
            p_name = os.path.splitext(file_path)[0]
            paradigms[p_name] = load_paradigm_config(p_name, feature_map=feature_map)

    return GrammarConfig(
        inventory=inventory,
        features=features,
        feature_map=feature_map,
        patterns=patterns,
        rules=rules,
        paradigms=paradigms,
    )
