"""
Resolves marker lists for a paradigm+feature-value combination.

Separated from yaml_server to avoid a circular import:
  yaml_server ← lexicon ← yaml_server (get_yaml_data_safe)
"""

from __future__ import annotations

from loguru import logger

from parC.lexicon import (
    get_roots,
    get_principal_part_for_all_roots,
    get_features_for_root,
)
from parC.yaml_utils.models import (
    Marker,
    PrincipalPartMarker,
    resolve_marker,
)
from parC.yaml_utils.yaml_server import get_markers, get_yaml_data_safe, get_feature_map
import itertools
from frozendict import frozendict

FeatureComboType = set[tuple[str, str]]


_markers_for_paradigm_cache = {}


def get_markers_for_paradigm(
    feature_values: FeatureComboType | dict[str, str],
    paradigm_name: str,
    include_features: bool = False,
    root: str | None = None,
    lexical_features: FeatureComboType | dict[str, str] | None = None,
) -> list[Marker] | list[tuple[Marker, FeatureComboType | str]]:
    """
    Get all markers for a requested feature set for a given paradigm.
    Resolves principal_part markers into StringMapMarker using the paradigm's lexicon.
    """
    fv_key = (
        frozenset(feature_values.items())
        if isinstance(feature_values, (dict, frozendict))
        else frozenset(feature_values)
    )
    lex_key = (
        frozenset(lexical_features.items())
        if isinstance(lexical_features, (dict, frozendict))
        else (frozenset(lexical_features) if lexical_features else frozenset())
    )
    cache_key = (fv_key, paradigm_name, include_features, root, lex_key)
    if cache_key in _markers_for_paradigm_cache:
        return list(_markers_for_paradigm_cache[cache_key])

    if isinstance(feature_values, (dict, frozendict)):
        feature_values: FeatureComboType = set(feature_values.items())
    # avoid side effects
    feature_values = feature_values.copy()

    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)

    # ignore fixed features
    fixed_features = get_fixed_features_for_paradigm(
        name=paradigm_name, kind="Paradigm"
    )
    feature_values -= fixed_features

    combos, marker_files, contingent_files = get_feature_combos_for_paradigm(
        name=paradigm_name, kind="Paradigm"
    )
    combos = [combo - fixed_features for combo in combos]

    if not any([feature_values == combo for combo in combos]):
        raise ValueError(
            f"Feature values {feature_values} not valid for paradigm {paradigm_name}"
        )

    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
    )
    lexical_feature_names = set(part_of_speech_data.get("lexical_features", []))

    if root:
        lex_set = set(get_features_for_root(part_of_speech, root))
        feature_values |= lex_set
    if lexical_features:
        if isinstance(lexical_features, (dict, frozendict)):
            lex_set = set(lexical_features.items())
        else:
            lex_set = set(lexical_features)
        feature_values |= lex_set

    markers = get_markers(
        feature_marker_files=marker_files,
        contingent_feature_marker_files=contingent_files,
        feature_values=feature_values,
        lexical_feature_names=lexical_feature_names,
    )

    # any global markers defined in the paradigm should be applied to all feature combinations
    # check if current feature set has a principal part, if so we don't override
    has_principal_part = any(marker.kind == "principal_part" for marker, _ in markers)
    if "global_markers" in paradigm_data:
        markers.extend(
            (resolve_marker(marker), "global")
            for marker in paradigm_data["global_markers"]
            if not (has_principal_part and marker.kind == "principal_part")
        )

    # global stage applies to any unstaged marker, but does not override
    # staged markers
    if "global_stage" in paradigm_data:
        for i, marker_tuple in enumerate(markers):
            marker, feature_set = marker_tuple
            if hasattr(marker, "stage") and marker.stage is None:
                markers[i] = (
                    marker._replace(stage=paradigm_data["global_stage"]),
                    feature_set,
                )

    # principle part markers need access to the lexicon to resolve into a StringMapMarker
    part_of_speech = paradigm_data["part_of_speech"]
    for i, marker_tuple in enumerate(markers):
        marker, feature_set = marker_tuple
        if marker.kind == "principal_part":
            roots = get_roots(part_of_speech)
            pps = get_principal_part_for_all_roots(part_of_speech, marker.value)
            markers[i] = (
                PrincipalPartMarker(
                    kind="string_map",
                    display_value=marker.value,
                    value=tuple(zip(roots, pps)),
                ),
                feature_set,
            )

    # order of application specified by paradigm's stage_order, if present
    stage_order: list[str] | None = paradigm_data.get("stage_order", [])
    # principal parts always come first
    # TODO: need to support suppletion
    # i.e. suppletion should also be first, and it should be mutually exclusive
    # with principal part
    stage_order.insert(0, "principal_part")
    markers.sort(
        key=lambda m: (
            stage_order.index(m[0].stage) if m[0].stage in stage_order else float("inf")
        )
    )

    if not include_features:
        markers = [marker for marker, _ in markers]

    _markers_for_paradigm_cache[cache_key] = list(markers)
    return markers


def get_fixed_features_for_paradigm(
    name: str, kind: str = "Paradigm"
) -> FeatureComboType:
    paradigm_data = get_yaml_data_safe(kind=kind, yaml_basename=name)
    fixed_features = set()
    for feature, value in paradigm_data["feature_markers"].items():
        if isinstance(value, str) and not value.startswith("$"):
            fixed_features.add((feature, value))

    return fixed_features


def get_free_features_for_paradigm(name: str, kind: str = "Paradigm") -> list[str]:
    paradigm_data = get_yaml_data_safe(kind=kind, yaml_basename=name)
    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
    )
    free_features = list(part_of_speech_data.get("features", []))

    for feature, value in paradigm_data.get("feature_markers", {}).items():
        if value is not None and not (isinstance(value, str) and value.startswith("$")):
            if feature in free_features:
                free_features.remove(feature)

    return free_features


def get_feature_combos_for_paradigm(
    name: str,
    feature_map: dict | None = None,
    kind: str = "Paradigm",
) -> tuple[list[FeatureComboType], list[str], list[str]]:
    """
    Return valid feature combos for a paradigm.
    Each combo is a set of (feature, value) pairs covering all free features
    plus any fixed feature values.
    """

    if feature_map is None:
        feature_map = get_feature_map()

    paradigm_data = get_yaml_data_safe(kind=kind, yaml_basename=name)
    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
    )

    fixed: dict[str, str] = {}
    marker_files: list[str] = []
    # assume all inflectional features are free unless explicitly fixed in the paradigm
    free_feature_names: list[str] = part_of_speech_data.get("features", [])

    for feature_name, ref in paradigm_data.get("feature_markers", {}).items():
        if ref is None:
            # feature is only exponed via contingent markers
            # or is unexponed
            continue
        if isinstance(ref, str) and ref.startswith("$"):
            marker_files.append(ref)
        else:
            fixed[feature_name] = ref
            if feature_name in free_feature_names:
                free_feature_names.remove(feature_name)

    contingent_files = list(paradigm_data.get("contingent_markers", []))

    free_value_lists = []
    for fname in free_feature_names:
        if fname not in feature_map:
            logger.warning(f"Feature '{fname}' not in feature map — skipping.")
            continue
        free_value_lists.append([(fname, v) for v in feature_map[fname]])

    if not free_value_lists:
        combos = [set(fixed.items())]
    else:
        combos = [
            set(fixed.items()) | set(combo_tuples)
            for combo_tuples in itertools.product(*free_value_lists)
        ]
    return combos, marker_files, contingent_files


def get_features_for_paradigm(name: str) -> set[str]:
    """
    Get the set of inflectional features for a given paradigm.
    """
    paradigm_data = get_yaml_data_safe("Paradigm", name)
    part_of_speech = paradigm_data["part_of_speech"]
    features = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
    ).get("features", [])
    return set(features)
