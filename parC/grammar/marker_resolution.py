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
    UnorderedMarker,
    PrincipalPartMarker,
    resolve_marker,
)
from parC.yaml_utils.yaml_server import (
    get_markers,
    get_yaml_data_safe,
    get_feature_map,
    get_yaml_path,
)
from parC.constants import get_yaml_dir
from parC.yaml_utils.cache import get_file_sha256
import itertools
import functools
import os
from frozendict import frozendict

FeatureComboType = set[tuple[str, str]]


_markers_for_paradigm_cache = {}
_precomputed_paradigm_markers_cache = {}


@functools.lru_cache(maxsize=1024)
def _get_paradigm_references(
    paradigm_name: str, paradigm_mtime: float
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        return ((), ())
    marker_files = []
    for ref in paradigm_data.get("feature_markers", {}).values():
        if isinstance(ref, str) and ref.startswith("$"):
            marker_files.append(ref)
    contingent_files = list(paradigm_data.get("contingent_markers", []))
    return tuple(sorted(marker_files)), tuple(sorted(contingent_files))


@functools.lru_cache(maxsize=1024)
def _get_paradigm_cache_key_cached(
    yaml_dir: str,
    paradigm_name: str,
    paradigm_hash: str,
    marker_files: tuple[str, ...],
    marker_mtimes: tuple[float, ...],
    contingent_files: tuple[str, ...],
    contingent_mtimes: tuple[float, ...],
) -> tuple:
    marker_hashes = tuple(
        get_file_sha256(get_yaml_path("FeatureMarkers", f)) for f in marker_files
    )
    contingent_hashes = tuple(
        get_file_sha256(get_yaml_path("ContingentFeatureMarkers", f))
        for f in contingent_files
    )
    return (yaml_dir, paradigm_name, paradigm_hash, marker_hashes, contingent_hashes)


_paradigm_cache_key_cache = {}


def get_paradigm_cache_key(paradigm_name: str) -> tuple:
    """Computes a unique cache key based on the current yaml_dir and referenced file hashes."""
    import time

    if "PYTEST_CURRENT_TEST" not in os.environ:
        now = time.monotonic()
        if paradigm_name in _paradigm_cache_key_cache:
            cached_time, cached_key = _paradigm_cache_key_cache[paradigm_name]
            if now - cached_time < 1.0:
                return cached_key

    yaml_dir = get_yaml_dir()

    # 1. Get paradigm yaml hash
    paradigm_path = get_yaml_path("Paradigm", paradigm_name)
    if not os.path.exists(paradigm_path):
        return (yaml_dir, paradigm_name, "", (), ())

    paradigm_mtime = os.path.getmtime(paradigm_path)
    paradigm_hash = get_file_sha256(paradigm_path)

    # 2. Get hashes of referenced marker files
    marker_files, contingent_files = _get_paradigm_references(
        paradigm_name, paradigm_mtime
    )

    marker_paths = [get_yaml_path("FeatureMarkers", f) for f in marker_files]
    marker_mtimes = tuple(
        os.path.getmtime(path) if os.path.exists(path) else 0.0 for path in marker_paths
    )

    contingent_paths = [
        get_yaml_path("ContingentFeatureMarkers", f) for f in contingent_files
    ]
    contingent_mtimes = tuple(
        os.path.getmtime(path) if os.path.exists(path) else 0.0
        for path in contingent_paths
    )

    res = _get_paradigm_cache_key_cached(
        yaml_dir,
        paradigm_name,
        paradigm_hash,
        marker_files,
        marker_mtimes,
        contingent_files,
        contingent_mtimes,
    )
    if "PYTEST_CURRENT_TEST" not in os.environ:
        _paradigm_cache_key_cache[paradigm_name] = (now, res)
    return res


def get_sorted_markers_for_paradigm(paradigm_name: str) -> dict:
    """
    Precomputes, resolves, and caches all markers in the paradigm sorted by their stage order.
    """
    key = get_paradigm_cache_key(paradigm_name)
    if key in _precomputed_paradigm_markers_cache:
        return _precomputed_paradigm_markers_cache[key]

    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm {paradigm_name} not found")

    marker_files = []
    for ref in paradigm_data.get("feature_markers", {}).values():
        if isinstance(ref, str) and ref.startswith("$"):
            marker_files.append(ref)
    contingent_files = list(paradigm_data.get("contingent_markers", []))

    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
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
        data = get_yaml_data_safe("ContingentFeatureMarkers", contingent_file)
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
        data = get_yaml_data_safe("FeatureMarkers", marker_file)
        marker_feature = data["feature"]
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

    res = {
        "all_markers_sorted": all_markers,
        "contingent_by_file": contingent_by_file,
        "contingent_features": contingent_features,
        "regular_by_feature": regular_by_feature,
        "global_markers": global_markers,
        "lexical_feature_names": lexical_feature_names,
    }
    _precomputed_paradigm_markers_cache[key] = res
    return res


def get_markers_for_paradigm(
    feature_values: FeatureComboType | dict[str, str],
    paradigm_name: str,
    include_features: bool = False,
    root: str | None = None,
    lexical_features: FeatureComboType | dict[str, str] | None = None,
    paradigm_data: dict = None,
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

    paradigm_hash_key = get_paradigm_cache_key(paradigm_name)
    cache_key = (
        fv_key,
        paradigm_name,
        include_features,
        root,
        lex_key,
        paradigm_hash_key,
    )
    if cache_key in _markers_for_paradigm_cache:
        return list(_markers_for_paradigm_cache[cache_key])

    if isinstance(feature_values, (dict, frozendict)):
        feature_values: FeatureComboType = set(feature_values.items())
    # avoid side effects
    feature_values = feature_values.copy()

    precomputed = get_sorted_markers_for_paradigm(paradigm_name)
    lexical_feature_names = precomputed["lexical_feature_names"]

    if paradigm_data is None:
        paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)

    # ignore fixed features
    fixed_features = get_fixed_features_for_paradigm(
        name=paradigm_name, kind="Paradigm", paradigm_data=paradigm_data
    )
    feature_values -= fixed_features

    # validate feature values are valid for this paradigm (each feature name is defined and its value exists in the feature map)
    feature_map = get_feature_map()
    for feat_name, feat_val in feature_values:
        if feat_name not in feature_map:
            raise ValueError(
                f"Feature '{feat_name}' in requested values is not defined in the feature definitions."
            )
        if feat_val not in feature_map[feat_name]:
            raise ValueError(
                f"Feature value '{feat_val}' for feature '{feat_name}' is not defined in the feature definitions."
            )

    part_of_speech = paradigm_data["part_of_speech"]
    if root:
        lex_set = set(get_features_for_root(part_of_speech, root))
        feature_values |= lex_set
    if lexical_features:
        if isinstance(lexical_features, (dict, frozendict)):
            lex_set = set(lexical_features.items())
        else:
            lex_set = set(lexical_features)
        feature_values |= lex_set

    # Selection Logic:
    unexponed_features = {
        feature for feature, _ in feature_values if feature not in lexical_feature_names
    }
    selected_markers = []

    # 1. Match contingent feature markers
    for contingent_file, leaves in precomputed["contingent_by_file"].items():
        for reqs, markers_list in leaves:
            if reqs.issubset(feature_values):
                selected_markers.extend(markers_list)
                unexponed_features -= precomputed["contingent_features"][
                    contingent_file
                ]
                break

    # 2. Match remaining features with regular feature markers
    for feature in list(unexponed_features):
        val = next((v for f, v in feature_values if f == feature), None)
        if val is not None:
            markers_list = (
                precomputed["regular_by_feature"].get(feature, {}).get(val, [])
            )
            if markers_list:
                selected_markers.extend(markers_list)
                unexponed_features.remove(feature)

    if unexponed_features:
        raise ValueError("Provided marker sets do not support requested feature set")

    # 3. Add global markers (unless overridden by principal parts)
    has_principal_part = any(
        marker.kind == "principal_part" for marker, _ in selected_markers
    )
    for marker, f_set in precomputed["global_markers"]:
        if not (has_principal_part and marker.kind == "principal_part"):
            selected_markers.append((marker, f_set))

    # To preserve the stage order precomputed in all_markers_sorted,
    # we filter all_markers_sorted to only keep the selected ones.
    selected_set = {
        (m, frozenset(fs) if isinstance(fs, set) else fs) for m, fs in selected_markers
    }

    markers = []
    for m, fs in precomputed["all_markers_sorted"]:
        fs_key = frozenset(fs) if isinstance(fs, set) else fs
        if (m, fs_key) in selected_set:
            markers.append((m, fs))

    if not include_features:
        markers = [marker for marker, _ in markers]

    _markers_for_paradigm_cache[cache_key] = list(markers)
    return markers


def get_fixed_features_for_paradigm(
    name: str, kind: str = "Paradigm", paradigm_data: dict = None
) -> FeatureComboType:
    if paradigm_data is None:
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


@functools.lru_cache(maxsize=200)
def get_contingent_markers_for_paradigm(paradigm_name):
    paradigm_data = get_yaml_data_safe(kind="Paradigm", yaml_basename=paradigm_name)
    return list(paradigm_data.get("contingent_markers", []))


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
