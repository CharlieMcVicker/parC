"""
Functions for fetching config YAML data.
Intuitively, this model handles serving any YAML data
that requires no "interpretation" of grammar (e.g.
compilation of regex strings or rule definitions to FSAs,
dependency and inheritance relationships, or filtering of
lexemes by lexical features).

Includes functions for loading and validating entire YAML files
and for loading specific objects from YAML files, viz:
- rules (rules/*.yaml)
- patterns (patterns/*.yaml)
- inventory items (inventory/*.yaml)
- markers (feature_markers/*.yaml, contingent_feature_markers/*.yaml)
- inflection stages (feature_markers/*.yaml, contingent_feature_markers/*.yaml)
- features (feature_definitions/*.yaml)
"""

import os

from loguru import logger
import yaml
from frozendict import frozendict

from parC.yaml_utils.schema_validation import (
    validate_yaml,
    CONFIG_KINDS,
    CONFIG_KIND_TO_PARDIR,
)
from parC.constants import get_yaml_dir
from parC.yaml_utils.models import (
    Inventory,
    InventoryItemMapType,
    InventoryItemContents,
    Marker,
    resolve_marker,
    Pattern,
    Rule,
    resolve_rule,
    SimpleRule,
    StringMapRule,
    RuleSequence,
    Feature,
    FeatureValue,
    UnorderedMarker,
    PrincipalPartMarker,
    FeatureValueDef,
)

"""
## File serving functions
"""


import functools
import copy


@functools.lru_cache(maxsize=1024)
def _get_yaml_data_safe_cached(kind: str, yaml_basename: str, file_hash: str) -> dict:
    if kind not in CONFIG_KINDS:
        logger.error(f"Invalid config kind: {kind}")
        return None

    yaml_file_path = get_yaml_path(kind, yaml_basename)

    with open(yaml_file_path, "r", encoding="utf-8") as f:
        yaml_data = yaml.safe_load(f)

    yaml_data = validate_yaml(target_kind=kind, data=yaml_data)
    if yaml_data is None:
        logger.error(f"Failed to validate YAML data from path: {yaml_file_path}")
        return None

    return yaml_data


def get_yaml_data_safe(kind: str, yaml_basename: str) -> dict:
    """
    Load a single YAML file and validate its contents against the expected schema.
    Returns None if the config kind is invalid or the YAML data fails validation,
    else returns the validated YAML data.
    """
    if kind not in CONFIG_KINDS:
        logger.error(f"Invalid config kind: {kind}")
        return None

    yaml_file_path = get_yaml_path(kind, yaml_basename)
    if not os.path.exists(yaml_file_path):
        return None

    from parC.yaml_utils.cache import get_file_sha256

    file_hash = get_file_sha256(yaml_file_path)

    data = _get_yaml_data_safe_cached(kind, yaml_basename, file_hash)
    if data is not None:
        return copy.deepcopy(data)
    return None


def get_yaml_path(kind, yaml_basename):
    yaml_basename = yaml_basename.removeprefix("$")
    if not yaml_basename.endswith(".yaml"):
        yaml_basename += ".yaml"

    yaml_file_path = os.path.join(
        get_yaml_dir(), CONFIG_KIND_TO_PARDIR[kind], kind, yaml_basename
    )

    return yaml_file_path


def get_yaml_kind(kind: str) -> dict[str, list[tuple[str, dict] | str]]:
    """
    Loads all YAML files for a given config kind and returns a dictionary
    with shape:
    {
        "valid": [(file_basename, yaml_data), ...],
        "invalid": [file_basename, ...]
    }
    """
    if kind not in CONFIG_KINDS:
        logger.error(f"Invalid config kind: {kind}")
        return None
    yaml_pardir = os.path.join(get_yaml_dir(), CONFIG_KIND_TO_PARDIR[kind], kind)
    yaml_files = [f for f in os.listdir(yaml_pardir) if f.endswith(".yaml")]
    result = {"valid": [], "invalid": []}
    for yaml_file in yaml_files:
        yaml_data = get_yaml_data_safe(kind, yaml_file)
        if yaml_data is not None:
            result["valid"].append((yaml_file, yaml_data))
        else:
            result["invalid"].append(yaml_file)
    return result


"""
## Inner data fetching functions
"""
"""
### Inventory
"""


def get_inventory_items() -> Inventory:
    """
    Fetch all inventory items from the YAML files.
    Returns a dictionary mapping item names to a tuple whose
    elements are every phone or tag associated with the item.

    Since inventory items are nested, recursively extract all phone
    and tag information for each node.
    """

    inventory_yaml_data = get_yaml_kind("Inventory")["valid"]
    inventory_items: InventoryItemMapType = {}
    all_phones: list[str] = []
    all_tags: list[str] = []

    def extract_phones_and_tags(item):
        phones = []
        tags = []
        if isinstance(item, (dict, frozendict)):
            phones.extend(item.get("phones", []))
            tags.extend(item.get("tags", []))
            for subitem in item.get("children", []):
                sub_phones, sub_tags = extract_phones_and_tags(subitem)
                inventory_items[subitem["ref"]] = InventoryItemContents(
                    phones=tuple(sub_phones), tags=tuple(sub_tags)
                )
                phones.extend(sub_phones)
                tags.extend(sub_tags)
        return phones, tags

    for file_path, yaml_data in inventory_yaml_data:
        for item_data in yaml_data["data"]:
            item_ref = item_data.get("ref")
            item_phones, item_tags = extract_phones_and_tags(item_data)
            # BUG: not adding refs recursively
            if item_ref in inventory_items:
                logger.exception(f"Duplicate item found: {item_ref} in {file_path}")
                continue
            inventory_items[item_ref] = InventoryItemContents(
                phones=tuple(item_phones), tags=tuple(item_tags)
            )
            all_phones.extend(item_phones)
            all_tags.extend(item_tags)

    return Inventory(
        item_map=inventory_items, phones=tuple(all_phones), tags=tuple(all_tags)
    )


"""
### Patterns
"""


def get_patterns() -> dict[str, Pattern]:
    """
    Fetch all patterns from the YAML files.
    Returns a dictionary mapping pattern names to their corresponding pattern objects.
    """
    patterns_yaml_data = get_yaml_kind("Patterns")["valid"]
    patterns: dict[str, Pattern] = {}

    for file_path, yaml_data in patterns_yaml_data:
        for pattern in yaml_data["patterns"]:
            pattern_ref = pattern.get("ref")
            if pattern_ref in patterns:
                logger.exception(
                    f"Duplicate pattern found: {pattern_ref} in {file_path}"
                )
                continue
            patterns[pattern_ref] = Pattern(
                pattern=pattern["pattern"],
                test_includes=tuple(pattern.get("test_includes", [])) or None,
                test_excludes=tuple(pattern.get("test_excludes", [])) or None,
                name=pattern.get("name"),
            )

    return patterns


"""
### Rules
"""


def get_rules() -> dict[str, Rule]:
    """
    Fetch all rules from the YAML files.
    Returns a dictionary mapping rule names to their corresponding rule objects.
    """
    rules_yaml_data = get_yaml_kind("Rules")["valid"]
    rules: dict[str, Rule] = {}

    for file_path, yaml_data in rules_yaml_data:
        for rule in yaml_data["rules"]:
            rule_name = rule.pop("name")
            if rule_name in rules:
                logger.exception(f"Duplicate rule found: {rule_name} in {file_path}")
                continue
            rules[rule_name] = resolve_rule(rule)

    return rules


"""
### Features
"""


def get_feature_map() -> dict[str, tuple[str, ...]]:
    """
    Fetch all features from the YAML files.
    Returns a dictionary mapping feature names to their corresponding feature objects
    (for easy indexing of a particular feature).
    """
    features_yaml_data = get_yaml_kind("FeatureDefinitions")["valid"]
    features: dict[str, tuple[str, ...]] = {}

    for file_path, yaml_data in features_yaml_data:
        for feature_name, feature_data in yaml_data["features"].items():
            if feature_name in features:
                logger.exception(
                    f"Duplicate feature found: {feature_name} in {file_path}"
                )
                continue
            names = []
            for item in feature_data:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    names.append(item["name"])
            features[feature_name] = tuple(names) + ("unmarked",)

    return features


def get_feature_array() -> tuple[Feature]:
    """
    Fetch all features from the YAML files.
    Return a tuple of Feature objects (for easy iteration).

    Unlike `get_feature_map`, don't perform validation of duplicate features.
    TODO: map out Python-side YAML validation so that it isn't hapharzardly done
    across the codebase.
    """
    features_yaml_data = get_yaml_kind("FeatureDefinitions")["valid"]
    features: list[Feature] = []

    for _, yaml_data in features_yaml_data:
        for feature_name, feature_data in yaml_data["features"].items():
            normalized_vals = []
            for item in feature_data:
                if isinstance(item, str):
                    normalized_vals.append(FeatureValueDef(name=item, acceptor=None))
                elif isinstance(item, dict):
                    normalized_vals.append(
                        FeatureValueDef(
                            name=item["name"], acceptor=item.get("acceptor")
                        )
                    )
            normalized_vals.append(FeatureValueDef(name="unmarked", acceptor=None))
            features.append(Feature(name=feature_name, values=tuple(normalized_vals)))

    return tuple(features)


def get_feature_values(feature: str) -> tuple[str]:
    """
    Fetch all values for a given feature.
    """
    feature_map = get_feature_map()
    return feature_map[feature]


"""
# Inflection stages
Get unique inflection stages from all marker files.
"""


def get_inflection_stages() -> set[str]:
    infleciton_stages = set()

    feature_marker_yaml = get_yaml_kind("FeatureMarkers")
    for _, yaml_data in feature_marker_yaml["valid"]:
        for _, marker_data in yaml_data["markers"].items():
            if marker_data and "stage" in marker_data:
                infleciton_stages.add(marker_data["stage"])

    contingent_feature_marker_yaml = get_yaml_kind("ContingentFeatureMarkers")
    for _, yaml_data in contingent_feature_marker_yaml["valid"]:
        for marker_data in yaml_data["markers"]:
            if marker_data and "stage" in marker_data:
                infleciton_stages.add(marker_data["stage"])

    return tuple(infleciton_stages)


"""
### Markers
Unlike inventory items, patterns and rules, markers are file-specific
and feature specific.
"""


def validate_requested_marker_files(
    feature_marker_files: list[str],
    contingent_feature_marker_files: list[str],
    requested_features: set[str],
) -> bool:
    """
    Ensure that all files are valid, there are no two (simple)
    feature marker exponing the same feature, and no markers
    expone features outside of the requested feature set.

    Does NOT check that all requested features are covered,
    as some may be null-marked.
    """
    feature_marker_yaml = get_yaml_kind("FeatureMarkers")
    covered_features: set[str] = set()

    for marker_file in feature_marker_files:
        if marker_file in feature_marker_yaml["invalid"]:
            logger.exception(
                f"Cannot perform inflection as source FeatureMarkers file {marker_file} is invalid."
            )
            return False
        resolved_yaml = [
            data
            for filename, data in feature_marker_yaml["valid"]
            if filename == marker_file
        ]
        if not resolved_yaml:
            logger.exception(f"Marker file {marker_file} not found.")
            return False
        data = resolved_yaml[0]
        feature = data["feature"]
        if feature in covered_features:
            logger.exception(f"Found duplicate marker files for feature {feature}.")
            return False
        if feature not in requested_features:
            logger.exception(
                f"Found marker file for feature {feature} outside of requested features."
            )
            return False
        covered_features.add(feature)

    contingent_marker_yaml = get_yaml_kind("ContingentFeatureMarkers")
    for marker_file in contingent_feature_marker_files:
        if marker_file in contingent_marker_yaml["invalid"]:
            logger.exception(
                f"Cannot perform inflection as source ContingentFeatureMarkers file {marker_file} is invalid."
            )
            return False
        resolved_yaml = [
            data
            for filename, data in contingent_marker_yaml["valid"]
            if filename == marker_file
        ]
        if not resolved_yaml:
            logger.exception(f"Marker file {marker_file} not found.")
            return False
        data = resolved_yaml[0]
        features = data["features"]
        for feature in features:
            if feature not in requested_features:
                logger.exception(
                    f"Found contingent marker file for feature {feature} outside of requested features."
                )
                return False

    return True


FeatureComboType = set[tuple[str, str]]


def get_markers(
    feature_marker_files: list[str],
    contingent_feature_marker_files: list[str],
    feature_values: set[tuple[str, str]] | dict[str, str],
    lexical_feature_names: set[str] | None = None,
) -> list[tuple[Marker, FeatureComboType]]:
    """
    Query all specified files for markers exponing the requested feature set.
    Selects contingent feature markers first, then regular feature markers
    for any features that still need to be exponed.

    Overlap is allowed between different sets of contingent markers (e.g. if
    one contingent marker set expones number and person and another expones
    person and gender, then both may be selected) but overlap is NOT allowed between
    contingent marker sets and regular marker sets, or between different regular marker sets.
    """

    if isinstance(feature_values, (dict, frozendict)):
        feature_values: FeatureComboType = set(feature_values.items())

    if not feature_values:
        return []

    if lexical_feature_names is None:
        lexical_feature_names = set()

    unexponed_features = {
        feature for feature, _ in feature_values if feature not in lexical_feature_names
    }
    markers = []

    # iterate through contingent feature markers and attempt
    # to match any valid markers within
    # since contingent markers can overlap, order of iteration does not matter
    for contingent_file in contingent_feature_marker_files:
        data = get_yaml_data_safe("ContingentFeatureMarkers", contingent_file)
        result = _get_valid_contingent_markers(data, feature_values)
        if result is not None:
            realization, matched_features = result
            contingent_feature_names = data.get("features", [])

            inflectional_contingent_names = [
                f for f in contingent_feature_names if f not in lexical_feature_names
            ]
            unexponed_features -= set(inflectional_contingent_names)

            contingent_feature_values = {
                (f, v) for f, v in matched_features if f not in lexical_feature_names
            }
            markers.extend(
                (marker, contingent_feature_values) for marker in realization
            )

    # attempt to match any remaining features with regular feature markers
    for marker_file in feature_marker_files:
        data = get_yaml_data_safe("FeatureMarkers", marker_file)
        marker_feature = data["feature"]
        if marker_feature in unexponed_features:
            requested_feature_value = [
                value for feature, value in feature_values if feature == marker_feature
            ]
            if not requested_feature_value:
                continue
            requested_feature_value = requested_feature_value[0]
            if markers_for_file := data["markers"].get(requested_feature_value, None):
                marker_feature_set: FeatureComboType = {
                    (marker_feature, requested_feature_value)
                }
                unexponed_features -= {data["feature"]}
                markers.extend(
                    (marker, marker_feature_set) for marker in markers_for_file
                )

    if unexponed_features:
        raise ValueError("Provided marker sets do not support requested feature set")

    markers = [(resolve_marker(marker), feature_set) for marker, feature_set in markers]
    return markers


def _get_valid_contingent_markers(
    data: dict,
    feature_values: set[tuple[str, str]] | dict[str, str],
) -> tuple[list[Marker], set[tuple[str, str]]] | None:
    """
    Attempt to find a valid contingent marker which matches the requested features.
    """
    lookup = dict(feature_values)
    features_list = data.get("features", [])
    if not features_list:
        return None

    curr = data.get("markers", {})
    matched_features = set()

    for f in features_list:
        if f not in lookup:
            return None
        val = lookup[f]
        if not isinstance(curr, dict) or val not in curr:
            return None
        curr = curr[val]
        matched_features.add((f, val))

    if curr is None:
        return [], matched_features

    if not isinstance(curr, list):
        return None

    return curr, matched_features


def kind_dir(kind: str) -> str:
    return os.path.join(get_yaml_dir(), CONFIG_KIND_TO_PARDIR[kind], kind)
