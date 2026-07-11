"""
Functional FST compilation for Rules and Markers.

Caches:
  rule FSTs   → in-memory only  (rules + inv + feat dirs)
  marker FSTs → in-memory only  (cleared on source changes)
"""

from __future__ import annotations

import pynini

from parC.fst_utils import ReservedSymbolMixin as R
from parC.yaml_utils.models import (
    Marker,
    Rule,
    SimpleRule,
    StringMapRule,
    RuleSequence,
    SingleStringMarker,
    StringTupleMarker,
    UnorderedMarker,
    PrincipalPartMarker,
)
from parC.yaml_utils.yaml_server import get_rules, kind_dir
from parC.grammar.acceptor_compilation import (
    fsa,
    word_fsa,
    get_sigma_star,
    get_symbol_table,
)
from parC.yaml_utils.cache import observed_cache, compute_cache_key, get_cached_fst, save_cached_fst

INVENTORY_DIR = kind_dir("Inventory")
FEATURES_DIR = kind_dir("FeatureDefinitions")
RULES_DIR = kind_dir("Rules")

"""
## Rule compilation
"""


def _compile_simple_rule(rule: SimpleRule) -> pynini.Fst:
    sigma_star = get_sigma_star()
    tau = pynini.cross(fsa(rule.input_pattern), fsa(rule.output_pattern)).optimize()
    l = fsa(rule.left_context) if rule.left_context else ""
    r = fsa(rule.right_context) if rule.right_context else ""
    return pynini.cdrewrite(tau, l, r, sigma_star)


def _compile_string_map_rule(rule: StringMapRule) -> pynini.Fst:
    sigma_star = get_sigma_star()
    tau = pynini.union(
        *[pynini.cross(fsa(i), fsa(o)) for i, o in rule.string_map]
    ).optimize()
    l = fsa(rule.left_context) if rule.left_context else ""
    r = fsa(rule.right_context) if rule.right_context else ""
    return pynini.cdrewrite(tau, l, r, sigma_star)


def compile_rule(rule: Rule) -> pynini.Fst | list[pynini.Fst]:
    if isinstance(rule, SimpleRule):
        return _compile_simple_rule(rule)
    if isinstance(rule, StringMapRule):
        return _compile_string_map_rule(rule)
    if isinstance(rule, RuleSequence):
        rules = get_rules()
        result: list[pynini.Fst] = []
        for name in rule.rules:
            sub_fst = compile_rule(rules[name])
            if isinstance(sub_fst, list):
                result.extend(sub_fst)
            else:
                result.append(sub_fst)
        return result
    raise ValueError(f"Unknown rule type: {type(rule)!r}")


"""
## Marker compilation
"""


def _compile_prefix(value: str) -> pynini.Fst:
    sigma_star = get_sigma_star()
    syms = get_symbol_table()
    bow = pynini.accep(R.bow, token_type=syms)
    tau = pynini.cross(bow, pynini.concat(bow, fsa(value)))
    return pynini.cdrewrite(tau, "", "", sigma_star)


def _compile_suffix(value: str) -> pynini.Fst:
    sigma_star = get_sigma_star()
    syms = get_symbol_table()
    eow = pynini.accep(R.eow, token_type=syms)
    tau = pynini.cross(eow, pynini.concat(fsa(value), eow))
    return pynini.cdrewrite(tau, "", "", sigma_star)


def _compile_string_map(string_map: tuple[tuple[str, str], ...]) -> pynini.Fst:
    # word-level substitution: cross(word_fsa(root), word_fsa(pp)) per entry
    return pynini.union(
        *[pynini.cross(word_fsa(i), word_fsa(o)) for i, o in string_map]
    ).optimize()


def compile_marker(marker: Marker) -> pynini.Fst:
    if isinstance(marker, SingleStringMarker):
        if marker.kind == "prefix":
            return _compile_prefix(marker.value)
        if marker.kind == "suffix":
            return _compile_suffix(marker.value)
        if marker.kind == "suppletion":
            sigma_star = get_sigma_star()
            tau = pynini.cross(sigma_star, fsa(marker.value))
            return pynini.cdrewrite(tau, "", "", sigma_star)
        if marker.kind == "rule":
            rules = get_rules()
            rule_name = marker.value.removeprefix("$")
            if rule_name not in rules:
                raise KeyError(
                    f"Rule '{marker.value}' not found in set of rules {list(rules.keys())}"
                )
            result = compile_rule(rules[rule_name])
            if isinstance(result, list):
                composed = result[0]
                for f in result[1:]:
                    composed = pynini.compose(composed, f)
                return composed
            return result
    if isinstance(marker, StringTupleMarker) and marker.kind == "replace":
        sigma_star = get_sigma_star()
        tau = pynini.cross(fsa(marker.value[0]), fsa(marker.value[1]))
        return pynini.cdrewrite(tau, "", "", sigma_star)
    if isinstance(marker, PrincipalPartMarker) and marker.kind == "string_map":
        return _compile_string_map(marker.value)
    if isinstance(marker, UnorderedMarker) and marker.kind == "principal_part":
        raise ValueError(
            "UnorderedMarker(principal_part) must be resolved to StringMapMarker "
            "via get_markers_for_paradigm before compilation"
        )
    raise ValueError(f"Unknown marker: {marker!r}")


from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex

"""
Public API
"""


import hashlib
import json

def get_rule_fst_key(rule_name: str) -> str:
    rule_name = rule_name.removeprefix("$")
    config_dirs = [
        kind_dir("Rules"),
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
    child_keys = {}
    rules = get_rules()
    rule = rules.get(rule_name)
    if isinstance(rule, RuleSequence):
        for sub_rule_name in rule.rules:
            child_keys[f"Rule/{sub_rule_name}"] = get_rule_fst_key(sub_rule_name)
    from parC.grammar.acceptor_compilation import get_symbol_table_key
    child_keys["symbol_table"] = get_symbol_table_key()
    return compute_cache_key(rule_name, "Rule", config_dirs, child_keys)


def get_marker_fst_key(marker: Marker) -> str:
    config_dirs = [
        kind_dir("Rules"),
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
        kind_dir("FeatureMarkers"),
        kind_dir("ContingentFeatureMarkers"),
    ]
    child_keys = {}
    if hasattr(marker, "kind") and marker.kind == "rule":
        rule_name = marker.value.removeprefix("$")
        child_keys[f"Rule/{rule_name}"] = get_rule_fst_key(rule_name)
    from parC.grammar.acceptor_compilation import get_symbol_table_key
    child_keys["symbol_table"] = get_symbol_table_key()
    
    marker_dict = marker._asdict()
    marker_json = json.dumps(marker_dict, sort_keys=True)
    marker_hash = hashlib.sha256(marker_json.encode("utf-8")).hexdigest()
    
    return compute_cache_key(f"marker_{marker_hash}", "Marker", config_dirs, child_keys)


@observed_cache(
    [
        kind_dir("Rules"),
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
)
def get_rule_fst(rule_name: str) -> pynini.Fst | list[pynini.Fst]:
    rule_name = rule_name.removeprefix("$")
    cache_key = get_rule_fst_key(rule_name)
    cached = get_cached_fst(cache_key)
    if cached is not None:
        return cached

    rules = get_rules()
    if rule_name not in rules:
        raise KeyError(
            f"Rule '{rule_name}' not found in set of rules {list(rules.keys())}"
        )
    rule = rules[rule_name]

    if isinstance(rule, RuleSequence):
        result = [get_rule_fst(name) for name in rule.rules]
        # rule sequence returns a flat list of FSTs
        flat_result = []
        for item in result:
            if isinstance(item, list):
                flat_result.extend(item)
            else:
                flat_result.append(item)
        save_cached_fst(cache_key, flat_result)
        return flat_result

    compiled = compile_rule(rule)
    save_cached_fst(cache_key, compiled)
    return compiled


@observed_cache(
    [
        kind_dir("Rules"),
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
        kind_dir("FeatureMarkers"),
    ]
)
def get_marker_fst(marker: Marker) -> pynini.Fst:
    cache_key = get_marker_fst_key(marker)
    cached = get_cached_fst(cache_key)
    if cached is not None:
        return cached
    compiled = compile_marker(marker)
    save_cached_fst(cache_key, compiled)
    return compiled
