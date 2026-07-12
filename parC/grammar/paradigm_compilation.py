"""
Paradigm inflect / parse / fuzzy-search graph compilation.

Caches:
  inflect + parse + search FSTs → get_yaml_dir()/.cache/Paradigm/{name}.{kind}.fst
  Invalidated when any of Paradigm, FeatureMarkers, ContingentFeatureMarkers,
  Inventory, FeatureDefinitions, or Rules dirs change.
"""

from __future__ import annotations

import os
import re

import pynini
from loguru import logger
from pynini.lib import pynutil
from typing import NamedTuple
from frozendict import frozendict

from parC.yaml_utils.cache import observed_cache, compute_cache_key
from parC.fst_utils import ReservedSymbolMixin as R
from parC.fst_utils import stringify_features
from parC.constants import get_yaml_dir
from parC.lexicon import get_gloss_for_root, get_roots, get_roots_with_lexical_features
from parC.yaml_utils.yaml_server import (
    get_feature_map,
    get_yaml_data_safe,
    kind_dir,
)
from parC.grammar.acceptor_compilation import (
    fsa,
    fsm_strings,
    fsm_strings_and_weights,
    word_fsa,
    get_sigma_star,
    get_special_fsas,
    get_symbol_table,
    filter_strings_by_pattern,
)
from parC.grammar.marker_resolution import (
    get_feature_combos_for_paradigm,
    get_features_for_paradigm,
    get_markers_for_paradigm,
    get_fixed_features_for_paradigm,
)

EDIT_BOUND = 5
EDIT_COST = 1.0


"""
## Helpers
"""


def _apply_markers(stem_fst: pynini.Fst, markers: list) -> pynini.Fst:
    from parC.grammar.transducer_compilation import get_marker_fst

    current = stem_fst
    for marker in markers:
        current = pynini.compose(current, get_marker_fst(marker))
    return current


@observed_cache([get_yaml_dir()])
def get_roots_for_paradigm(paradigm_name: str) -> list[str]:
    """
    Get the roots associated with a given paradigm.
    Paradigms may further filter roots from their associated
    part of speech using either a set of lexical feature values
    or a regex pattern filter.
    """
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    part_of_speech = paradigm_data["part_of_speech"]

    if filter := paradigm_data.get("filter", None):
        if lexical_features := filter.get("lexical_features", None):
            roots = get_roots_with_lexical_features(
                part_of_speech, lexical_features=lexical_features
            )
        else:
            roots = get_roots(part_of_speech)
        if pattern_filter := filter.get("pattern", None):
            roots_fsa = pynini.union(*[word_fsa(root) for root in roots])
            roots = filter_strings_by_pattern(roots_fsa, pattern_filter)
    else:
        roots = get_roots(part_of_speech)

    return roots


"""
## Graph builders
"""


def build_inflect_graph(paradigm_name: str) -> pynini.Fst:
    """root[features...] → surface form."""
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    feature_map = get_feature_map()
    roots = get_roots_for_paradigm(paradigm_name=paradigm_name)
    combos, _, _ = get_feature_combos_for_paradigm(
        name=paradigm_name, feature_map=feature_map, kind="Paradigm"
    )

    inflect_fsts: list[pynini.Fst] = []
    for root in roots:
        root_fsa = word_fsa(root)
        for feature_values in combos:
            try:
                markers = get_markers_for_paradigm(
                    feature_values, paradigm_name, root=root
                )
                inflected_output = pynini.project(
                    _apply_markers(root_fsa, markers), project_type="output"
                )
            except Exception:
                # logger.debug(
                #     f"Skipping {paradigm_name} root={root} fv={feature_values}: {e}"
                # )
                continue

            feature_str = stringify_features(feature_values)
            inflect_input = (
                pynini.concat(root_fsa, fsa(feature_str)) if feature_str else root_fsa
            )
            inflect_fsts.append(
                pynini.cross(inflect_input, inflected_output).optimize()
            )

    if not inflect_fsts:
        return pynini.Fst()
    return pynini.union(*inflect_fsts).optimize()


def _get_all_markers_from_config(paradigm_name: str) -> list[Marker]:
    from parC.yaml_utils.yaml_server import get_yaml_data_safe
    from parC.yaml_utils.models import resolve_marker
    from parC.grammar.op_tags import extract_contingent_markers

    markers = []
    seen = set()

    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if not paradigm_data:
        return []

    global_stage = paradigm_data.get("global_stage", None)

    def add_marker(m_dict):
        try:
            resolved = resolve_marker(m_dict)
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                markers.append(resolved)
            if global_stage and hasattr(resolved, "stage") and resolved.stage is None:
                staged = resolved._replace(stage=global_stage)
                staged_key = str(staged)
                if staged_key not in seen:
                    seen.add(staged_key)
                    markers.append(staged)
        except Exception:
            pass

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

    return markers


@observed_cache([get_yaml_dir()])
def compile_paradigm_grammar(
    paradigm_name: str,
    infer_lexical_features: bool = False,
    lexical_features: tuple[tuple[str, str], ...] | None = None,
) -> pynini.Fst:
    """
    Compile a root-independent paradigm transducer (T_paradigm) exactly once.
    Covers Phase 1, Phase 2, and Phase 3 of the paradigm FST compilation pipeline.
    """
    from parC.grammar.op_tags import get_op_tag

    # Phase 1: Label-to-Marker Transducer
    current_fst = get_label_to_marker_fst(
        paradigm_name,
        infer_lexical_features=infer_lexical_features,
        lexical_features=lexical_features,
    )

    # Discover and order all stages present on the paradigm's markers
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    unique_markers = _get_all_markers_from_config(paradigm_name)

    # Order stages
    stage_order = list(paradigm_data.get("stage_order", []))
    if "principal_part" not in stage_order:
        stage_order.insert(0, "principal_part")

    stages_present = {getattr(m, "stage", None) for m in unique_markers}
    execution_stages = []
    for s in stage_order:
        if s in stages_present:
            execution_stages.append(s)

    other_stages = [s for s in stages_present if s not in stage_order]
    other_stages = sorted(other_stages, key=lambda x: (x is None, str(x)))
    execution_stages.extend(other_stages)

    # Sequentially compose Phase 1 FST with Phase 2's stage realization transducers
    for stage in execution_stages:
        stage_realization = get_stage_realization_fst(paradigm_name, stage)
        current_fst = pynini.compose(current_fst, stage_realization)

    # Compose with Phase 3's final surface filter
    final_filter = get_final_surface_filter_fst(paradigm_name)
    current_fst = pynini.compose(current_fst, final_filter)

    return current_fst


def build_inflect_graph_for_root_regex(
    paradigm_name: str,
    root_regex: str | pynini.Fst,
    lexical_features: FeatureComboType | dict[str, str] | None = None,
    infer_lexical_features: bool = False,
) -> pynini.Fst:
    """root_regex[lexical_features][inflectional_features] → surface form."""
    import itertools

    if isinstance(root_regex, str):
        root_fsa = fsa(R.bow + root_regex + R.eow)
    else:
        root_fsa = root_regex

    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    feature_map = get_feature_map()
    combos, _, _ = get_feature_combos_for_paradigm(
        name=paradigm_name, feature_map=feature_map, kind="Paradigm"
    )

    if infer_lexical_features:
        part_of_speech = paradigm_data["part_of_speech"]
        part_of_speech_data = get_yaml_data_safe(
            yaml_basename=part_of_speech, kind="PartOfSpeech"
        )
        lexical_feature_names = part_of_speech_data.get("lexical_features", [])

        # Prune lexical features to only those referenced in this paradigm's rules/markers
        referenced_lexical_features = set()
        contingent_files = paradigm_data.get("contingent_markers", [])
        for contingent_file in contingent_files:
            contingent_data = get_yaml_data_safe("ContingentFeatureMarkers", contingent_file)
            if contingent_data:
                for f in contingent_data.get("features", []):
                    if f in lexical_feature_names:
                        referenced_lexical_features.add(f)
        for f in paradigm_data.get("feature_markers", {}).keys():
            if f in lexical_feature_names:
                referenced_lexical_features.add(f)

        lexical_value_lists = []
        for fname in lexical_feature_names:
            if fname not in feature_map:
                logger.warning(
                    f"Lexical feature '{fname}' not in feature map — skipping."
                )
                continue
            if fname in referenced_lexical_features:
                lexical_value_lists.append([(fname, v) for v in feature_map[fname]])

        if not lexical_value_lists:
            lexical_combos = [set()]
        else:
            lexical_combos = [
                set(combo_tuples)
                for combo_tuples in itertools.product(*lexical_value_lists)
            ]
    else:
        if lexical_features:
            if isinstance(lexical_features, (dict, frozendict)):
                lex_set = set(lexical_features.items())
            else:
                lex_set = set(lexical_features)
            lexical_combos = [lex_set]
            referenced_lexical_features = {f for f, _ in lex_set}
            lexical_feature_names = sorted(list(referenced_lexical_features))
        else:
            lexical_combos = [set()]
            referenced_lexical_features = set()
            lexical_feature_names = []

    from collections import defaultdict
    marker_to_combinations = defaultdict(list)
    input_parts = []
    tag_seqs = []

    # We can pre-compile tag acceptors to avoid calling fsa(...) repeatedly.
    tag_fsas = {}
    for fname, fvals in feature_map.items():
        for val in fvals:
            tag_str = f"[{fname}={val}]"
            tag_fsas[tag_str] = pynini.accep(tag_str, token_type=get_symbol_table())

    def get_feature_fsa(feat_vals: FeatureComboType | dict[str, str]) -> pynini.Fst | None:
        if isinstance(feat_vals, (dict, frozendict)):
            feat_vals = list(feat_vals.items())
        sorted_feats = sorted(feat_vals)
        if not sorted_feats:
            return None
        parts = [tag_fsas[f"[{f}={v}]"] for f, v in sorted_feats]
        curr = parts[0]
        for part in parts[1:]:
            curr = pynini.concat(curr, part)
        return curr

    for lexical_combo in lexical_combos:
        for feature_values in combos:
            try:
                markers = get_markers_for_paradigm(
                    feature_values,
                    paradigm_name,
                    root=None,
                    lexical_features=lexical_combo,
                    include_features=True,
                )
            except Exception:
                # logger.debug(
                #     f"Skipping {paradigm_name} root_regex={root_regex} fv={feature_values} lex={lexical_combo}: {e}"
                # )
                continue

            if infer_lexical_features:
                lex_tags = []
                for fname in lexical_feature_names:
                    val = next((v for f, v in lexical_combo if f == fname), None)
                    if val is not None:
                        lex_tags.append(f"[{fname}={val}]")
                inflect_tags = [f"[{f}={v}]" for f, v in sorted(list(feature_values))]
                combo_tags = lex_tags + inflect_tags
            else:
                combo_tags = [f"[{f}={v}]" for f, v in sorted(list(feature_values))]
            for marker, _ in markers:
                marker_to_combinations[marker].append(combo_tags)

            # Build the lexical FST preserving alphabetical order of all lexical features.
            # Unreferenced lexical features are unioned over all their possible values in FST space.
            lexical_parts = []
            for fname in lexical_feature_names:
                if fname in referenced_lexical_features:
                    val = next(v for f, v in lexical_combo if f == fname)
                    lexical_parts.append(tag_fsas[f"[{fname}={val}]"])
                else:
                    lexical_parts.append(
                        pynini.union(*[tag_fsas[f"[{fname}={v}]"] for v in feature_map[fname]])
                    )
            if lexical_parts:
                lexical_fsa = lexical_parts[0]
                for part in lexical_parts[1:]:
                    lexical_fsa = pynini.concat(lexical_fsa, part)
            else:
                lexical_fsa = None

            inflectional_fsa = get_feature_fsa(feature_values)

            # Gather tag sequences for the tag domain to perform static difference
            if infer_lexical_features and lexical_fsa:
                if inflectional_fsa is not None:
                    tag_seq = pynini.concat(lexical_fsa, inflectional_fsa)
                else:
                    tag_seq = lexical_fsa
            else:
                tag_seq = inflectional_fsa
            if tag_seq:
                tag_seqs.append(tag_seq)

            inflect_input = root_fsa
            if infer_lexical_features and lexical_fsa:
                inflect_input = pynini.concat(inflect_input, lexical_fsa)

            if inflectional_fsa is not None:
                inflect_input = pynini.concat(inflect_input, inflectional_fsa)
            input_parts.append(inflect_input)

    if not input_parts:
        return pynini.Fst()

    cascade_domain = pynini.union(*input_parts).optimize()

    # Retrieve pre-compiled universal grammar FST and compose directly with cascade_domain
    lexical_features_hashable = None
    if lexical_features:
        if isinstance(lexical_features, (dict, frozendict)):
            lexical_features_hashable = tuple(sorted(lexical_features.items()))
        else:
            lexical_features_hashable = tuple(sorted(list(lexical_features)))

    grammar_fst = compile_paradigm_grammar(
        paradigm_name,
        infer_lexical_features=infer_lexical_features,
        lexical_features=lexical_features_hashable,
    )
    return pynini.compose(cascade_domain, grammar_fst).optimize()


def build_parse_graph(inflect_graph: pynini.Fst) -> pynini.Fst:
    return pynini.invert(inflect_graph).optimize()


def build_search_lexicon_and_leftfactor(
    inflect_graph: pynini.Fst,
) -> tuple[pynini.Fst, pynini.Fst]:
    """Fuzzy-searchable form lattice via edit transducers."""
    sigma = get_special_fsas()["sigma"]
    sigma_star = get_sigma_star()
    syms = get_symbol_table()

    insert_fst = pynutil.insert(
        pynini.accep(R.insert, weight=EDIT_COST / 2, token_type=syms)
    )
    delete_fst = pynini.cross(
        sigma,
        pynini.accep(R.delete, weight=EDIT_COST / 2, token_type=syms),
    )
    substitute_fst = pynini.cross(
        sigma,
        pynini.accep(R.substitute, weight=EDIT_COST / 2, token_type=syms),
    )
    edit_fst = pynini.union(insert_fst, delete_fst, substitute_fst).optimize()

    left_factor = sigma_star.copy()
    for _ in range(EDIT_BOUND):
        left_factor = pynini.concat(
            left_factor, pynini.concat(edit_fst.ques, sigma_star)
        )
    left_factor.optimize()

    right_factor = pynini.invert(left_factor)
    insert_label = syms.find(R.insert)
    delete_label = syms.find(R.delete)
    right_factor = right_factor.relabel_pairs(
        ipairs=[(insert_label, delete_label), (delete_label, insert_label)]
    )

    form_lattice = pynini.project(inflect_graph, project_type="output")
    search_lexicon = pynini.compose(right_factor, form_lattice).optimize()
    return search_lexicon, left_factor


"""
## Cache warming and public entry points
"""

_FST_KINDS = ("inflect", "parse", "search_lexicon", "search_left_factor")


def get_paradigm_cache_key(paradigm_name: str) -> str:
    config_dirs = [
        kind_dir("Paradigm"),
        kind_dir("PartOfSpeech"),
        kind_dir("Wordlists"),
        kind_dir("FeatureMarkers"),
        kind_dir("ContingentFeatureMarkers"),
        kind_dir("Rules"),
        kind_dir("Patterns"),
        kind_dir("Inventory"),
        kind_dir("FeatureDefinitions"),
    ]
    child_keys = {}
    try:
        from parC.grammar.transducer_compilation import get_marker_fst_key
        feature_map = get_feature_map()
        combos, _, _ = get_feature_combos_for_paradigm(
            name=paradigm_name, feature_map=feature_map, kind="Paradigm"
        )
        seen_markers = set()
        roots = get_roots_for_paradigm(paradigm_name)
        for combo in combos:
            try:
                markers = get_markers_for_paradigm(combo, paradigm_name, root=None)
                for marker in markers:
                    seen_markers.add(marker)
            except Exception:
                for root in roots[:5]:
                    try:
                        markers = get_markers_for_paradigm(combo, paradigm_name, root=root)
                        for marker in markers:
                            if isinstance(marker, tuple):
                                seen_markers.add(marker[0])
                            else:
                                seen_markers.add(marker)
                    except Exception:
                        pass
        
        for marker in seen_markers:
            m_key = get_marker_fst_key(marker)
            child_keys[f"Marker/{m_key}"] = m_key
    except Exception as e:
        logger.warning(f"Error resolving marker dependencies for paradigm {paradigm_name}: {e}")
        
    description = f"Paradigm '{paradigm_name}' ({len(seen_markers)} markers resolved)"
    return compute_cache_key(paradigm_name, "Paradigm", config_dirs, child_keys, description=description)


def _load_paradigm_cached(cache_key: str) -> tuple[pynini.Fst, pynini.Fst, pynini.Fst, pynini.Fst] | None:
    from parC.yaml_utils.cache import CACHE_DIR, record_cache_miss
    fsts = []
    for k in _FST_KINDS:
        path = os.path.join(CACHE_DIR, f"{cache_key}_{k}.fst")
        if not os.path.exists(path):
            logger.debug(f"Paradigm cache MISS for key {cache_key}: missing {k}.fst")
            record_cache_miss(cache_key)
            return None
        try:
            fsts.append(pynini.Fst.read(path))
        except Exception as e:
            logger.debug(f"Paradigm cache MISS for key {cache_key}: failed to read {k}.fst: {e}")
            record_cache_miss(cache_key)
            return None
    logger.debug(f"Paradigm cache HIT for key {cache_key}")
    return tuple(fsts)


def _save_paradigm_cached(
    cache_key: str,
    inflect: pynini.Fst,
    parse: pynini.Fst,
    search_lexicon: pynini.Fst,
    search_left_factor: pynini.Fst,
) -> None:
    from parC.yaml_utils.cache import CACHE_DIR, record_cache_save
    inflect.write(os.path.join(CACHE_DIR, f"{cache_key}_inflect.fst"))
    parse.write(os.path.join(CACHE_DIR, f"{cache_key}_parse.fst"))
    search_lexicon.write(os.path.join(CACHE_DIR, f"{cache_key}_search_lexicon.fst"))
    search_left_factor.write(os.path.join(CACHE_DIR, f"{cache_key}_search_left_factor.fst"))
    record_cache_save(cache_key)


@observed_cache([get_yaml_dir()])
def _get_or_build(
    paradigm_name: str, graph_type: str, force_rebuild: bool = False
) -> pynini.Fst:
    graph_index = _FST_KINDS.index(graph_type)
    cache_key = get_paradigm_cache_key(paradigm_name)
    if not force_rebuild:
        loaded = _load_paradigm_cached(cache_key)
        if loaded is not None:
            return loaded[graph_index]

    inflect = build_inflect_graph(paradigm_name)
    parse = build_parse_graph(inflect)
    search_lexicon, search_left_factor = build_search_lexicon_and_leftfactor(inflect)
    _save_paradigm_cached(cache_key, inflect, parse, search_lexicon, search_left_factor)

    graph_tuple = (inflect, parse, search_lexicon, search_left_factor)
    return graph_tuple[graph_index]


def get_inflect_graph(paradigm_name: str) -> pynini.Fst:
    return _get_or_build(paradigm_name, "inflect")


def get_parse_graph(paradigm_name: str) -> pynini.Fst:
    return _get_or_build(paradigm_name, "parse")


@observed_cache([get_yaml_dir()])
def get_open_parse_graph(paradigm_name: str) -> pynini.Fst:
    inflect_graph = build_inflect_graph_for_root_regex(paradigm_name, "<Phone>*", infer_lexical_features=True)
    return pynini.invert(inflect_graph).optimize()


def get_search_graphs(paradigm_name: str) -> tuple[pynini.Fst, pynini.Fst]:
    return (
        _get_or_build(paradigm_name, "search_lexicon"),
        _get_or_build(paradigm_name, "search_left_factor"),
    )


"""
Public API
"""


@observed_cache([get_yaml_dir()])
def parse(form: str, kind: str = "Paradigm", name: str = "", open_ended: bool = False) -> list[dict]:
    form_fsa = word_fsa(form)
    parse_graph = get_open_parse_graph(name) if open_ended else get_parse_graph(name)
    paradigm_data = get_yaml_data_safe(yaml_basename=name, kind=kind)
    lexicon_basename = paradigm_data.get("part_of_speech", "")

    parse_lattice = (form_fsa @ parse_graph).optimize()
    parse_strs = fsm_strings(parse_lattice)
    parses = []
    for s in parse_strs:
        feat_matches = re.findall(r"\[([^=\]]+)=([^\]]+)\]", s)
        root = re.sub(r"\[[^\]]+\]", "", s).strip()
        gloss = get_gloss_for_root(lexicon_basename, root)
        parses.append({"root": root, "features": dict(feat_matches), "gloss": gloss})

    return parses


@observed_cache([get_yaml_dir()])
def inflect(
    root: str,
    feature_values: set[tuple[str, str]] | dict[str, str],
    name: str,
) -> list[str]:

    if isinstance(feature_values, (dict, frozendict)):
        feature_values = set(feature_values.items())

    fixed_features = get_fixed_features_for_paradigm(name=name, kind="Paradigm")
    feature_values |= fixed_features
    features = set(feature for feature, _ in feature_values)
    expected_features = get_features_for_paradigm(name)
    if not features == expected_features:
        raise ValueError(
            f"Feature set {features} does not match expected features {expected_features} for paradigm '{name}'."
        )

    inflect_graph = get_inflect_graph(name)
    feature_str = stringify_features(feature_values)
    input_fsa = (
        pynini.concat(word_fsa(root), fsa(feature_str))
        if feature_str
        else word_fsa(root)
    )
    output_lattice = pynini.compose(input_fsa, inflect_graph).optimize()
    output_lattice = pynini.project(output_lattice, project_type="output")
    surface_forms = fsm_strings(output_lattice, strip_all_tags=True)

    return surface_forms


class InflectStage(NamedTuple):
    root: str
    stage: str
    surface_forms: list[str]
    feature_values: set[tuple[str, str]] | str = ""
    marker_kind: str = ""
    marker_value: str = ""


@observed_cache([get_yaml_dir()])
def inflect_stages(
    root: str,
    feature_values: tuple[tuple[str, str]],
    name: str,
) -> list[InflectStage]:
    """
    Inflect word and save table with each successive stage of inflection,
    returning in a format for printing to a table with the following format:

    | Root  | Features                  | Marker | Form       |
    | $root | Initial                   |        | $root      |
    | $root | person=1sg, tense=present | suffix | $root-suff |
    ...
    | $root | Final                     |        | $surface   |

    """

    if isinstance(feature_values, (dict, frozendict)):
        feature_values = set(feature_values.items())

    fixed_features = get_fixed_features_for_paradigm(name=name, kind="Paradigm")
    feature_values |= fixed_features
    features = set(feature for feature, _ in feature_values)
    expected_features = get_features_for_paradigm(name)
    if not features == expected_features:
        raise ValueError(
            f"Feature set {features} does not match expected features {expected_features} for paradigm '{name}'."
        )

    marker_tuples = get_markers_for_paradigm(
        feature_values, name, include_features=True, root=root
    )

    initial_stage = InflectStage(
        root=root,
        surface_forms=[root],
        stage="Initial",
    )
    current_fst = word_fsa(root)
    stages = [initial_stage]
    for marker, marker_features in marker_tuples:
        current_fst = _apply_markers(current_fst, [marker])
        surface_forms = fsm_strings(
            current_fst, nshortest=5, strip_word_edge_symbols=True
        )
        marker_value = (
            marker.display_value if hasattr(marker, "display_value") else marker.value
        )
        current_stage = InflectStage(
            root=root,
            feature_values=marker_features,
            surface_forms=surface_forms,
            marker_kind=marker.kind,
            marker_value=marker_value,
            stage=marker.stage,
        )
        stages.append(current_stage)

    final_strings = fsm_strings(current_fst, nshortest=5, strip_all_tags=True)
    final_stage = InflectStage(
        root=root,
        surface_forms=final_strings,
        stage="final",
    )
    stages.append(final_stage)

    # prepare feature sets for printing
    for i, stage in enumerate(stages):
        if isinstance(stage.feature_values, set):
            feature_string = stringify_features(stage.feature_values)
            feature_string = feature_string.lstrip("[").rstrip("]")
            feature_string = feature_string.replace("][", ", ")
            stage = stage._replace(feature_values=feature_string)

        if isinstance(stage.marker_value, tuple):
            marker_value_str = " > ".join(stage.marker_value)
            stage = stage._replace(marker_value=marker_value_str)

        stages[i] = stage
    return stages


@observed_cache([get_yaml_dir()])
def search(
    kind: str, name: str, form: str, nshortest: int, do_parse: bool = True, open_ended: bool = False
) -> list[tuple[str, float]] | list[dict]:
    search_lexicon, left_factor = get_search_graphs(name)
    form_fsa = word_fsa(form)
    left_factor_lattice = pynini.compose(form_fsa, left_factor).optimize()
    edit_graph = pynini.compose(left_factor_lattice, search_lexicon)
    hits = fsm_strings_and_weights(edit_graph, strip_all_tags=True, nshortest=nshortest)

    if do_parse:
        parses = []
        for hit, weight in hits:
            current_parse = [item.copy() for item in parse(hit, kind=kind, name=name, open_ended=open_ended)]
            [parse_item.update(edit_distance=weight) for parse_item in current_parse]
            [parse_item.update(form=hit) for parse_item in current_parse]
            parses.extend(current_parse)
        return parses

    return hits


def get_label_to_marker_fst(
    paradigm_name: str,
    infer_lexical_features: bool = False,
    lexical_features: FeatureComboType | dict[str, str] | None = None,
) -> pynini.Fst:
    """
    Constructs the exponence transducer. It maps the sequence of feature tags 
    (e.g., [prefix_class=a_stem][aspect=present]) to their corresponding sequence 
    of operational tags for every valid combination of features, while acting 
    as the identity on the stem segments and boundaries.
    """
    from parC.yaml_utils.models import Inventory
    from parC.yaml_utils.yaml_server import get_inventory_items
    from parC.grammar.op_tags import get_op_tag

    # 1. Build identity map for non-feature alphabet
    syms = get_symbol_table()
    inventory = get_inventory_items()
    phones = list(dict.fromkeys(inventory.phones))
    
    non_feature_symbols = (
        phones
        + list(dict.fromkeys(inventory.tags))  # any non-feature tags from inventory
        + list(R.boundary_symbols)
        + list(R.edit_tags)
        + list(R.bow_eow_tags)
    )
    non_feature_fsa = pynini.union(
        *[pynini.accep(sym, token_type=syms) for sym in non_feature_symbols if sym]
    ).optimize()
    non_feature_identity = non_feature_fsa

    # 2. Collect feature combinations
    feature_map = get_feature_map()
    combos, _, _ = get_feature_combos_for_paradigm(
        name=paradigm_name, feature_map=feature_map, kind="Paradigm"
    )

    if infer_lexical_features:
        paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
        part_of_speech = paradigm_data["part_of_speech"]
        part_of_speech_data = get_yaml_data_safe(
            yaml_basename=part_of_speech, kind="PartOfSpeech"
        )
        lexical_feature_names = part_of_speech_data.get("lexical_features", [])

        # Prune lexical features to only those referenced in this paradigm's rules/markers
        referenced_lexical_features = set()
        contingent_files = paradigm_data.get("contingent_markers", [])
        for contingent_file in contingent_files:
            contingent_data = get_yaml_data_safe("ContingentFeatureMarkers", contingent_file)
            if contingent_data:
                for f in contingent_data.get("features", []):
                    if f in lexical_feature_names:
                        referenced_lexical_features.add(f)
        for f in paradigm_data.get("feature_markers", {}).keys():
            if f in lexical_feature_names:
                referenced_lexical_features.add(f)

        lexical_value_lists = []
        for fname in lexical_feature_names:
            if fname not in feature_map:
                continue
            if fname in referenced_lexical_features:
                lexical_value_lists.append([(fname, v) for v in feature_map[fname]])

        if not lexical_value_lists:
            lexical_combos = [set()]
        else:
            import itertools
            lexical_combos = [
                set(combo_tuples)
                for combo_tuples in itertools.product(*lexical_value_lists)
            ]
    else:
        if lexical_features:
            if isinstance(lexical_features, (dict, frozendict)):
                lex_set = set(lexical_features.items())
            else:
                lex_set = set(lexical_features)
            lexical_combos = [lex_set]
            referenced_lexical_features = {f for f, _ in lex_set}
            lexical_feature_names = sorted(list(referenced_lexical_features))
        else:
            lexical_combos = [set()]
            lexical_feature_names = []

    mapping_list = []
    for lexical_combo in lexical_combos:
        for feature_values in combos:
            try:
                markers = get_markers_for_paradigm(
                    feature_values,
                    paradigm_name,
                    root=None,
                    lexical_features=lexical_combo,
                    include_features=True,
                )
            except Exception:
                continue

            # Build the sequence of feature tags as an acceptor (the input side)
            input_tag_strs = []
            if infer_lexical_features:
                for fname in lexical_feature_names:
                    val = next((v for f, v in lexical_combo if f == fname), None)
                    if val is not None:
                        input_tag_strs.append(f"[{fname}={val}]")
            
            # Sort inflectional features alphabetically
            sorted_inflect = sorted(list(feature_values))
            for f, v in sorted_inflect:
                input_tag_strs.append(f"[{f}={v}]")

            if not input_tag_strs:
                input_fsa = pynini.accep("", token_type=syms)
            else:
                input_fsa = pynini.accep(input_tag_strs[0], token_type=syms)
                for tag_str in input_tag_strs[1:]:
                    input_fsa = pynini.concat(input_fsa, pynini.accep(tag_str, token_type=syms))

            # Build the sequence of operational tags as an acceptor (the output side)
            op_tags_seq = []
            for marker, _ in markers:
                op_tags_seq.append(get_op_tag(marker))

            if not op_tags_seq:
                output_fsa = pynini.accep("", token_type=syms)
            else:
                output_fsa = pynini.accep(op_tags_seq[0], token_type=syms)
                for op_tag in op_tags_seq[1:]:
                    output_fsa = pynini.concat(output_fsa, pynini.accep(op_tag, token_type=syms))

            cross_trans = pynini.cross(input_fsa, output_fsa)
            mapping_list.append(cross_trans)

    if not mapping_list:
        return non_feature_identity.star.optimize()

    mapping_union = pynini.union(*mapping_list).optimize()
    exponence_transducer = pynini.union(non_feature_identity, mapping_union).star.optimize()
    return exponence_transducer


get_exponence_transducer = get_label_to_marker_fst


def get_marker_swapping_transducer(marker: Marker) -> pynini.Fst:
    """
    Handles prefix and suffix cdrewrite-based swapping/insertion relative to [BOW] and [EOW],
    and falls back to gated-composition + tag deletion for rule markers.
    """
    from parC.grammar.transducer_compilation import get_marker_fst, get_trigger_fsa
    from parC.grammar.op_tags import get_op_tag
    from parC.grammar.acceptor_compilation import get_symbol_table, get_sigma_star

    syms = get_symbol_table()
    sigma_star = get_sigma_star()
    op_tag = get_op_tag(marker)
    T_tag = pynini.accep(op_tag, token_type=syms)

    trigger_fsa = get_trigger_fsa([op_tag], syms, sigma_star)
    marker_applied = pynini.compose(trigger_fsa, get_marker_fst(marker))
    delete_tag = pynini.cdrewrite(
        pynini.cross(T_tag, pynini.accep("", token_type=syms)),
        "",
        "",
        sigma_star,
    )
    return pynini.compose(marker_applied, delete_tag).optimize()


@observed_cache([get_yaml_dir()])
def get_stage_realization_fst(paradigm_name: str, stage: str) -> pynini.Fst:
    """
    Constructs the stage realization transducer for a given stage of a paradigm.
    This transducer maps strings containing operational tags (flags) associated
    with active markers at this stage to their phonetic realizations, while
    deleting the tags. Strings not containing any active tags pass through unchanged.
    """
    from parC.grammar.transducer_compilation import get_trigger_fsa
    from parC.grammar.op_tags import get_op_tag

    syms = get_symbol_table()
    sigma_star = get_sigma_star()

    # 1. Find all unique markers for the given paradigm.
    unique_markers = _get_all_markers_from_config(paradigm_name)

    # 2. Filter to those whose .stage matches requested stage
    active_markers = [m for m in unique_markers if hasattr(m, "stage") and m.stage == stage]

    # 3. Build triggered rule FSTs
    triggered_fsts = []
    active_tags = []
    for marker in active_markers:
        op_tag = get_op_tag(marker)
        active_tags.append(op_tag)

        T_trigger = get_marker_swapping_transducer(marker)
        triggered_fsts.append(T_trigger)

    # 4. Identity map for non-flagged strings
    if active_tags:
        # Build optimized identity map matching any string not ending in an active tag.
        # Since tags are only at the end of the word, a string lacks active tags iff
        # its final symbol is not in active_tags.
        all_syms = [syms.find(i) for i in range(1, syms.num_symbols())]
        active_tags_set = set(active_tags)
        non_tag_symbols = [s for s in all_syms if s not in active_tags_set]

        non_tag_fsa = pynini.union(
            *[pynini.accep(s, token_type=syms) for s in non_tag_symbols if s]
        )
        identity_map = pynini.union(
            pynini.accep("", token_type=syms),
            pynini.concat(sigma_star, non_tag_fsa)
        ).optimize()
    else:
        identity_map = sigma_star

    final_fst = pynini.union(identity_map, *triggered_fsts)
    return final_fst


get_stage_transducer = get_stage_realization_fst


@observed_cache([get_yaml_dir()])
def get_final_surface_filter_fst(paradigm_name: str) -> pynini.Fst:
    """
    Obtains the phone, boundary, and word_edge FSAs from get_special_fsas(),
    unions them, and applies .star.optimize() to create a strict filter acceptor.
    """
    special_fsas = get_special_fsas()
    phone_fsa = special_fsas["phone"]
    boundary_fsa = special_fsas["boundary"]
    word_edge_fsa = special_fsas["word_edge"]

    filter_fsa = pynini.union(phone_fsa, boundary_fsa, word_edge_fsa).star.optimize()
    return filter_fsa


get_final_surface_filter = get_final_surface_filter_fst


