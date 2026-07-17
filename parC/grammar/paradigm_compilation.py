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
import functools

import pynini
from loguru import logger
from pynini.lib import pynutil
from typing import Callable, NamedTuple
from frozendict import frozendict

BinaryFstOp = Callable[[pynini.Fst, pynini.Fst], pynini.Fst]

from parC.yaml_utils.cache import (
    is_fst_cache_valid,
    save_fst,
    load_fst,
    observed_cache,
    compute_cache_key,
    get_cached_fst,
    save_cached_fst,
)
from parC.fst_utils import ReservedSymbolMixin as R
from parC.fst_utils import stringify_features
from parC.constants import get_yaml_dir
from parC.lexicon import get_gloss_for_root, get_roots, get_roots_with_lexical_features
from parC.yaml_utils.schema_validation import CONFIG_KIND_TO_PARDIR
from parC.yaml_utils.yaml_server import (
    get_feature_map,
    get_yaml_kind,
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
    get_features_for_paradigm,
    get_markers_for_paradigm,
    get_fixed_features_for_paradigm,
    get_sorted_markers_for_paradigm,
)


from parC.grammar.transducer_compilation import get_marker_fst

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


def binary_fold(
    fsts: list[pynini.Fst],
    op: Callable[[pynini.Fst, pynini.Fst], pynini.Fst],
) -> pynini.Fst:
    if not fsts:
        return pynini.Fst()
    if len(fsts) == 1:
        return fsts[0]
    pivot = len(fsts) // 2
    return op(
        binary_fold(fsts[:pivot], op),
        binary_fold(fsts[pivot:], op),
    )


def intersect_and_optimize(a: pynini.Fst, b: pynini.Fst) -> pynini.Fst:
    return pynini.rmepsilon(pynini.connect(pynini.intersect(a, b)))


def binary_fold_intersect(
    fsts: list[pynini.Fst],
):
    return binary_fold(fsts, intersect_and_optimize)


def compose_and_optimize(a: pynini.Fst, b: pynini.Fst) -> pynini.Fst:
    return pynini.rmepsilon(pynini.connect(pynini.compose(a, b)))


def binary_fold_compose(
    fsts: list[pynini.Fst],
):
    return binary_fold(fsts, compose_and_optimize)


def union_and_optimize(a: pynini.Fst, b: pynini.Fst) -> pynini.Fst:
    return pynini.union(a, b).optimize()


def binary_fold_union(
    fsts: list[pynini.Fst],
):
    return binary_fold(fsts, union_and_optimize)


def _apply_feature_acceptor_constraints(
    root_fsa: pynini.Fst,
    feature_values: (
        list[tuple[str, str]] | set[tuple[str, str]] | tuple[tuple[str, str], ...]
    ),
) -> pynini.Fst:
    """
    Applies any feature-value acceptor constraints defined on the feature_values
    by intersecting the root_fsa (which contains [BOW]...[EOW]) with the feature acceptors.
    """
    from parC.grammar.acceptor_compilation import get_feature_acceptor_fsts

    feature_acceptors = get_feature_acceptor_fsts()
    if not feature_acceptors:
        return root_fsa

    constrained_fsa = root_fsa
    optimized_feature_acceptors = []
    for f, v in feature_values:
        key = f"{f}={v}"
        if key in feature_acceptors:
            acceptor_fst = feature_acceptors[key]
            # Wrap acceptor_fst with [BOW] and [EOW] to match root_fsa
            bow_fsa = pynini.accep(R.bow, token_type=get_symbol_table())
            eow_fsa = pynini.accep(R.eow, token_type=get_symbol_table())
            wrapped_acceptor = pynini.concat(
                bow_fsa, pynini.concat(acceptor_fst, eow_fsa)
            ).optimize()
            optimized_feature_acceptors.append(wrapped_acceptor)

    if len(optimized_feature_acceptors):
        all_acceptors = binary_fold_intersect(optimized_feature_acceptors)
        constrained_fsa = pynini.intersect(constrained_fsa, all_acceptors).optimize()

    return constrained_fsa


def _resolve_lexical_combos(
    paradigm_data: dict,
    feature_map: dict[str, list[str]],
    lexical_features: FeatureComboType | dict[str, str] | None = None,
    infer_lexical_features: bool = False,
) -> tuple[list[set[tuple[str, str]]], set[str], list[str]]:
    """
    Helper to resolve lexical combos, referenced lexical features, and sorted lexical feature names.
    """
    import itertools

    if infer_lexical_features:
        part_of_speech = paradigm_data["part_of_speech"]
        part_of_speech_data = get_yaml_data_safe(
            yaml_basename=part_of_speech, kind="PartOfSpeech"
        )
        lexical_feature_names = sorted(
            list(part_of_speech_data.get("lexical_features", []))
        )

        # Prune lexical features to only those referenced in this paradigm's rules/markers
        referenced_lexical_features = set()
        contingent_files = paradigm_data.get("contingent_markers", [])
        for contingent_file in contingent_files:
            contingent_data = get_yaml_data_safe(
                "ContingentFeatureMarkers", contingent_file
            )
            if contingent_data:
                for f in contingent_data.get("features", []):
                    if f in lexical_feature_names:
                        referenced_lexical_features.add(f)
        for f in paradigm_data.get("feature_markers", {}).keys():
            if f in lexical_feature_names:
                referenced_lexical_features.add(f)

        lexical_value_lists = []
        for fname in list(lexical_feature_names):
            if fname not in feature_map:
                logger.warning(
                    f"Lexical feature '{fname}' not in feature map — skipping."
                )
                if fname in referenced_lexical_features:
                    referenced_lexical_features.remove(fname)
                lexical_feature_names.remove(fname)
                continue
            if fname in referenced_lexical_features:
                options = [(fname, v) for v in feature_map[fname]]
                lexical_value_lists.append(options)

        if not lexical_value_lists:
            lexical_combos = [set()]
        else:
            lexical_combos = []
            for combo_tuples in itertools.product(*lexical_value_lists):
                lexical_combos.append(set(combo_tuples))
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

    return lexical_combos, referenced_lexical_features, lexical_feature_names


def _compose_stages_incrementally(
    cascade_domain: pynini.Fst,
    gated_fsts: list[pynini.Fst],
) -> pynini.Fst:
    import hashlib

    composed_fst = cascade_domain
    current_key = hashlib.sha256(cascade_domain.write_to_string()).hexdigest()

    for idx, stage_fst in enumerate(gated_fsts):
        stage_hash = hashlib.sha256(stage_fst.write_to_string()).hexdigest()
        next_key = hashlib.sha256(
            f"{current_key}_{stage_hash}".encode("utf-8")
        ).hexdigest()

        cached_composed = get_cached_fst(f"composition_{next_key}")
        if cached_composed is not None:
            composed_fst = cached_composed
            logger.debug(
                f"Incremental composition cache HIT for stage {idx} ({next_key})"
            )
        else:
            pre_states = composed_fst.num_states()
            pre_arcs = sum(
                composed_fst.num_arcs(state) for state in composed_fst.states()
            )
            composed_fst = pynini.compose(composed_fst, stage_fst).optimize()
            post_states = composed_fst.num_states()
            post_arcs = sum(
                composed_fst.num_arcs(state) for state in composed_fst.states()
            )
            logger.info(
                f"[PERF][COMPOSE STAGES][{idx+1}/{len(gated_fsts)}] states/arcs {pre_states}=>{post_states} - {pre_arcs}=>{post_arcs}"
            )
            save_cached_fst(f"composition_{next_key}", composed_fst)
            logger.debug(
                f"Incremental composition cache MISS for stage {idx}, composed and cached ({next_key})"
            )

        current_key = next_key

    return composed_fst


def _get_slot_fsas(ordered_features, tag_fsas, feature_map, exponed_in_stage):
    # Hoist slot_fsas using native Fst mutation to avoid heavy transducer unions
    syms = get_symbol_table()
    slot_fsas = {}

    # Compile stage-specific wildcards
    slot_fsas = {}
    for fname in ordered_features:
        if fname in exponed_in_stage:
            # Feature is exponed in this stage: other markers in this stage must require it to be absent
            slot_fsas[fname] = pynini.accep("", token_type=syms)
        else:
            # Feature is NOT exponed in this stage: other markers can match any value
            val_options = [tag_fsas[f"[{fname}={v}]"] for v in feature_map[fname]]
            slot_fsas[fname] = pynini.union(*val_options).optimize()

    return slot_fsas


def _compile_stage_cascade(
    paradigm_name: str,
    paradigm_data: dict,
    input_parts: list[pynini.Fst],
    tag_domain: pynini.Fst,
    marker_to_combinations: dict,
    tag_fsas: dict[str, pynini.Fst],
    ordered_features: list[str],
    cache_key: str = None,
) -> pynini.Fst:
    """
    Compiles the sequential stage-by-stage composition cascade using gated transducers.
    """
    if not input_parts:
        return pynini.Fst()

    cascade_domain = pynini.union(*input_parts).optimize()
    syms = get_symbol_table()
    sigma_star = get_sigma_star()

    gated_fsts = None
    if cache_key:
        gated_fsts = get_cached_fst(f"{cache_key}_stages")

    if gated_fsts is not None:
        logger.debug(f"Loaded cached stages for key {cache_key}")
    else:
        logger.info(
            f"[PERF][stage cascade] len marker_to_combinations, inner size: f{len(marker_to_combinations), sum(len(v) for v in marker_to_combinations.values())}"
        )

        logger.info(
            f"[PERF][stage cascade] cascade domain built - len input_parts {len(input_parts)}"
        )
        # Group markers by stage
        from collections import defaultdict

        stage_to_markers = defaultdict(list)
        for marker in marker_to_combinations.keys():
            stage = getattr(marker, "stage", None)
            if stage is None:
                stage = "unknown"
            stage_to_markers[stage].append(marker)

        logger.info(f"[PERF][stage cascade] grouped markers by stage")

        # Order stages
        stage_order = list(paradigm_data.get("stage_order", []))
        if "principal_part" not in stage_order:
            stage_order.insert(0, "principal_part")

        ordered_stages = []
        if "principal_part" in stage_to_markers:
            ordered_stages.append("principal_part")
        for s in stage_order:
            if s != "principal_part" and s in stage_to_markers:
                ordered_stages.append(s)
        for s in stage_to_markers:
            if s not in ordered_stages:
                ordered_stages.append(s)

        logger.info(f"[PERF][stage cascade] ordered stages")

        from parC.grammar.acceptor_compilation import get_special_fsas

        special_fsas = get_special_fsas()
        phone_fsa = special_fsas["phone"]
        boundary_fsa = special_fsas["boundary"]
        word_edge_fsa = special_fsas["word_edge"]
        flag_fsa = special_fsas.get("flag")
        # Include flag symbols (temp tags like [DIST], [WI], [NI]) in the
        # stems domain so that the trigger_fsa in later stages can still match
        # intermediate forms that already have dummy tokens inserted by
        # earlier stages (e.g. add_dist inserts [DIST] before insert_dist fires).
        if flag_fsa is not None:
            sigma_phones_and_boundaries = pynini.union(
                phone_fsa, boundary_fsa, word_edge_fsa, flag_fsa
            ).optimize()
        else:
            sigma_phones_and_boundaries = pynini.union(
                phone_fsa, boundary_fsa, word_edge_fsa
            ).optimize()
        stems_domain_acceptor = sigma_phones_and_boundaries.star.optimize()

        logger.info(f"[PERF][stage cascade] stem domain acceptor built")

        feature_map = get_feature_map()

        # Apply sequential composition cascade by stage
        gated_fsts = []
        for stage_no, stage in enumerate(ordered_stages):
            logger.info(
                f"[PERF][stage cascade] building stage '{stage}' {(stage_no+1)/len(ordered_stages)}"
            )
            markers = stage_to_markers[stage]

            # Identify features exponed in this stage
            exponed_in_stage = set()
            for marker in markers:
                for fs in marker_to_combinations[marker]:
                    for fname in dict(fs).keys():
                        exponed_in_stage.add(fname)

            slot_fsas = _get_slot_fsas(
                ordered_features=ordered_features,
                feature_map=feature_map,
                tag_fsas=tag_fsas,
                exponed_in_stage=exponed_in_stage,
            )

            def get_constraint_fsa(fs) -> pynini.Fst:
                if not fs:
                    return pynini.accep("", token_type=syms)
                fs_dict = dict(fs)
                parts = []
                for fname in ordered_features:
                    if fname in fs_dict:
                        val = fs_dict[fname]
                        parts.append(tag_fsas[f"[{fname}={val}]"])
                    else:
                        parts.append(slot_fsas[fname])
                seq = parts[0]
                for part in parts[1:]:
                    seq = pynini.concat(seq, part)
                return seq

            trigger_paths = []

            # Check if there is a global marker in this stage
            has_global = any(
                not fs for marker in markers for fs in marker_to_combinations[marker]
            )

            stage_constraints = []
            for marker in markers:
                combo_tag_lists = marker_to_combinations[marker]

                marker_constraints = []
                for fs in combo_tag_lists:
                    constraint_fsa = get_constraint_fsa(fs)
                    marker_constraints.append(constraint_fsa)
                    if not has_global:
                        stage_constraints.append(constraint_fsa)

                if marker_constraints:
                    marker_constraint_union = pynini.union(
                        *marker_constraints
                    ).optimize()
                    marker_trigger_tags = pynini.intersect(
                        tag_domain, marker_constraint_union
                    ).optimize()
                    marker_trigger_fsa = pynini.concat(
                        stems_domain_acceptor, marker_trigger_tags
                    ).optimize()
                else:
                    marker_trigger_fsa = sigma_star

                base_fst = get_marker_fst(marker)
                if getattr(marker, "kind", None) == "string_map":
                    tags_list = list(tag_fsas.values())
                    if tags_list:
                        tags_star = pynini.union(*tags_list).star.optimize()
                        base_fst = pynini.concat(base_fst, tags_star).optimize()
                trigger_paths.append(pynini.compose(marker_trigger_fsa, base_fst))

            if not has_global and stage_constraints:
                stage_constraint_union = pynini.union(*stage_constraints).optimize()
                non_trigger_tags = pynini.difference(
                    tag_domain, stage_constraint_union
                ).optimize()
                non_trigger_fsa = pynini.concat(
                    stems_domain_acceptor, non_trigger_tags
                ).optimize()
                gated_fst = pynini.union(
                    *(trigger_paths + [non_trigger_fsa])
                ).optimize()
            else:
                gated_fst = pynini.union(*trigger_paths).optimize()

            gated_fsts.append(gated_fst)
        logger.info(f"[PERF][stage cascade] stages built")

        if cache_key:
            save_cached_fst(f"{cache_key}_stages", gated_fsts)

    composed_fst = _compose_stages_incrementally(cascade_domain, gated_fsts)

    logger.info(f"[PERF][stage cascade] stages composed left-to-right")

    return composed_fst


@functools.lru_cache(maxsize=128)
def _get_active_combos_for_paradigm(
    paradigm_name: str,
    lexical_combos: frozenset[frozenset[tuple[str, str]]],
) -> list[frozenset[tuple[str, str]]]:
    """
    Identify active inflectional feature combos that are referenced/mapped in the markers
    by checking if they can be successfully exponed/resolved, computed directly from
    marker definitions.
    """
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    precomputed = get_sorted_markers_for_paradigm(paradigm_name)

    fixed_features = frozenset(
        get_fixed_features_for_paradigm(
            name=paradigm_name, kind="Paradigm", paradigm_data=paradigm_data
        )
    )

    regular_by_feature = precomputed["regular_by_feature"]
    regular_features_supported = {
        feat: {val for val, markers in val_dict.items() if markers}
        for feat, val_dict in regular_by_feature.items()
    }

    contingent_by_file = precomputed["contingent_by_file"]
    contingent_features = precomputed["contingent_features"]

    contingent_specs = []
    for cf, leaves in contingent_by_file.items():
        reqs_list = [reqs for reqs, markers in leaves if markers]
        if reqs_list:
            contingent_specs.append((reqs_list, contingent_features[cf]))

    # Identify all inflectional features (excluding lexical ones) that can have active markers.
    part_of_speech = paradigm_data["part_of_speech"]
    part_of_speech_data = get_yaml_data_safe(
        yaml_basename=part_of_speech, kind="PartOfSpeech"
    )
    free_features = list(part_of_speech_data.get("features", []))
    for feat_name, ref in paradigm_data.get("feature_markers", {}).items():
        if ref is not None and not (isinstance(ref, str) and ref.startswith("$")):
            if feat_name in free_features:
                free_features.remove(feat_name)

    # Build potential combinations from active regular features and contingent spec paths
    feature_map = get_feature_map()
    import itertools

    # Each regular feature with markers defines its possible values.
    # If a feature has regular markers, we only use values that are actually supported.
    # If a feature has no regular markers, it might only be exponed contingently.
    # For contingent-only features, we can restrict their values to those actually defined
    # in the contingent specs to avoid multiplying by values that have no rules.
    contingent_active_values = {}
    for reqs_list, cf_features in contingent_specs:
        for reqs in reqs_list:
            for feat, val in reqs:
                if feat not in contingent_active_values:
                    contingent_active_values[feat] = set()
                contingent_active_values[feat].add(val)

    feature_branches = {}
    for feat in free_features:
        if feat in regular_features_supported:
            feature_branches[feat] = list(regular_features_supported[feat])
        elif feat in contingent_active_values:
            feature_branches[feat] = list(contingent_active_values[feat])
        else:
            # Unexponed feature. We can draw values from the feature map.
            feature_branches[feat] = list(feature_map.get(feat, []))

    # Compute cartesian product over only these active subsets of feature values (much smaller than full space)
    keys = list(feature_branches.keys())
    value_tuples = list(itertools.product(*(feature_branches[k] for k in keys)))

    raw_combos = []
    for vt in value_tuples:
        raw_combos.append(frozenset(zip(keys, vt)))

    logger.debug(
        f"[_get_active_combos_for_paradigm] Paradigm '{paradigm_name}': "
        f"Materialized active-branches subset of size {len(raw_combos)} "
        f"(features: { {k: len(v) for k, v in feature_branches.items()} })"
    )

    active = []
    lex_list = [dict(lc) for lc in lexical_combos] if lexical_combos else [{}]

    # Convert contingent reqs lists to dicts for O(1) checks
    contingent_specs_dicts = []
    for reqs_list, cf_features in contingent_specs:
        reqs_dicts = [dict(r) for r in reqs_list]
        contingent_specs_dicts.append((reqs_dicts, cf_features))

    for combo in raw_combos:
        # Re-attach fixed features
        full_combo = combo | fixed_features
        inflect_combo = full_combo - fixed_features
        inflect_dict = dict(inflect_combo)
        target_features = set(inflect_dict.keys())

        resolved = False
        for lex_dict in lex_list:
            # Union of two dicts is extremely fast
            feat_vals = {**inflect_dict, **lex_dict}

            unexponed = target_features.copy()
            for reqs_dicts, cf_features in contingent_specs_dicts:
                matched = False
                for reqs in reqs_dicts:
                    # check if reqs is subset of feat_vals
                    # O(1) membership check for each element instead of set intersection/subset
                    if all(feat_vals.get(k) == v for k, v in reqs.items()):
                        matched = True
                        break
                if matched:
                    unexponed -= cf_features

            still_unexponed = False
            for f in unexponed:
                v = inflect_dict.get(f)
                if v is None or v not in regular_features_supported.get(f, set()):
                    still_unexponed = True
                    break

            if not still_unexponed:
                resolved = True
                break

        if resolved:
            active.append(full_combo)

    logger.debug(
        f"[_get_active_combos_for_paradigm] Paradigm '{paradigm_name}': "
        f"Filtered down to {len(active)} active combos"
    )

    return active


def _compile_inflect_graph_shared(
    paradigm_name: str,
    paradigm_data: dict,
    feature_map: dict,
    roots_and_lexical_combos: list[tuple[pynini.Fst, FeatureComboType]],
    infer_lexical_features: bool,
    lexical_feature_names: list[str],
    referenced_lexical_features: set[str],
    lex_combos_set_for_active: frozenset,
    cache_key: str = None,
    is_open_root: bool = False,
    base_cache_key: str = None,
    non_deterministic_cleanup: bool = False,
) -> pynini.Fst:
    from collections import defaultdict

    marker_to_combinations = defaultdict(list)
    input_parts = []
    tag_seqs = []

    # Pre-compile tag acceptors to avoid calling fsa(...) repeatedly.
    tag_fsas = {}
    for fname, fvals in feature_map.items():
        for val in fvals:
            tag_str = f"[{fname}={val}]"
            tag_fsas[tag_str] = pynini.accep(tag_str, token_type=get_symbol_table())

    def get_feature_fsa(
        feat_vals: FeatureComboType | dict[str, str],
    ) -> pynini.Fst:
        if isinstance(feat_vals, (dict, frozendict)):
            feat_vals = list(feat_vals.items())
        sorted_feats = sorted(feat_vals)
        if not sorted_feats:
            return pynini.accep("", token_type=get_symbol_table())
        parts = [tag_fsas[f"[{f}={v}]"] for f, v in sorted_feats]
        curr = parts[0]
        for part in parts[1:]:
            curr = pynini.concat(curr, part)
        return curr

    active_combos = _get_active_combos_for_paradigm(
        paradigm_name,
        lex_combos_set_for_active,
    )

    # 1. Populate marker_to_combinations statically from precomputed markers
    from parC.grammar.marker_resolution import get_sorted_markers_for_paradigm

    precomputed = get_sorted_markers_for_paradigm(paradigm_name)
    for pm, fs in precomputed["all_markers_sorted"]:
        fs_frozen = (
            frozenset(fs.items())
            if isinstance(fs, dict)
            else (frozenset(fs) if fs != "global" and fs != "unknown" else frozenset())
        )
        if fs_frozen not in marker_to_combinations[pm]:
            marker_to_combinations[pm].append(fs_frozen)

    # 2. Build the tag domain / sequences for inflectional features
    inflectional_fsas = []
    for feature_values in active_combos:
        inflectional_fsa = get_feature_fsa(feature_values)
        inflectional_fsas.append(inflectional_fsa)

    if inflectional_fsas:
        inflectional_union = pynini.union(*inflectional_fsas).optimize()
    else:
        inflectional_union = None

    # 3. For each root, build its base + lexical tags
    root_fsas = []
    for root_fsa, lexical_combo in roots_and_lexical_combos:
        # Constrain root by its lexical combo
        constrained_root_fsa = _apply_feature_acceptor_constraints(
            root_fsa, lexical_combo
        )
        if constrained_root_fsa.num_states() == 0:
            continue

        lexical_parts = []
        if infer_lexical_features:
            for fname in lexical_feature_names:
                if fname in referenced_lexical_features:
                    val_opt = next((v for f, v in lexical_combo if f == fname), None)
                    if val_opt is not None and val_opt != "":
                        lexical_parts.append(tag_fsas[f"[{fname}={val_opt}]"])
                else:
                    lexical_parts.append(
                        pynini.union(
                            *[tag_fsas[f"[{fname}={v}]"] for v in feature_map[fname]]
                        )
                    )
        if lexical_parts:
            lexical_fsa = lexical_parts[0]
            for part in lexical_parts[1:]:
                lexical_fsa = pynini.concat(lexical_fsa, part)
        else:
            lexical_fsa = None

        if lexical_fsa:
            root_with_lex = pynini.concat(constrained_root_fsa, lexical_fsa)
        else:
            root_with_lex = constrained_root_fsa

        root_fsas.append(root_with_lex)

    if not root_fsas:
        return pynini.Fst()

    roots_union = pynini.union(*root_fsas).optimize()
    # 4. Construct tag_domain and input cascade domain
    # Deduplicate lexical combinations to avoid constructing redundant tag sequences
    unique_lexical_combos = []
    seen_combos = set()
    for _, lexical_combo in roots_and_lexical_combos:
        combo_tuple = tuple(sorted(lexical_combo))
        if combo_tuple not in seen_combos:
            seen_combos.add(combo_tuple)
            unique_lexical_combos.append(lexical_combo)

    # Build lexical_tags_union
    lexical_fsas_for_domain = []
    for lexical_combo in unique_lexical_combos:
        lexical_parts = []
        if infer_lexical_features:
            for fname in lexical_feature_names:
                if fname in referenced_lexical_features:
                    val_opt = next((v for f, v in lexical_combo if f == fname), None)
                    if val_opt is not None and val_opt != "":
                        tag_fsa = tag_fsas[f"[{fname}={val_opt}]"]
                        lexical_parts.append(tag_fsa)
                else:
                    lexical_parts.append(
                        pynini.union(
                            *[tag_fsas[f"[{fname}={v}]"] for v in feature_map[fname]]
                        )
                    )
        if lexical_parts:
            lexical_fsa = lexical_parts[0]
            for part in lexical_parts[1:]:
                lexical_fsa = pynini.concat(lexical_fsa, part)
            lexical_fsas_for_domain.append(lexical_fsa)
        else:
            lexical_fsas_for_domain.append(
                pynini.accep("", token_type=get_symbol_table())
            )

    if lexical_fsas_for_domain:
        lexical_tags_union = pynini.union(*lexical_fsas_for_domain).optimize()
    else:
        lexical_tags_union = None

    if lexical_tags_union and inflectional_union:
        tag_domain = pynini.concat(lexical_tags_union, inflectional_union).optimize()
    elif lexical_tags_union:
        tag_domain = lexical_tags_union
    elif inflectional_union:
        tag_domain = inflectional_union
    else:
        tag_domain = pynini.accep("", token_type=get_symbol_table())

    cascade_domain = None
    if cache_key and is_open_root:
        cascade_domain = get_cached_fst(f"{cache_key}_open_input_acceptor")

    if cascade_domain is None:
        if inflectional_union:
            cascade_domain = pynini.concat(roots_union, inflectional_union).optimize()
        else:
            cascade_domain = roots_union
        if cache_key and is_open_root:
            save_cached_fst(f"{cache_key}_open_input_acceptor", cascade_domain)

    input_parts = [cascade_domain]

    ordered_features = []
    if infer_lexical_features:
        ordered_features.extend(lexical_feature_names)
    ordered_features.extend(
        sorted(list(set().union(*[set(dict(c).keys()) for c in active_combos])))
    )

    res = _compile_stage_cascade(
        paradigm_name=paradigm_name,
        paradigm_data=paradigm_data,
        input_parts=input_parts,
        tag_domain=tag_domain,
        marker_to_combinations=marker_to_combinations,
        tag_fsas=tag_fsas,
        ordered_features=ordered_features,
        cache_key=(
            (f"{base_cache_key}_infer" if infer_lexical_features else base_cache_key)
            if base_cache_key
            else cache_key
        ),
    )

    # Cleanup Tags
    sigma_star = get_sigma_star()
    if non_deterministic_cleanup:
        delete_tags = pynini.union(
            *[pynutil.delete(tag_fsas[t]) | tag_fsas[t] for t in tag_fsas]
        )
    else:
        delete_tags = pynini.union(*[pynutil.delete(tag_fsas[t]) for t in tag_fsas])

    cleanup_fst = pynini.cdrewrite(delete_tags, "", "", sigma_star).optimize()

    res = pynini.compose(res, cleanup_fst).optimize()

    logger.info(f"[PERF][cleanup] cleanup filters applied")

    if cache_key and is_open_root:
        save_cached_fst(f"{cache_key}_open_inflect", res)

    return res


def build_inflect_graph(paradigm_name: str) -> pynini.Fst:
    """root[features...] → surface form."""
    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    feature_map = get_feature_map()
    roots = get_roots_for_paradigm(paradigm_name=paradigm_name)
    from parC.lexicon import get_features_for_root

    part_of_speech = paradigm_data["part_of_speech"]

    roots_and_lexical_combos = []
    lexical_combos_list = []
    for root in roots:
        lex_set = frozenset(get_features_for_root(part_of_speech, root))
        roots_and_lexical_combos.append((word_fsa(root), lex_set))
        lexical_combos_list.append(lex_set)

    base_cache_key = get_paradigm_cache_key(paradigm_name)

    return _compile_inflect_graph_shared(
        paradigm_name=paradigm_name,
        paradigm_data=paradigm_data,
        feature_map=feature_map,
        roots_and_lexical_combos=roots_and_lexical_combos,
        infer_lexical_features=False,
        lexical_feature_names=[],
        referenced_lexical_features=set(),
        lex_combos_set_for_active=frozenset(lexical_combos_list),
        base_cache_key=base_cache_key,
    )


def build_inflect_graph_for_root_regex(
    paradigm_name: str,
    root_regex: str | pynini.Fst,
    lexical_features: FeatureComboType | dict[str, str] | None = None,
    infer_lexical_features: bool = False,
    cache_key: str = None,
    non_deterministic_cleanup: bool = False,
) -> pynini.Fst:
    """root_regex[lexical_features][inflectional_features] → surface form."""
    is_open_root = root_regex == "<Phone>*"
    base_cache_key = None
    if is_open_root:
        base_cache_key = get_paradigm_cache_key(paradigm_name)
        if cache_key is None:
            cache_key = base_cache_key
        # Append settings suffix to cache_key
        settings = []
        if infer_lexical_features:
            settings.append("infer")
        if non_deterministic_cleanup:
            settings.append("nd_cleanup")
        if lexical_features:
            import hashlib

            if isinstance(lexical_features, (dict, frozendict)):
                feats_tuple = tuple(sorted(lexical_features.items()))
            else:
                feats_tuple = tuple(sorted(lexical_features))
            feats_str = ",".join(
                f"{f}={v}" if isinstance(v, str) else f"{f[0]}={f[1]}"
                for f in feats_tuple
            )
            settings.append(f"lex_{hashlib.sha256(feats_str.encode()).hexdigest()[:8]}")

        if settings:
            cache_key = f"{cache_key}_{'_'.join(settings)}"

    if cache_key and is_open_root:
        open_inflect = get_cached_fst(f"{cache_key}_open_inflect")
        if open_inflect is not None:
            return open_inflect

    if isinstance(root_regex, str):
        root_fsa = fsa(R.bow + root_regex + R.eow)
    else:
        root_fsa = root_regex

    paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
    if paradigm_data is None:
        raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

    feature_map = get_feature_map()
    lexical_combos, referenced_lexical_features, lexical_feature_names = (
        _resolve_lexical_combos(
            paradigm_data=paradigm_data,
            feature_map=feature_map,
            lexical_features=lexical_features,
            infer_lexical_features=infer_lexical_features,
        )
    )

    roots_and_lexical_combos = [(root_fsa, combo) for combo in lexical_combos]

    if infer_lexical_features:
        lex_combos_set = frozenset(frozenset(c) for c in lexical_combos)
    else:
        if lexical_features:
            if isinstance(lexical_features, (dict, frozendict)):
                lex_set = frozenset(lexical_features.items())
            else:
                lex_set = frozenset(lexical_features)
            lex_combos_set = frozenset([lex_set])
        else:
            lex_combos_set = frozenset([frozenset()])

    return _compile_inflect_graph_shared(
        paradigm_name=paradigm_name,
        paradigm_data=paradigm_data,
        feature_map=feature_map,
        roots_and_lexical_combos=roots_and_lexical_combos,
        infer_lexical_features=infer_lexical_features,
        lexical_feature_names=lexical_feature_names,
        referenced_lexical_features=referenced_lexical_features,
        lex_combos_set_for_active=lex_combos_set,
        cache_key=cache_key,
        is_open_root=is_open_root,
        base_cache_key=base_cache_key,
        non_deterministic_cleanup=non_deterministic_cleanup,
    )


def get_open_inflect_graph(
    paradigm_name: str,
    non_deterministic_cleanup: bool = False,
    infer_lexical_features: bool = False,
    cache_key: str = None,
):
    if not cache_key:
        cache_key = get_paradigm_cache_key(paradigm_name)
    suffix_parts = []
    if infer_lexical_features:
        suffix_parts.append("infer")
    if non_deterministic_cleanup:
        suffix_parts.append("nd_cleanup")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    open_inflect_key = f"{cache_key}_open_inflect{suffix}"

    open_inflect = get_cached_fst(open_inflect_key)
    if open_inflect is None:
        open_inflect = build_inflect_graph_for_root_regex(
            paradigm_name,
            "<Phone>*",
            cache_key=cache_key,
            non_deterministic_cleanup=non_deterministic_cleanup,
            infer_lexical_features=infer_lexical_features,
        )

    save_cached_fst(open_inflect_key, open_inflect)

    return open_inflect


def get_open_parse_graph(
    paradigm_name: str,
    non_deterministic_cleanup: bool = False,
    infer_lexical_features: bool = False,
) -> pynini.Fst:
    """
    Returns an open parse graph for the given paradigm by building the inflect graph
    with '<Phone>*' as the root regex and inverting it.
    Uses yaml_dir/config dependency aware caching for the open parse graph and its subcomponents.
    """
    cache_key = get_paradigm_cache_key(paradigm_name)
    suffix_parts = []
    if infer_lexical_features:
        suffix_parts.append("infer")
    if non_deterministic_cleanup:
        suffix_parts.append("nd_cleanup")
    suffix = f"_{'_'.join(suffix_parts)}" if suffix_parts else ""
    open_parse_key = f"{cache_key}_open_parse{suffix}"

    open_parse = get_cached_fst(open_parse_key)
    if open_parse is not None:
        logger.debug(f"Open parse graph cache HIT for paradigm {paradigm_name}")
        return open_parse

    logger.debug(f"Open parse graph cache MISS for paradigm {paradigm_name}")

    open_inflect = get_open_inflect_graph(
        paradigm_name,
        non_deterministic_cleanup,
        infer_lexical_features,
        cache_key=cache_key,
    )

    open_parse = build_parse_graph(open_inflect)
    save_cached_fst(open_parse_key, open_parse)
    return open_parse


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
    seen_markers = set()
    try:
        from parC.grammar.transducer_compilation import get_marker_fst_key
        from parC.grammar.marker_resolution import get_sorted_markers_for_paradigm

        precomputed = get_sorted_markers_for_paradigm(paradigm_name)
        for marker, _ in precomputed["all_markers_sorted"]:
            seen_markers.add(marker)

        for marker in seen_markers:
            m_key = get_marker_fst_key(marker)
            child_keys[f"Marker/{m_key}"] = m_key
    except Exception as e:
        logger.warning(
            f"Error resolving marker dependencies for paradigm {paradigm_name}: {e}"
        )

    description = f"Paradigm '{paradigm_name}' ({len(seen_markers)} markers resolved)"
    return compute_cache_key(
        paradigm_name, "Paradigm", config_dirs, child_keys, description=description
    )


def _load_paradigm_cached(
    cache_key: str,
) -> tuple[pynini.Fst, pynini.Fst, pynini.Fst, pynini.Fst] | None:
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
            logger.debug(
                f"Paradigm cache MISS for key {cache_key}: failed to read {k}.fst: {e}"
            )
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
    search_left_factor.write(
        os.path.join(CACHE_DIR, f"{cache_key}_search_left_factor.fst")
    )
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


def get_search_graphs(paradigm_name: str) -> tuple[pynini.Fst, pynini.Fst]:
    return (
        _get_or_build(paradigm_name, "search_lexicon"),
        _get_or_build(paradigm_name, "search_left_factor"),
    )


"""
Public API
"""


@observed_cache([get_yaml_dir()])
def parse(form: str, kind: str = "Paradigm", name: str = "") -> list[dict]:
    form_fsa = word_fsa(form)
    parse_graph = get_parse_graph(name)
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
    kind: str, name: str, form: str, nshortest: int, do_parse: bool = True
) -> list[tuple[str, float]] | list[dict]:
    search_lexicon, left_factor = get_search_graphs(name)
    form_fsa = word_fsa(form)
    left_factor_lattice = pynini.compose(form_fsa, left_factor).optimize()
    edit_graph = pynini.compose(left_factor_lattice, search_lexicon)
    hits = fsm_strings_and_weights(edit_graph, strip_all_tags=True, nshortest=nshortest)

    if do_parse:
        parses = []
        for hit, weight in hits:
            current_parse = [item.copy() for item in parse(hit, kind=kind, name=name)]
            [parse_item.update(edit_distance=weight) for parse_item in current_parse]
            [parse_item.update(form=hit) for parse_item in current_parse]
            parses.extend(current_parse)
        return parses

    return hits
