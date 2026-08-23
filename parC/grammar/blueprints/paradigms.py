"""Layer 4: StageCascadeBlueprint (Morphotactic Stage Cascade & Paradigm Inflection).

Wraps stage-gated transducer compilation, tag domain construction, and open paradigm inflection
while taking explicit AlphabetBlueprint, MarkerLibraryBlueprint, and RulePipelineBlueprint dependencies.
"""

from __future__ import annotations

import pynini

from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.blueprints.transducers import (
    MarkerLibraryBlueprint,
    RulePipelineBlueprint,
)
from parC.grammar.paradigm_compilation import (
    _compile_stage_cascade,
    _get_active_combos_for_paradigm,
    get_open_inflect_graph,
)
from parC.grammar.marker_resolution import get_sorted_markers_for_paradigm
from parC.yaml_utils.yaml_server import get_feature_map, get_yaml_data_safe


class StageCascadeBlueprint:
    """Blueprint for Layer 4 Morphotactic Stage Cascade & Paradigm Inflection.

    Stores paradigm configuration models explicitly and holds explicit lower-layer blueprints
    as dependencies.
    """

    def __init__(
        self,
        paradigm_name: str,
        paradigm_data: dict,
        feature_map: dict[str, list[str]],
        alphabet: AlphabetBlueprint,
        patterns: PatternLibraryBlueprint,
        rules: RulePipelineBlueprint,
        markers: MarkerLibraryBlueprint,
    ) -> None:
        self.paradigm_name = paradigm_name
        self.paradigm_data = paradigm_data
        self.feature_map = feature_map
        self.alphabet = alphabet
        self.patterns = patterns
        self.rules = rules
        self.markers = markers

    @classmethod
    def from_paradigm(
        cls,
        paradigm_name: str,
        alphabet: AlphabetBlueprint | None = None,
        patterns: PatternLibraryBlueprint | None = None,
        rules: RulePipelineBlueprint | None = None,
        markers: MarkerLibraryBlueprint | None = None,
    ) -> StageCascadeBlueprint:
        """Factory method to construct StageCascadeBlueprint for a named paradigm."""
        paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
        if paradigm_data is None:
            raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")
        feature_map = get_feature_map()

        alph = alphabet or AlphabetBlueprint.from_config()
        pats = patterns or PatternLibraryBlueprint.from_config()
        ruls = rules or RulePipelineBlueprint.from_config()
        mars = markers or MarkerLibraryBlueprint.from_config()

        return cls(
            paradigm_name=paradigm_name,
            paradigm_data=paradigm_data,
            feature_map=feature_map,
            alphabet=alph,
            patterns=pats,
            rules=ruls,
            markers=mars,
        )

    def get_tag_domain_acceptor(
        self,
        lexical_combos: frozenset | None = None,
    ) -> pynini.Fst:
        """Constructs tag domain acceptor FSA for active feature combinations in a paradigm."""
        lex_set = (
            lexical_combos if lexical_combos is not None else frozenset([frozenset()])
        )
        active_combos = _get_active_combos_for_paradigm(self.paradigm_name, lex_set)
        syms = self.alphabet.get_symbol_table()

        tag_fsas = {}
        for fname, fvals in self.feature_map.items():
            for val in fvals:
                tag_str = f"[{fname}={val}]"
                tag_fsas[tag_str] = pynini.accep(tag_str, token_type=syms)

        inflectional_fsas = []
        for combo in active_combos:
            sorted_feats = sorted(combo)
            if not sorted_feats:
                inflectional_fsas.append(pynini.accep("", token_type=syms))
            else:
                parts = [tag_fsas[f"[{f}={v}]"] for f, v in sorted_feats]
                curr = parts[0]
                for part in parts[1:]:
                    curr = pynini.concat(curr, part)
                inflectional_fsas.append(curr)

        if inflectional_fsas:
            return pynini.union(*inflectional_fsas).optimize()
        return pynini.accep("", token_type=syms)

    def get_stage_gated_transducers(
        self,
        input_parts: list[pynini.Fst] | None = None,
    ) -> list[pynini.Fst]:
        """Builds stage-gated realization transducers for the given paradigm."""
        syms = self.alphabet.get_symbol_table()

        tag_fsas = {}
        for fname, fvals in self.feature_map.items():
            for val in fvals:
                tag_str = f"[{fname}={val}]"
                tag_fsas[tag_str] = pynini.accep(tag_str, token_type=syms)

        from collections import defaultdict

        marker_to_combinations = defaultdict(list)
        precomputed = get_sorted_markers_for_paradigm(self.paradigm_name)
        for pm, fs in precomputed["all_markers_sorted"]:
            fs_frozen = (
                frozenset(fs.items())
                if isinstance(fs, dict)
                else (
                    frozenset(fs)
                    if fs != "global" and fs != "unknown"
                    else frozenset()
                )
            )
            if fs_frozen not in marker_to_combinations[pm]:
                marker_to_combinations[pm].append(fs_frozen)

        tag_domain = self.get_tag_domain_acceptor()

        if input_parts is None:
            sigma_star = self.alphabet.get_sigma_star()
            cascade_domain = pynini.concat(sigma_star, tag_domain).optimize()
            input_parts = [cascade_domain]

        ordered_features = sorted(list(self.feature_map.keys()))

        cache_key = f"blueprint_{self.paradigm_name}"
        from parC.yaml_utils.cache import get_cached_fst

        _compile_stage_cascade(
            paradigm_name=self.paradigm_name,
            paradigm_data=self.paradigm_data,
            input_parts=input_parts,
            tag_domain=tag_domain,
            marker_to_combinations=marker_to_combinations,
            tag_fsas=tag_fsas,
            ordered_features=ordered_features,
            cache_key=cache_key,
        )
        stages = get_cached_fst(f"{cache_key}_stages")
        if stages is not None:
            return stages
        return []

    def build_open_inflect_graph(
        self,
        root_regex: str | pynini.Fst = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
        cache_key: str | None = None,
    ) -> pynini.Fst:
        """Builds open inflection graph FST for a paradigm using explicit blueprint data."""
        from frozendict import frozendict

        is_open_root = root_regex == "<Phone>*"
        base_cache_key = None
        if is_open_root:
            from parC.grammar.paradigm_compilation import get_paradigm_cache_key

            base_cache_key = get_paradigm_cache_key(self.paradigm_name)
            if cache_key is None:
                cache_key = base_cache_key
            settings = []
            if infer_lexical_features:
                settings.append("infer")
            if non_deterministic_cleanup:
                settings.append("nd_cleanup")
            if lexical_features and not infer_lexical_features:
                import hashlib

                if isinstance(lexical_features, (dict, frozendict)):
                    feats_tuple = tuple(sorted(lexical_features.items()))
                else:
                    feats_tuple = tuple(sorted(lexical_features))
                feats_str = ",".join(f"{f}={v}" for f, v in feats_tuple)
                settings.append(
                    f"lex_{hashlib.sha256(feats_str.encode()).hexdigest()[:8]}"
                )

            if settings:
                cache_key = f"{cache_key}_{'_'.join(settings)}"

        from parC.grammar.paradigm_compilation import _in_memory_fst_cache
        from parC.yaml_utils.cache import get_cached_fst

        if cache_key and is_open_root:
            mem_key = f"{cache_key}_open_inflect"
            if mem_key in _in_memory_fst_cache:
                return _in_memory_fst_cache[mem_key]
            open_inflect = get_cached_fst(mem_key)
            if open_inflect is not None:
                _in_memory_fst_cache[mem_key] = open_inflect
                return open_inflect

        special_fsas = self.alphabet.get_special_fsas()
        if is_open_root:
            open_root_template = self.paradigm_data.get("open_root_template")
            if open_root_template:
                open_root_fsa = self.patterns.compile_pattern_string(
                    open_root_template, alphabet=self.alphabet
                )
            else:
                phone_fsa = special_fsas["phone"]
                user_tag_fsa = special_fsas["user_tag"]
                open_root_fsa = pynini.union(phone_fsa, user_tag_fsa).star.optimize()

            bow_fsa = special_fsas["bow"]
            eow_fsa = special_fsas["eow"]
            root_fsa = pynini.concat(
                bow_fsa, pynini.concat(open_root_fsa, eow_fsa)
            ).optimize()
        elif isinstance(root_regex, str):
            compiled = self.patterns.compile_pattern_string(
                root_regex, alphabet=self.alphabet
            )
            bow_fsa = special_fsas["bow"]
            eow_fsa = special_fsas["eow"]
            root_fsa = pynini.concat(
                bow_fsa, pynini.concat(compiled, eow_fsa)
            ).optimize()
        else:
            root_fsa = root_regex

        from parC.grammar.paradigm_compilation import (
            _compile_inflect_graph_shared,
            _resolve_lexical_combos,
        )

        lexical_combos, referenced_lexical_features, lexical_feature_names = (
            _resolve_lexical_combos(
                paradigm_data=self.paradigm_data,
                feature_map=self.feature_map,
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

        res = _compile_inflect_graph_shared(
            paradigm_name=self.paradigm_name,
            paradigm_data=self.paradigm_data,
            feature_map=self.feature_map,
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
        if cache_key and is_open_root:
            _in_memory_fst_cache[f"{cache_key}_open_inflect"] = res
        return res


def stagecascadeblueprint_from_paradigm(
    paradigm_name: str,
    alphabet: AlphabetBlueprint | None = None,
    patterns: PatternLibraryBlueprint | None = None,
    rules: RulePipelineBlueprint | None = None,
    markers: MarkerLibraryBlueprint | None = None,
) -> StageCascadeBlueprint:
    """Standalone factory function to construct StageCascadeBlueprint from paradigm_name."""
    return StageCascadeBlueprint.from_paradigm(
        paradigm_name=paradigm_name,
        alphabet=alphabet,
        patterns=patterns,
        rules=rules,
        markers=markers,
    )


