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

    Wraps stage-gated transducer generation and incremental stage composition while taking
    explicit lower-layer blueprints as dependencies.
    """

    def __init__(
        self,
        alphabet: AlphabetBlueprint | None = None,
        patterns: PatternLibraryBlueprint | None = None,
        rules: RulePipelineBlueprint | None = None,
        markers: MarkerLibraryBlueprint | None = None,
    ) -> None:
        self.alphabet = alphabet or AlphabetBlueprint()
        self.patterns = patterns or PatternLibraryBlueprint(alphabet=self.alphabet)
        self.rules = rules or RulePipelineBlueprint(
            alphabet=self.alphabet, patterns=self.patterns
        )
        self.markers = markers or MarkerLibraryBlueprint(
            alphabet=self.alphabet, patterns=self.patterns
        )

    def get_tag_domain_acceptor(
        self,
        paradigm_name: str,
        lexical_combos: frozenset | None = None,
    ) -> pynini.Fst:
        """Constructs tag domain acceptor FSA for active feature combinations in a paradigm."""
        lex_set = lexical_combos if lexical_combos is not None else frozenset([frozenset()])
        active_combos = _get_active_combos_for_paradigm(paradigm_name, lex_set)
        feature_map = get_feature_map()
        syms = self.alphabet.get_symbol_table()

        tag_fsas = {}
        for fname, fvals in feature_map.items():
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
        paradigm_name: str,
        input_parts: list[pynini.Fst] | None = None,
    ) -> list[pynini.Fst]:
        """Builds stage-gated realization transducers for the given paradigm."""
        paradigm_data = get_yaml_data_safe("Paradigm", paradigm_name)
        if paradigm_data is None:
            raise ValueError(f"Paradigm '{paradigm_name}' not found or invalid.")

        feature_map = get_feature_map()
        syms = self.alphabet.get_symbol_table()

        tag_fsas = {}
        for fname, fvals in feature_map.items():
            for val in fvals:
                tag_str = f"[{fname}={val}]"
                tag_fsas[tag_str] = pynini.accep(tag_str, token_type=syms)

        from collections import defaultdict
        marker_to_combinations = defaultdict(list)
        precomputed = get_sorted_markers_for_paradigm(paradigm_name)
        for pm, fs in precomputed["all_markers_sorted"]:
            fs_frozen = (
                frozenset(fs.items())
                if isinstance(fs, dict)
                else (frozenset(fs) if fs != "global" and fs != "unknown" else frozenset())
            )
            if fs_frozen not in marker_to_combinations[pm]:
                marker_to_combinations[pm].append(fs_frozen)

        tag_domain = self.get_tag_domain_acceptor(paradigm_name)

        if input_parts is None:
            sigma_star = self.alphabet.get_sigma_star()
            cascade_domain = pynini.concat(sigma_star, tag_domain).optimize()
            input_parts = [cascade_domain]

        ordered_features = sorted(list(feature_map.keys()))

        cache_key = f"blueprint_{paradigm_name}"
        from parC.yaml_utils.cache import get_cached_fst
        _compile_stage_cascade(
            paradigm_name=paradigm_name,
            paradigm_data=paradigm_data,
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
        paradigm_name: str,
        root_regex: str = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
    ) -> pynini.Fst:
        """Builds open inflection graph FST for a paradigm."""
        if root_regex == "<Phone>*":
            return get_open_inflect_graph(
                paradigm_name=paradigm_name,
                lexical_features=lexical_features,
                non_deterministic_cleanup=non_deterministic_cleanup,
                infer_lexical_features=infer_lexical_features,
            )
        from parC.grammar.paradigm_compilation import build_inflect_graph_for_root_regex
        return build_inflect_graph_for_root_regex(
            paradigm_name=paradigm_name,
            root_regex=root_regex,
            lexical_features=lexical_features,
            non_deterministic_cleanup=non_deterministic_cleanup,
            infer_lexical_features=infer_lexical_features,
        )
