"""Layer 5: ParsingEngineBlueprint (Parse Graph Inversion & Search Lattice Engine).

Wraps open parse graph inversion and search lattice construction while taking explicit
StageCascadeBlueprint (and transitively lower-level) dependencies.
"""

from __future__ import annotations

import pynini

from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.paradigms import StageCascadeBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.blueprints.transducers import (
    MarkerLibraryBlueprint,
    RulePipelineBlueprint,
)
from parC.grammar.paradigm_compilation import (
    build_parse_graph,
    build_search_lexicon_and_leftfactor,
)


class ParsingEngineBlueprint:
    """Blueprint for Layer 5 Parsing Engine and Fuzzy Search Lattice.

    Wraps parse graph inversion and search graph/lattice construction while taking explicit
    lower-layer blueprints as dependencies.
    """

    def __init__(
        self,
        cascade: StageCascadeBlueprint | None = None,
        alphabet: AlphabetBlueprint | None = None,
        patterns: PatternLibraryBlueprint | None = None,
        rules: RulePipelineBlueprint | None = None,
        markers: MarkerLibraryBlueprint | None = None,
    ) -> None:
        if cascade is not None:
            self.cascade = cascade
        else:
            self.cascade = StageCascadeBlueprint(
                alphabet=alphabet,
                patterns=patterns,
                rules=rules,
                markers=markers,
            )

    @property
    def alphabet(self) -> AlphabetBlueprint:
        return self.cascade.alphabet

    @property
    def patterns(self) -> PatternLibraryBlueprint:
        return self.cascade.patterns

    @property
    def rules(self) -> RulePipelineBlueprint:
        return self.cascade.rules

    @property
    def markers(self) -> MarkerLibraryBlueprint:
        return self.cascade.markers

    def build_open_parse_graph(
        self,
        paradigm_name: str,
        root_regex: str = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
    ) -> pynini.Fst:
        """Builds open parse graph FST by inverting open inflection graph FST."""
        open_inflect = self.cascade.build_open_inflect_graph(
            paradigm_name=paradigm_name,
            root_regex=root_regex,
            lexical_features=lexical_features,
            infer_lexical_features=infer_lexical_features,
            non_deterministic_cleanup=non_deterministic_cleanup,
        )
        return build_parse_graph(open_inflect)

    def build_search_lattice(
        self,
        paradigm_name: str | None = None,
        inflect_graph: pynini.Fst | None = None,
        root_regex: str = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
    ) -> tuple[pynini.Fst, pynini.Fst]:
        """Builds search lexicon and search left factor FSTs for fuzzy search lattice.

        Either paradigm_name or inflect_graph must be provided.
        """
        if inflect_graph is None:
            if paradigm_name is None:
                raise ValueError("Either paradigm_name or inflect_graph must be provided.")
            inflect_graph = self.cascade.build_open_inflect_graph(
                paradigm_name=paradigm_name,
                root_regex=root_regex,
                lexical_features=lexical_features,
                infer_lexical_features=infer_lexical_features,
                non_deterministic_cleanup=non_deterministic_cleanup,
            )
        return build_search_lexicon_and_leftfactor(inflect_graph)
