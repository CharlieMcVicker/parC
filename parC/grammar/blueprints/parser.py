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

    Holds explicit StageCascadeBlueprint as dependency and exposes methods for parse graph inversion
    and search lattice construction.
    """

    def __init__(
        self,
        cascade: StageCascadeBlueprint,
    ) -> None:
        self.cascade = cascade

    @classmethod
    def from_paradigm(
        cls,
        paradigm_name: str,
        alphabet: AlphabetBlueprint | None = None,
        patterns: PatternLibraryBlueprint | None = None,
        rules: RulePipelineBlueprint | None = None,
        markers: MarkerLibraryBlueprint | None = None,
    ) -> ParsingEngineBlueprint:
        """Constructs ParsingEngineBlueprint for a named paradigm."""
        cascade = StageCascadeBlueprint.from_paradigm(
            paradigm_name=paradigm_name,
            alphabet=alphabet,
            patterns=patterns,
            rules=rules,
            markers=markers,
        )
        return cls(cascade=cascade)

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
        root_regex: str | pynini.Fst = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
    ) -> pynini.Fst:
        """Builds open parse graph FST by inverting open inflection graph FST."""
        open_inflect = self.cascade.build_open_inflect_graph(
            root_regex=root_regex,
            lexical_features=lexical_features,
            infer_lexical_features=infer_lexical_features,
            non_deterministic_cleanup=non_deterministic_cleanup,
        )
        return build_parse_graph(open_inflect)

    def build_search_lattice(
        self,
        inflect_graph: pynini.Fst | None = None,
        root_regex: str | pynini.Fst = "<Phone>*",
        lexical_features: dict[str, str] | None = None,
        infer_lexical_features: bool = False,
        non_deterministic_cleanup: bool = False,
    ) -> tuple[pynini.Fst, pynini.Fst]:
        """Builds search lexicon and search left factor FSTs for fuzzy search lattice."""
        if inflect_graph is None:
            inflect_graph = self.cascade.build_open_inflect_graph(
                root_regex=root_regex,
                lexical_features=lexical_features,
                infer_lexical_features=infer_lexical_features,
                non_deterministic_cleanup=non_deterministic_cleanup,
            )
        return build_search_lexicon_and_leftfactor(inflect_graph)

