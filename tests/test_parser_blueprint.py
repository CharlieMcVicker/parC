"""Unit tests for Layer 5 ParsingEngineBlueprint."""

import pynini
import pytest

from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.paradigms import StageCascadeBlueprint
from parC.grammar.blueprints.parser import ParsingEngineBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.blueprints.transducers import (
    MarkerLibraryBlueprint,
    RulePipelineBlueprint,
)
from parC.grammar.paradigm_compilation import (
    build_parse_graph,
    get_open_inflect_graph,
)


def test_parsing_engine_blueprint_from_paradigm():
    """Test ParsingEngineBlueprint.from_paradigm factory method and dependency properties."""
    pe_bp = ParsingEngineBlueprint.from_paradigm("verb_a_stem")
    assert isinstance(pe_bp.cascade, StageCascadeBlueprint)
    assert isinstance(pe_bp.alphabet, AlphabetBlueprint)
    assert isinstance(pe_bp.patterns, PatternLibraryBlueprint)
    assert isinstance(pe_bp.rules, RulePipelineBlueprint)
    assert isinstance(pe_bp.markers, MarkerLibraryBlueprint)

    alph = AlphabetBlueprint.from_config()
    pat = PatternLibraryBlueprint.from_config()
    rul = RulePipelineBlueprint.from_config()
    mar = MarkerLibraryBlueprint.from_config()
    casc = StageCascadeBlueprint.from_paradigm(
        "verb_a_stem", alphabet=alph, patterns=pat, rules=rul, markers=mar
    )

    pe_bp_custom = ParsingEngineBlueprint(cascade=casc)
    assert pe_bp_custom.cascade is casc
    assert pe_bp_custom.alphabet is alph
    assert pe_bp_custom.patterns is pat
    assert pe_bp_custom.rules is rul
    assert pe_bp_custom.markers is mar


def test_build_open_parse_graph():
    """Test build_open_parse_graph produces inverted parse graph matching build_parse_graph(open_inflect)."""
    pe_bp = ParsingEngineBlueprint.from_paradigm("verb_a_stem")
    parse_graph = pe_bp.build_open_parse_graph()
    assert isinstance(parse_graph, pynini.Fst)

    expected_inflect = get_open_inflect_graph("verb_a_stem")
    expected_parse = build_parse_graph(expected_inflect)
    assert parse_graph.num_states() == expected_parse.num_states()


def test_build_search_lattice_from_paradigm():
    """Test build_search_lattice with paradigm_name factory."""
    pe_bp = ParsingEngineBlueprint.from_paradigm("verb_a_stem")
    search_lexicon, search_left_factor = pe_bp.build_search_lattice()
    assert isinstance(search_lexicon, pynini.Fst)
    assert isinstance(search_left_factor, pynini.Fst)
    assert search_lexicon.num_states() > 0
    assert search_left_factor.num_states() > 0


def test_build_search_lattice_from_inflect_graph():
    """Test build_search_lattice with explicit inflect_graph parameter."""
    pe_bp = ParsingEngineBlueprint.from_paradigm("verb_a_stem")
    inflect_fst = get_open_inflect_graph("verb_a_stem")
    search_lexicon, search_left_factor = pe_bp.build_search_lattice(inflect_graph=inflect_fst)
    assert isinstance(search_lexicon, pynini.Fst)
    assert isinstance(search_left_factor, pynini.Fst)
    assert search_lexicon.num_states() > 0
    assert search_left_factor.num_states() > 0

