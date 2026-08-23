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


def test_parsing_engine_blueprint_init():
    """Test default initialization and explicit dependency injection."""
    pe_bp = ParsingEngineBlueprint()
    assert isinstance(pe_bp.cascade, StageCascadeBlueprint)
    assert isinstance(pe_bp.alphabet, AlphabetBlueprint)
    assert isinstance(pe_bp.patterns, PatternLibraryBlueprint)
    assert isinstance(pe_bp.rules, RulePipelineBlueprint)
    assert isinstance(pe_bp.markers, MarkerLibraryBlueprint)

    alph = AlphabetBlueprint()
    pat = PatternLibraryBlueprint(alphabet=alph)
    rul = RulePipelineBlueprint(alphabet=alph, patterns=pat)
    mar = MarkerLibraryBlueprint(alphabet=alph, patterns=pat)
    casc = StageCascadeBlueprint(alphabet=alph, patterns=pat, rules=rul, markers=mar)

    pe_bp_custom = ParsingEngineBlueprint(cascade=casc)
    assert pe_bp_custom.cascade is casc
    assert pe_bp_custom.alphabet is alph
    assert pe_bp_custom.patterns is pat
    assert pe_bp_custom.rules is rul
    assert pe_bp_custom.markers is mar


def test_build_open_parse_graph():
    """Test build_open_parse_graph produces inverted parse graph matching build_parse_graph(open_inflect)."""
    pe_bp = ParsingEngineBlueprint()
    parse_graph = pe_bp.build_open_parse_graph("verb_a_stem")
    assert isinstance(parse_graph, pynini.Fst)

    expected_inflect = get_open_inflect_graph("verb_a_stem")
    expected_parse = build_parse_graph(expected_inflect)
    assert parse_graph.num_states() == expected_parse.num_states()


def test_build_search_lattice_from_paradigm():
    """Test build_search_lattice with paradigm_name parameter."""
    pe_bp = ParsingEngineBlueprint()
    search_lexicon, search_left_factor = pe_bp.build_search_lattice(paradigm_name="verb_a_stem")
    assert isinstance(search_lexicon, pynini.Fst)
    assert isinstance(search_left_factor, pynini.Fst)
    assert search_lexicon.num_states() > 0
    assert search_left_factor.num_states() > 0


def test_build_search_lattice_from_inflect_graph():
    """Test build_search_lattice with explicit inflect_graph parameter."""
    pe_bp = ParsingEngineBlueprint()
    inflect_fst = get_open_inflect_graph("verb_a_stem")
    search_lexicon, search_left_factor = pe_bp.build_search_lattice(inflect_graph=inflect_fst)
    assert isinstance(search_lexicon, pynini.Fst)
    assert isinstance(search_left_factor, pynini.Fst)
    assert search_lexicon.num_states() > 0
    assert search_left_factor.num_states() > 0


def test_build_search_lattice_missing_args():
    """Test build_search_lattice raises ValueError when neither argument is provided."""
    pe_bp = ParsingEngineBlueprint()
    with pytest.raises(ValueError, match="Either paradigm_name or inflect_graph must be provided."):
        pe_bp.build_search_lattice()
