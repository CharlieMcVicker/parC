"""Unit tests for Layer 4 StageCascadeBlueprint."""

import pynini
import pytest

from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.paradigms import StageCascadeBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.blueprints.transducers import (
    MarkerLibraryBlueprint,
    RulePipelineBlueprint,
)
from parC.grammar.paradigm_compilation import get_open_inflect_graph


def test_stage_cascade_blueprint_from_paradigm():
    """Test StageCascadeBlueprint.from_paradigm factory method."""
    sc_bp = StageCascadeBlueprint.from_paradigm("verb_a_stem")
    assert isinstance(sc_bp.alphabet, AlphabetBlueprint)
    assert isinstance(sc_bp.patterns, PatternLibraryBlueprint)
    assert isinstance(sc_bp.rules, RulePipelineBlueprint)
    assert isinstance(sc_bp.markers, MarkerLibraryBlueprint)
    assert sc_bp.paradigm_name == "verb_a_stem"

    alph = AlphabetBlueprint.from_config()
    pat = PatternLibraryBlueprint.from_config()
    rul = RulePipelineBlueprint.from_config()
    mar = MarkerLibraryBlueprint.from_config()

    sc_bp_custom = StageCascadeBlueprint.from_paradigm(
        "verb_a_stem", alphabet=alph, patterns=pat, rules=rul, markers=mar
    )
    assert sc_bp_custom.alphabet is alph
    assert sc_bp_custom.patterns is pat
    assert sc_bp_custom.rules is rul
    assert sc_bp_custom.markers is mar


def test_get_tag_domain_acceptor():
    """Test get_tag_domain_acceptor returns valid FST for a paradigm."""
    sc_bp = StageCascadeBlueprint.from_paradigm("verb_a_stem")
    tag_domain = sc_bp.get_tag_domain_acceptor()
    assert isinstance(tag_domain, pynini.Fst)
    assert tag_domain.num_states() > 0


def test_get_stage_gated_transducers():
    """Test get_stage_gated_transducers returns list of FSTs for a paradigm."""
    sc_bp = StageCascadeBlueprint.from_paradigm("verb_a_stem")
    gated_fsts = sc_bp.get_stage_gated_transducers()
    assert isinstance(gated_fsts, list)
    assert len(gated_fsts) > 0
    assert all(isinstance(fst, pynini.Fst) for fst in gated_fsts)

    with pytest.raises(ValueError):
        StageCascadeBlueprint.from_paradigm("non_existent_paradigm")


def test_build_open_inflect_graph():
    """Test build_open_inflect_graph matches get_open_inflect_graph output."""
    sc_bp = StageCascadeBlueprint.from_paradigm("verb_a_stem")
    graph = sc_bp.build_open_inflect_graph()
    expected = get_open_inflect_graph("verb_a_stem")
    assert isinstance(graph, pynini.Fst)
    assert graph.num_states() == expected.num_states()


def test_build_open_inflect_graph_custom_blueprint_injection():
    """Test build_open_inflect_graph with custom child blueprints injected."""
    alph = AlphabetBlueprint.from_config()
    pat = PatternLibraryBlueprint.from_config()
    rul = RulePipelineBlueprint.from_config()
    mar = MarkerLibraryBlueprint.from_config()

    sc_bp = StageCascadeBlueprint.from_paradigm(
        "verb_a_stem", alphabet=alph, patterns=pat, rules=rul, markers=mar
    )
    graph = sc_bp.build_open_inflect_graph()
    assert isinstance(graph, pynini.Fst)
    assert graph.num_states() > 0


def test_build_open_inflect_graph_custom_root_regex():
    """Test build_open_inflect_graph with custom root regex string and FST."""
    sc_bp = StageCascadeBlueprint.from_paradigm("verb_a_stem")

    # Custom regex pattern
    graph_pattern = sc_bp.build_open_inflect_graph(root_regex="c+a+n+t+")
    assert isinstance(graph_pattern, pynini.Fst)
    assert graph_pattern.num_states() > 0

    # Custom compiled FST
    custom_root_fst = sc_bp.patterns.compile_pattern_string("cant", alphabet=sc_bp.alphabet)
    graph_fst = sc_bp.build_open_inflect_graph(root_regex=custom_root_fst)
    assert isinstance(graph_fst, pynini.Fst)
    assert graph_fst.num_states() > 0



