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


def test_stage_cascade_blueprint_init():
    """Test default initialization and explicit dependency injection."""
    sc_bp = StageCascadeBlueprint()
    assert isinstance(sc_bp.alphabet, AlphabetBlueprint)
    assert isinstance(sc_bp.patterns, PatternLibraryBlueprint)
    assert isinstance(sc_bp.rules, RulePipelineBlueprint)
    assert isinstance(sc_bp.markers, MarkerLibraryBlueprint)

    alph = AlphabetBlueprint()
    pat = PatternLibraryBlueprint(alphabet=alph)
    rul = RulePipelineBlueprint(alphabet=alph, patterns=pat)
    mar = MarkerLibraryBlueprint(alphabet=alph, patterns=pat)

    sc_bp_custom = StageCascadeBlueprint(
        alphabet=alph, patterns=pat, rules=rul, markers=mar
    )
    assert sc_bp_custom.alphabet is alph
    assert sc_bp_custom.patterns is pat
    assert sc_bp_custom.rules is rul
    assert sc_bp_custom.markers is mar


def test_get_tag_domain_acceptor():
    """Test get_tag_domain_acceptor returns valid FST for a paradigm."""
    sc_bp = StageCascadeBlueprint()
    tag_domain = sc_bp.get_tag_domain_acceptor("verb_a_stem")
    assert isinstance(tag_domain, pynini.Fst)
    assert tag_domain.num_states() > 0


def test_get_stage_gated_transducers():
    """Test get_stage_gated_transducers returns list of FSTs for a paradigm."""
    sc_bp = StageCascadeBlueprint()
    gated_fsts = sc_bp.get_stage_gated_transducers("verb_a_stem")
    assert isinstance(gated_fsts, list)
    assert len(gated_fsts) > 0
    assert all(isinstance(fst, pynini.Fst) for fst in gated_fsts)

    with pytest.raises(ValueError):
        sc_bp.get_stage_gated_transducers("non_existent_paradigm")


def test_build_open_inflect_graph():
    """Test build_open_inflect_graph matches get_open_inflect_graph output."""
    sc_bp = StageCascadeBlueprint()
    graph = sc_bp.build_open_inflect_graph("verb_a_stem")
    expected = get_open_inflect_graph("verb_a_stem")
    assert isinstance(graph, pynini.Fst)
    assert graph.num_states() == expected.num_states()

