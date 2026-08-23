"""Unit tests for Layer 3 RulePipelineBlueprint and MarkerLibraryBlueprint."""

import pynini
import pytest

from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.blueprints.transducers import (
    MarkerLibraryBlueprint,
    RulePipelineBlueprint,
)
from parC.grammar.transducer_compilation import compile_marker, compile_rule
from parC.yaml_utils.models import (
    Inventory,
    RuleSequence,
    SimpleRule,
    SingleStringMarker,
    StringTupleMarker,
)


def test_rule_pipeline_blueprint_default():
    """Test RulePipelineBlueprint with default project environment config."""
    rule_bp = RulePipelineBlueprint()
    assert isinstance(rule_bp.alphabet, AlphabetBlueprint)
    assert isinstance(rule_bp.patterns, PatternLibraryBlueprint)

    # Pick a rule from default config if available
    rules = rule_bp.rules
    assert isinstance(rules, dict)
    assert len(rules) > 0

    rule_name = next(iter(rules.keys()))
    res = rule_bp.compile_rule_transducer(rule_name)
    if isinstance(res, list):
        assert all(isinstance(fst, pynini.Fst) for fst in res)
    else:
        assert isinstance(res, pynini.Fst)


def test_rule_pipeline_blueprint_custom():
    """Test RulePipelineBlueprint with explicit custom rules and dependencies."""
    inv = Inventory(item_map={}, phones=("a", "b", "c"), tags=())
    alph_bp = AlphabetBlueprint(inventory=inv, features=())
    pat_bp = PatternLibraryBlueprint(alphabet=alph_bp)

    custom_rules = {
        "rule_a_to_b": SimpleRule(input_pattern="a", output_pattern="b"),
        "rule_seq": RuleSequence(rule_sequence=("rule_a_to_b",)),
    }

    rule_bp = RulePipelineBlueprint(
        alphabet=alph_bp, patterns=pat_bp, rules=custom_rules
    )

    # Test compile simple rule
    fst = rule_bp.compile_rule_transducer("rule_a_to_b")
    assert isinstance(fst, pynini.Fst)

    # Test get rule sequence
    seq_fsts = rule_bp.get_rule_sequence_fst("rule_seq")
    assert isinstance(seq_fsts, list)
    assert len(seq_fsts) == 1
    assert isinstance(seq_fsts[0], pynini.Fst)

    # Test error cases
    with pytest.raises(KeyError):
        rule_bp.compile_rule_transducer("non_existent_rule")

    with pytest.raises(TypeError):
        rule_bp.get_rule_sequence_fst("rule_a_to_b")


def test_marker_library_blueprint():
    """Test MarkerLibraryBlueprint compilation and stage grouping."""
    marker_bp = MarkerLibraryBlueprint()

    m1 = SingleStringMarker(kind="prefix", value="ba_", stage="prefix_stage")
    m2 = SingleStringMarker(kind="suffix", value="_ab", stage="suffix_stage")
    m3 = StringTupleMarker(kind="replace", value=("a", "b"), stage="suffix_stage")
    m4 = SingleStringMarker(kind="prefix", value="ba", stage=None)


    # Test compile_marker_transducer
    fst1 = marker_bp.compile_marker_transducer(m1)
    assert isinstance(fst1, pynini.Fst)

    # Test gated compilation using a valid tag from default symbol table
    syms = marker_bp.alphabet.get_symbol_table()
    # Find any tag symbol in syms (tags in default inventory or boundary tags)
    valid_tag = "[BOW]"
    gated_fst = marker_bp.compile_marker_transducer(m1, trigger_tags=(valid_tag,))
    assert isinstance(gated_fst, pynini.Fst)


    # Test stage grouping
    markers = [m1, m2, m3, m4]
    by_stage = marker_bp.get_markers_by_stage(markers)
    assert "prefix_stage" in by_stage
    assert "suffix_stage" in by_stage
    assert "unspecified" in by_stage

    assert by_stage["prefix_stage"] == [m1]
    assert by_stage["suffix_stage"] == [m2, m3]
    assert by_stage["unspecified"] == [m4]

    # Test compile_stage_markers
    stage_compiled = marker_bp.compile_stage_markers(
        markers, trigger_tags_map={m1: (valid_tag,)}
    )

    assert "prefix_stage" in stage_compiled
    assert "suffix_stage" in stage_compiled
    assert len(stage_compiled["prefix_stage"]) == 1
    assert isinstance(stage_compiled["prefix_stage"][0], pynini.Fst)
    assert len(stage_compiled["suffix_stage"]) == 2
    assert all(isinstance(f, pynini.Fst) for f in stage_compiled["suffix_stage"])


def test_pure_functions_preserved():
    """Verify underlying pure functions compile_rule and compile_marker remain intact."""
    rule = SimpleRule(input_pattern="x", output_pattern="y")
    rfst = compile_rule(rule)
    assert isinstance(rfst, pynini.Fst)

    marker = SingleStringMarker(kind="prefix", value="p-")
    mfst = compile_marker(marker)
    assert isinstance(mfst, pynini.Fst)
