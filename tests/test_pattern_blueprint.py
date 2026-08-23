"""Unit tests for Layer 2 PatternLibraryBlueprint."""

import pynini
import pytest

from parC.grammar.acceptor_compilation import (
    _build_class_fsts,
    compile_all_patterns,
    get_pattern_fsts,
)
from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.yaml_utils.models import Inventory, Pattern


def test_pattern_library_blueprint_default():
    """Test PatternLibraryBlueprint with default project environment config."""
    alphabet_bp = AlphabetBlueprint.from_config()
    pattern_bp = PatternLibraryBlueprint.from_config()

    all_acceptors = pattern_bp.get_all_pattern_acceptors(alphabet_bp)
    assert isinstance(all_acceptors, dict)
    assert len(all_acceptors) > 0

    # Test get_pattern_acceptor for an existing key
    sample_key = next(iter(all_acceptors.keys()))
    acc = pattern_bp.get_pattern_acceptor(sample_key, alphabet_bp)
    assert isinstance(acc, pynini.Fst)

    # Test error handling for non-existent pattern key
    with pytest.raises(KeyError):
        pattern_bp.get_pattern_acceptor("NON_EXISTENT_PATTERN_12345", alphabet_bp)


def test_pattern_library_blueprint_explicit_alphabet_and_custom_patterns():
    """Test PatternLibraryBlueprint with custom AlphabetBlueprint and custom patterns."""
    custom_inv = Inventory(item_map={}, phones=("a", "b", "c"), tags=())
    alphabet_bp = AlphabetBlueprint(inventory=custom_inv, features=())

    custom_pats = {
        "<Vowel>": Pattern(name="Vowel", pattern="(a|b)"),
        "<PatternAB>": Pattern(name="PatternAB", pattern="ab"),
        "<PatternRef>": Pattern(name="PatternRef", pattern="<Vowel>c"),
    }

    pattern_bp = PatternLibraryBlueprint(patterns=custom_pats)

    acceptors = pattern_bp.get_all_pattern_acceptors(alphabet_bp)
    assert "<Vowel>" in acceptors
    assert "<PatternAB>" in acceptors
    assert "<PatternRef>" in acceptors

    vowel_fst = pattern_bp.get_pattern_acceptor("<Vowel>", alphabet_bp)
    ab_fst = pattern_bp.get_pattern_acceptor("<PatternAB>", alphabet_bp)
    ref_fst = pattern_bp.get_pattern_acceptor("<PatternRef>", alphabet_bp)

    syms = alphabet_bp.get_symbol_table()
    a_fst = pynini.accep("a", token_type=syms)
    b_fst = pynini.accep("b", token_type=syms)
    ac_fst = pynini.accep("a", token_type=syms) + pynini.accep("c", token_type=syms)

    # Test vowel acceptor matches 'a' and 'b'
    assert pynini.compose(a_fst, vowel_fst).num_states() > 0
    assert pynini.compose(b_fst, vowel_fst).num_states() > 0

    # Test pattern reference '<Vowel>c' matches 'a' concatenated with 'c'
    assert pynini.compose(ac_fst, ref_fst).num_states() > 0


def test_compile_pattern_string_on_the_fly():
    """Test PatternLibraryBlueprint.compile_pattern_string method."""
    alphabet_bp = AlphabetBlueprint.from_config()
    pattern_bp = PatternLibraryBlueprint.from_config()

    compiled_fst = pattern_bp.compile_pattern_string("[BOW]a[EOW]", alphabet=alphabet_bp)
    assert isinstance(compiled_fst, pynini.Fst)

    syms = alphabet_bp.get_symbol_table()
    a_fst = pynini.accep("[BOW]", token_type=syms) + pynini.accep("a", token_type=syms) + pynini.accep("[EOW]", token_type=syms)
    assert pynini.compose(a_fst, compiled_fst).num_states() > 0


def test_pure_function_signatures_preserved():
    """Verify that pure compilation functions remain operational and signatures are preserved."""
    inv = Inventory(item_map={}, phones=("m", "n"), tags=())
    syms = AlphabetBlueprint(inventory=inv, features=()).get_symbol_table()
    class_fsts = _build_class_fsts(syms, inv)
    assert isinstance(class_fsts, dict)
