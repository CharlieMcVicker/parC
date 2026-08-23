"""Unit tests for Layer 1 AlphabetBlueprint."""

import pynini
import pytest

from parC.grammar.acceptor_compilation import build_symbol_table, get_special_fsas, get_symbol_table
from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.yaml_utils.models import Feature, FeatureValueDef, Inventory


def test_alphabet_blueprint_default():
    """Test AlphabetBlueprint with default project environment config via from_config factory."""
    bp = AlphabetBlueprint.from_config()

    syms = bp.get_symbol_table()
    assert isinstance(syms, pynini.SymbolTable)
    assert syms.write_to_string() == get_symbol_table().write_to_string()

    phone = bp.get_phone_acceptor()
    assert isinstance(phone, pynini.Fst)

    sigma_star = bp.get_sigma_star()
    assert isinstance(sigma_star, pynini.Fst)

    boundary = bp.get_boundary_acceptor()
    assert isinstance(boundary, pynini.Fst)

    flag = bp.get_flag_acceptor()
    assert isinstance(flag, pynini.Fst)

    special_fsas = bp.get_special_fsas()
    assert isinstance(special_fsas, dict)
    assert "phone" in special_fsas
    assert "sigma_star" in special_fsas


def test_alphabet_blueprint_custom_inventory():
    """Test AlphabetBlueprint with explicit custom inventory and feature definitions."""
    custom_inv = Inventory(item_map={}, phones=("a", "b", "k"), tags=("+N",))
    custom_feats = (
        Feature(
            name="pos",
            values=(FeatureValueDef(name="verb"), FeatureValueDef(name="noun")),
        ),
    )

    bp = AlphabetBlueprint(inventory=custom_inv, features=custom_feats)

    syms = bp.get_symbol_table()
    assert isinstance(syms, pynini.SymbolTable)
    assert syms.find("a") != -1
    assert syms.find("b") != -1
    assert syms.find("k") != -1
    assert syms.find("+N") != -1
    assert syms.find("[pos=verb]") != -1
    assert syms.find("[pos=noun]") != -1

    phone_fst = bp.get_phone_acceptor()
    assert isinstance(phone_fst, pynini.Fst)
    # Check that custom phone 'a' is accepted by phone_fst
    a_fst = pynini.accep("a", token_type=syms)
    composed = pynini.compose(a_fst, phone_fst)
    assert composed.num_states() > 0

    sigma_star_fst = bp.get_sigma_star()
    assert isinstance(sigma_star_fst, pynini.Fst)


def test_pure_function_signatures_preserved():
    """Verify that pure functions build_symbol_table and get_special_fsas remain untouched."""
    inv = Inventory(item_map={}, phones=("p", "t"), tags=())
    feats = ()

    syms = build_symbol_table(inv, feats)
    assert isinstance(syms, pynini.SymbolTable)
    assert syms.find("p") != -1
    assert syms.find("t") != -1

    default_specials = get_special_fsas()
    assert isinstance(default_specials, dict)
    assert "sigma" in default_specials
