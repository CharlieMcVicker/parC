"""Layer 1: AlphabetBlueprint (Phonemic Alphabet & Tag Symbol Space).

Wraps symbol table compilation and special symbol acceptor FSAs while preserving
pure function signatures in `acceptor_compilation.py`.
"""

from __future__ import annotations

import pynini

from parC.grammar.acceptor_compilation import (
    _build_special_fsas,
    build_symbol_table,
    get_special_fsas,
    get_symbol_table,
)
from parC.yaml_utils.models import Feature, Inventory
from parC.yaml_utils.yaml_server import get_feature_array, get_inventory_items


class AlphabetBlueprint:
    """Blueprint for Layer 1 Phonemic Alphabet and Tag Symbol Space.

    Exposes named domain methods for accessing symbol tables and special FSAs
    (phone, flag, boundary, sigma_star).
    """

    def __init__(
        self,
        inventory: Inventory | None = None,
        features: tuple[Feature, ...] | None = None,
        syms: pynini.SymbolTable | None = None,
    ) -> None:
        self._custom_inventory = inventory
        self._custom_features = features
        self._syms = syms
        self._special_fsas: dict[str, pynini.Fst] | None = None

    @property
    def inventory(self) -> Inventory:
        if self._custom_inventory is not None:
            return self._custom_inventory
        return get_inventory_items()

    @property
    def features(self) -> tuple[Feature, ...]:
        if self._custom_features is not None:
            return self._custom_features
        return get_feature_array()

    def get_symbol_table(self) -> pynini.SymbolTable:
        if self._syms is None:
            if self._custom_inventory is not None or self._custom_features is not None:
                self._syms = build_symbol_table(self.inventory, self.features)
            else:
                self._syms = get_symbol_table()
        return self._syms

    def get_special_fsas(self) -> dict[str, pynini.Fst]:
        if self._special_fsas is None:
            syms = self.get_symbol_table()
            if self._custom_inventory is not None or self._custom_features is not None:
                self._special_fsas = _build_special_fsas(
                    syms, self.inventory, self.features
                )
            else:
                self._special_fsas = get_special_fsas()
        return self._special_fsas

    def get_phone_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["phone"]

    def get_sigma_star(self) -> pynini.Fst:
        return self.get_special_fsas()["sigma_star"]

    def get_boundary_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["boundary"]

    def get_flag_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["flag"]
