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
        inventory: Inventory,
        features: tuple[Feature, ...],
        syms: pynini.SymbolTable | None = None,
        special_fsas: dict[str, pynini.Fst] | None = None,
    ) -> None:
        self.inventory = inventory
        self.features = features
        self._syms = syms
        self._special_fsas = special_fsas

    @classmethod
    def from_config(cls) -> AlphabetBlueprint:
        """Constructs AlphabetBlueprint by reading global config once."""
        return cls(
            inventory=get_inventory_items(),
            features=get_feature_array(),
        )

    def get_symbol_table(self) -> pynini.SymbolTable:
        if self._syms is None:
            self._syms = build_symbol_table(self.inventory, self.features)
        return self._syms

    def get_special_fsas(self) -> dict[str, pynini.Fst]:
        if self._special_fsas is None:
            syms = self.get_symbol_table()
            self._special_fsas = _build_special_fsas(
                syms, self.inventory, self.features
            )
        return self._special_fsas

    def get_phone_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["phone"]

    def get_sigma_star(self) -> pynini.Fst:
        return self.get_special_fsas()["sigma_star"]

    def get_boundary_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["boundary"]

    def get_flag_acceptor(self) -> pynini.Fst:
        return self.get_special_fsas()["flag"]

