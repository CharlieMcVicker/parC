"""Layer 2: PatternLibraryBlueprint (Pattern Library & Class/Pattern Acceptors).

Wraps pattern acceptor compilation and recursive descent pattern parsing
while taking explicit AlphabetBlueprint dependency.
"""

from __future__ import annotations

import pynini

from parC.grammar.acceptor_compilation import (
    _build_class_fsts,
    _build_token_map,
    _parse_pattern,
    compile_all_patterns,
    get_feature_acceptor_fsts,
    get_pattern_fsts,
)
from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.yaml_utils.models import Pattern
from parC.yaml_utils.yaml_server import get_patterns


class PatternLibraryBlueprint:
    """Blueprint for Layer 2 Pattern Library & Class/Pattern Acceptors.

    Wraps pattern compilation functions and recursive descent pattern parsing,
    explicitly accepting an AlphabetBlueprint instance for symbol table / FSA resolution.
    """

    def __init__(
        self,
        alphabet: AlphabetBlueprint | None = None,
        patterns: dict[str, Pattern] | None = None,
    ) -> None:
        self.alphabet = alphabet or AlphabetBlueprint()
        self._custom_patterns = patterns
        self._pattern_acceptors: dict[str, pynini.Fst] | None = None

    @property
    def patterns(self) -> dict[str, Pattern]:
        if self._custom_patterns is not None:
            return self._custom_patterns
        return get_patterns()

    def get_all_pattern_acceptors(self) -> dict[str, pynini.Fst]:
        """Returns compiled pattern acceptors mapping pattern/class name to FST acceptor."""
        if self._pattern_acceptors is None:
            if (
                self._custom_patterns is not None
                or self.alphabet._custom_inventory is not None
                or self.alphabet._custom_features is not None
            ):
                syms = self.alphabet.get_symbol_table()
                inventory = self.alphabet.inventory
                features = self.alphabet.features
                patterns = self.patterns
                special_fsas = self.alphabet.get_special_fsas()
                class_fsts = _build_class_fsts(syms, inventory)
                phone_starts = {p[0] for p in inventory.phones}
                token_map = _build_token_map(syms, inventory, features, patterns)
                self._pattern_acceptors = compile_all_patterns(
                    patterns,
                    token_map,
                    phone_starts,
                    syms,
                    special_fsas["sigma"],
                    special_fsas,
                    class_fsts,
                )
            else:
                self._pattern_acceptors = get_pattern_fsts()
        return self._pattern_acceptors

    def get_pattern_acceptor(self, name: str) -> pynini.Fst:
        """Get compiled acceptor FST for a specific named pattern or inventory class."""
        acceptors = self.get_all_pattern_acceptors()
        if name not in acceptors:
            raise KeyError(
                f"Pattern or class name '{name}' not found in compiled pattern acceptors."
            )
        return acceptors[name]

    def compile_pattern_string(self, pattern_str: str) -> pynini.Fst:
        """Compile a pattern DSL string on-the-fly using the symbol space and pattern library."""
        syms = self.alphabet.get_symbol_table()
        inventory = self.alphabet.inventory
        features = self.alphabet.features
        patterns = self.patterns
        special_fsas = self.alphabet.get_special_fsas()
        phone_starts = {p[0] for p in inventory.phones}
        token_map = _build_token_map(syms, inventory, features, patterns)
        compiled_patterns = self.get_all_pattern_acceptors()
        sigma = special_fsas["sigma"]
        return _parse_pattern(
            pattern_str,
            token_map,
            phone_starts,
            compiled_patterns,
            syms,
            sigma,
            special_fsas,
        )
