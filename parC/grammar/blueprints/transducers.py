"""Layer 3: RulePipelineBlueprint & MarkerLibraryBlueprint (Atomic Phonological & Exponence Operations).

Wraps rewrite rules and morphological exponence operations while taking explicit
AlphabetBlueprint and PatternLibraryBlueprint dependencies.
"""

from __future__ import annotations

import pynini

from parC.grammar.config_loader import GrammarConfig
from parC.grammar.blueprints.alphabet import AlphabetBlueprint
from parC.grammar.blueprints.patterns import PatternLibraryBlueprint
from parC.grammar.transducer_compilation import (
    compile_marker,
    compile_rule,
    get_gated_marker_fst,
    get_marker_fst,
    get_rule_fst,
)
from parC.yaml_utils.models import Marker, Rule, RuleSequence
from parC.yaml_utils.yaml_server import get_rules


class RulePipelineBlueprint:
    """Blueprint for Layer 3 Rule Pipeline & Rewrite Transducers.

    Stores rule models explicitly and receives lower-layer blueprints via method injection.
    """

    def __init__(
        self,
        rules: dict[str, Rule],
    ) -> None:
        self.rules = rules

    @classmethod
    def from_config(cls, config: GrammarConfig | None = None) -> RulePipelineBlueprint:
        """Constructs RulePipelineBlueprint from a pre-loaded GrammarConfig or by reading global config."""
        if config is not None:
            return cls(rules=config.rules)
        return cls(rules=get_rules())

    def _compile_rule_obj(self, rule: Rule) -> pynini.Fst | list[pynini.Fst]:
        """Internal helper to compile a Rule instance using self.rules for RuleSequence resolution."""
        if isinstance(rule, RuleSequence):
            result: list[pynini.Fst] = []
            for name in rule.rule_sequence:
                name_clean = name.removeprefix("$")
                if name_clean not in self.rules:
                    raise KeyError(
                        f"Rule '{name_clean}' not found in set of rules {list(self.rules.keys())}"
                    )
                sub_fst = self._compile_rule_obj(self.rules[name_clean])
                if isinstance(sub_fst, list):
                    result.extend(sub_fst)
                else:
                    result.append(sub_fst)
            return result
        return compile_rule(rule)

    def compile_rule_transducer(self, rule_name: str) -> pynini.Fst | list[pynini.Fst]:
        """Compiles or retrieves FST(s) for a given rule name."""
        rule_name_clean = rule_name.removeprefix("$")
        if rule_name_clean not in self.rules:
            raise KeyError(
                f"Rule '{rule_name_clean}' not found in set of rules {list(self.rules.keys())}"
            )
        return self._compile_rule_obj(self.rules[rule_name_clean])

    def get_rule_sequence_fst(self, sequence_name: str) -> list[pynini.Fst]:
        """Retrieves flat list of FSTs for a rule sequence."""
        rule_name_clean = sequence_name.removeprefix("$")
        if rule_name_clean not in self.rules:
            raise KeyError(
                f"Rule sequence '{rule_name_clean}' not found in set of rules {list(self.rules.keys())}"
            )
        rule = self.rules[rule_name_clean]
        if not isinstance(rule, RuleSequence):
            raise TypeError(
                f"Rule '{rule_name_clean}' is not a RuleSequence instance (got {type(rule)!r})"
            )
        res = self.compile_rule_transducer(rule_name_clean)
        if isinstance(res, list):
            return res
        return [res]


class MarkerLibraryBlueprint:
    """Blueprint for Layer 3 Marker Library & Exponence Transducers.

    Wraps marker compilation functions and stage grouping methods.
    """

    def __init__(self) -> None:
        pass

    @classmethod
    def from_config(cls) -> MarkerLibraryBlueprint:
        """Constructs MarkerLibraryBlueprint."""
        return cls()

    def compile_marker_transducer(
        self, marker: Marker, trigger_tags: tuple[str, ...] | list[str] | None = None
    ) -> pynini.Fst:
        """Compiles or retrieves FST for a marker, optionally gated by trigger_tags."""
        if trigger_tags:
            return get_gated_marker_fst(marker, tuple(trigger_tags))
        return get_marker_fst(marker)

    def get_markers_by_stage(
        self, markers: list[Marker] | tuple[Marker, ...]
    ) -> dict[str, list[Marker]]:
        """Groups a sequence/collection of markers by their 'stage' attribute."""
        staged: dict[str, list[Marker]] = {}
        for m in markers:
            stage = getattr(m, "stage", None) or "unspecified"
            if stage not in staged:
                staged[stage] = []
            staged[stage].append(m)
        return staged

    def compile_stage_markers(
        self,
        markers: list[Marker] | tuple[Marker, ...],
        trigger_tags_map: dict[Marker, tuple[str, ...]] | None = None,
    ) -> dict[str, list[pynini.Fst]]:
        """Groups markers by stage and compiles each marker into an FST."""
        staged = self.get_markers_by_stage(markers)
        trigger_map = trigger_tags_map or {}
        compiled_staged: dict[str, list[pynini.Fst]] = {}
        for stage, stage_markers in staged.items():
            compiled_staged[stage] = [
                self.compile_marker_transducer(m, trigger_map.get(m))
                for m in stage_markers
            ]
        return compiled_staged

