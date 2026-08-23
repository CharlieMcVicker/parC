# Engineering Specification: Meta-Label Query Lattice Compiler & Acceptor Filter

**Target Location:** Consuming Application Layer (Lexicon Derivation / Lexicography Orchestration Pipeline)

**Dependencies:** `parC` (via Blueprint FST exports), `pynini`, standard library `dataclasses`

---

## 1. Architectural Overview & Objective

The objective of this module is to construct **underspecified input query lattices** ($\mathcal{Q} = \text{Surface} \cdot \mathcal{L}_{\text{restricted}}$) to constrain `parC`'s inverted parse graph ($\mathcal{P}$).

Rather than modifying `parC`'s internal grammar cascades, the consuming application consumes `parC`'s compiled **Label Acceptor FSTs** (which define valid morphotactic tag sequences) and restricts them via **Constraint Intersections** driven by high-level **Meta-Labels** (e.g., citation form defaults, number overrides, known aspect classes).

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│ CONSUMING CODEBASE                                                                      │
│                                                                                         │
│  [MetaLabel Profile] ──▶ [MetaConstraintCompiler]                                       │
│                                  │                                                      │
│                                  ▼ (Intersects Feature Masks)                           │
│                        [Restricted Tag Acceptor] ──┐                                    │
│                                                    │ Concatenate                        │
│  "surface_form" ───────────────────────────────────┴──▶ [Query Lattice Q]               │
└───────────────────────────────────────────────────────────────────┬─────────────────────┘
                                                                    │ Compose (Q ∘ P)
┌───────────────────────────────────────────────────────────────────▼─────────────────────┐
│ parC CORE ENGINE                                                                        │
│                                                                                         │
│  Exported Assets:                                                                       │
│  - Tag Morphotactic Acceptor (L_base)                                                   │
│  - Inverted Morphological Parse Graph (P = T^-1)                                        │
│                                                                                         │
│  Result: Compact Output Parse Graph (Discovered Lexical Stems, Aspect Classes, Sets)   │
└─────────────────────────────────────────────────────────────────────────────────────────┘

```

---

## 2. Core Data Models (Python `dataclasses`)

The domain models define declarative constraints on feature slots, resolve interactions/defaults between meta-labels, and maintain discovered lexical state across iterative extraction passes.

```python
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Union


class MatchMode(str, Enum):
    """Specifies how feature values should be matched in the slot."""
    EXACT = "exact"          # Matches only the specified value
    ONE_OF = "one_of"        # Matches any value in the provided list
    ANY = "any"              # Unconstrained wildcard (.*)
    EXCLUDE = "exclude"      # Must not match the provided value(s)


@dataclass
class FeatureConstraint:
    """Constraint on a single morphosyntactic or lexical slot."""
    slot_name: str                                  # e.g., "person_number", "aspect_class", "tense"
    mode: MatchMode = MatchMode.EXACT
    values: List[str] = field(default_factory=list) # e.g., ["3pl", "3du_incl"] or ["class_2"]


@dataclass
class MetaLabelDefinition:
    """Definition of an abstract meta-label and its associated feature constraints."""
    id: str                                         # e.g., "META_3_PRS", "META_PL_ONLY"
    description: Optional[str] = None
    constraints: List[FeatureConstraint] = field(default_factory=list)
    
    # Priority for resolving conflicting constraints across composite meta-labels
    priority: int = 0


@dataclass
class MacroProfile:
    """A collection of active meta-labels applied to a reference form query."""
    name: str                                               # e.g., "citation_default_verb"
    meta_label_ids: List[str] = field(default_factory=list) # e.g., ["META_3_PRS", "META_SET_A"]
    dynamic_constraints: List[FeatureConstraint] = field(default_factory=list)
    # Dynamic constraints allow injecting discovered features (e.g., aspect class) from earlier steps


@dataclass
class ParsedLexicalProfile:
    """Discovered lexical features extracted from constrained parse results."""
    surface_form: str
    root: str
    aspect_class: Optional[str] = None
    pronoun_set: Optional[str] = None
    full_tag_sequence: str = ""
    confidence_weight: float = 0.0

```

---

## 3. Query Lattice Compiler (`MetaConstraintCompiler`)

The compiler loads base morphotactic acceptors exported by `parC` blueprints and applies feature-mask acceptors via OpenFST/Pynini operations.

```python
import pynini
from pynini.lib import byte


class MetaConstraintCompiler:
    """
    Assembles restricted tag acceptors and combines them with surface forms 
    to create underspecified parse query lattices.
    """

    def __init__(
        self,
        base_tag_acceptor: pynini.Fst,
        meta_registry: Dict[str, MetaLabelDefinition],
        symbol_table: Optional[pynini.SymbolTable] = None
    ):
        """
        Args:
            base_tag_acceptor: The base morpheme/tag sequence acceptor from parC.
            meta_registry: Dictionary of available MetaLabelDefinition configs.
            symbol_table: Shared parC symbol table for tag boundaries.
        """
        self.base_tag_acceptor = base_tag_acceptor.copy()
        self.meta_registry = meta_registry
        self.symbol_table = symbol_table

    def build_slot_mask(self, constraint: FeatureConstraint) -> pynini.Fst:
        """
        Compiles an unanchored feature-slot constraint acceptor:
        F_slot = Sigma* . [slot_name=value] . Sigma*
        """
        # [TECHNICAL DETAIL TO BE FILLED]: Exact delimiter/bracket escaping 
        # syntax used by parC tag formatting (e.g., "[slot=val]" vs "<slot:val>").
        sigma_star = pynini.closure(pynini.union(*[pynini.accep(c) for c in "..."])) # Placeholder
        
        if constraint.mode == MatchMode.ONE_OF:
            slot_patterns = [f"[{constraint.slot_name}={val}]" for val in constraint.values]
            target_fsa = pynini.union(*[pynini.accep(p) for p in slot_patterns])
        elif constraint.mode == MatchMode.EXACT:
            target_fsa = pynini.accep(f"[{constraint.slot_name}={constraint.values[0]}]")
        else:
            raise NotImplementedError(f"Constraint mode {constraint.mode} not supported yet.")

        return pynini.optimize(sigma_star + target_fsa + sigma_star)

    def compile_restricted_tag_acceptor(self, profile: MacroProfile) -> pynini.Fst:
        """
        Intersects parC's base morphotactic tag acceptor with all active meta constraints.
        L_restricted = L_base ∩ F_1 ∩ F_2 ∩ ... ∩ F_n
        """
        restricted_fsa = self.base_tag_acceptor.copy()

        # Collect all active constraints from meta labels
        all_constraints: List[FeatureConstraint] = []
        for meta_id in profile.meta_label_ids:
            meta_def = self.meta_registry[meta_id]
            all_constraints.extend(meta_def.constraints)
        
        # Add runtime dynamic constraints
        all_constraints.extend(profile.dynamic_constraints)

        # Intersect each slot constraint
        for constraint in all_constraints:
            slot_mask = self.build_slot_mask(constraint)
            restricted_fsa = pynini.intersect(restricted_fsa, slot_mask)
            restricted_fsa.optimize()

        return restricted_fsa

    def build_query_lattice(self, surface_form: str, profile: MacroProfile) -> pynini.Fst:
        """
        Constructs the final input query FST:
        Q = accep(surface_form) . L_restricted
        """
        surface_fsa = pynini.accep(surface_form)
        tag_lattice = self.compile_restricted_tag_acceptor(profile)
        
        # [TECHNICAL DETAIL TO BE FILLED]: Root/Stem boundary token or separator 
        # required between surface string and inflection tag domain.
        query_fst = surface_fsa + tag_lattice
        return pynini.optimize(query_fst)

```

---

## 4. Execution Workflow in the Consumer

```python
class LexiconExtractionSession:
    """Orchestrates iterative parsing and dynamic feature discovery."""

    def __init__(self, parc_engine, compiler: MetaConstraintCompiler):
        self.engine = parc_engine
        self.compiler = compiler

    def parse_citation_and_derive_paradigm(self, citation_form: str, other_forms: List[str]):
        # 1. Parse citation form with broad meta labels
        citation_profile = MacroProfile(
            name="citation_pass",
            meta_label_ids=["META_3_PRS"]
        )
        query_lattice = self.compiler.build_query_lattice(citation_form, citation_profile)
        
        # 2. Run parC parse (Q ∘ P)
        parses = self.engine.parse_lattice(query_lattice)
        lexical_profile = self.extract_discovered_features(parses)
        
        # 3. Use discovered aspect class & pronoun set to tightly constrain subsequent forms
        dependent_profile = MacroProfile(
            name="dependent_pass",
            meta_label_ids=["META_1SG_PRS"],
            dynamic_constraints=[
                FeatureConstraint(slot_name="aspect_class", values=[lexical_profile.aspect_class]),
                FeatureConstraint(slot_name="pronoun_set", values=[lexical_profile.pronoun_set])
            ]
        )
        
        # 4. Parse subsequent paradigm forms with zero ambiguity
        for form in other_forms:
            dep_query = self.compiler.build_query_lattice(form, dependent_profile)
            dep_parses = self.engine.parse_lattice(dep_query)
            # Process fully resolved forms...

```

---

## 5. Technical Questions for the `parC` Blueprint Integration Agent

The companion agent inspecting `parC`'s blueprint layer should resolve the following engineering questions:

1. **Tag Acceptor Blueprint Export:**
* Which blueprint class in `parC` generates the canonical morphotactic tag sequence acceptor ($\mathcal{L}_{\text{base}}$), and what method/property exports it as an OpenFST/Pynini object?


2. **Alphabet & Symbol Table Alignment:**
* Does `parC` compile tag tokens as multi-character bracketed byte strings (e.g., `"[tense=prs]"`) or as discrete integer IDs from a dedicated `SymbolTable`? How are token boundaries delimited?


3. **Parse Engine Lattice Ingestion:**
* Does `ParsingEngineBlueprint.build_open_parse_graph()` accept an arbitrary `pynini.Fst` input lattice directly, or is there a specific interface method for passing pre-composed input transducers?


4. **Discovered Feature Extraction Utilities:**
* What helper utilities exist in `parC` for unpacking serialized parse string paths into structured slot-value dictionaries (e.g., isolating `root`, `aspect_class`, and `person_number`)?