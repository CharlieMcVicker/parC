# parC

**parC** is a toolkit for building and applying morphological analyzers (finite-state-transducer-based parsers) from linguistic fieldwork data.

---

## Overview

`parC` processes language grammar specifications defined in structured YAML configuration files into executable Finite State Transducers (FSTs) using `pynini`.

---

## Ports & Adapters Architecture & 5-Layer FST Blueprints

`parC` follows a strict **Ports & Adapters (Hexagonal Architecture)** design:

1. **Upfront I/O Phase (Adapters / Loaders)**: `config_loader.py` reads all inventory, feature maps, patterns, rules, and paradigm marker mappings upfront into strongly-typed `GrammarConfig` and `ParadigmConfig` dataclasses.
2. **Pure Blueprint Domain Layer**: Domain Blueprint objects (`parC/grammar/blueprints/`) are constructed with **pure, strictly-typed constructors** with zero disk I/O side effects.
3. **In-Memory FST Composition Engine**: FST-producing methods operate purely on in-memory domain models to build atomic sub-FSTs and compose them hierarchically.

```mermaid
flowchart TD
    subgraph PortsAdapters["Upfront I/O Adapter (config_loader.py)"]
        CL["load_grammar_config() / load_paradigm_config()"] -->|GrammarConfig / ParadigmConfig| PureLayer
    end

    subgraph PureLayer["Pure Blueprint Domain Layer"]
        subgraph Layer1["Layer 1: Alphabet & Symbol Space"]
            L1["AlphabetBlueprint"] -->|symbol tables & special FSAs| L1_Out["SymbolTable / Acceptors"]
        end

        subgraph Layer2["Layer 2: Pattern Regex Acceptors"]
            L2["PatternLibraryBlueprint"] -->|named pattern acceptors| L2_Out["Pattern FST Dictionary"]
        end

        subgraph Layer3["Layer 3: Phonological & Exponence Operations"]
            L3a["RulePipelineBlueprint"] -->|cdrewrite transducers| L3_Out["Rule Rewrite FSTs"]
            L3b["MarkerLibraryBlueprint"] -->|morpheme exponence| L3_Out2["Marker Transducers"]
        end

        subgraph Layer4["Layer 4: Morphotactic Stage Cascade"]
            L4["StageCascadeBlueprint"] -->|stage-gated composition| L4_Out["Open Inflection Graph FST"]
        end

        subgraph Layer5["Layer 5: Inversion & Search Engine"]
            L5["ParsingEngineBlueprint"] -->|inversion & edit lattice| L5_Out["Open Parse & Search Lattice FSTs"]
        end
    end

    Layer1 --> Layer2
    Layer1 --> Layer3
    Layer2 --> Layer3
    Layer3 --> Layer4
    Layer4 --> Layer5
```

### Blueprints Summary

- **Layer 1: `AlphabetBlueprint` (`parC/grammar/blueprints/alphabet.py`)**  
  Pure domain wrapper for phone inventory, feature tags (`[feat=val]`), edit symbols, boundary markers, and `pynini.SymbolTable`.
- **Layer 2: `PatternLibraryBlueprint` (`parC/grammar/blueprints/patterns.py`)**  
  Compiles named phonological pattern regex acceptors (`<Consonants>`, `<Vowel>`) using recursive descent. Receives `AlphabetBlueprint` via method injection.
- **Layer 3: `RulePipelineBlueprint` & `MarkerLibraryBlueprint` (`parC/grammar/blueprints/transducers.py`)**  
  Compiles phonological rewrite rules (`pynini.cdrewrite`) and morphological exponence markers into reusable sub-FSTs.
- **Layer 4: `StageCascadeBlueprint` (`parC/grammar/blueprints/paradigms.py`)**  
  Consumes pre-loaded `ParadigmConfig` and lower-layer blueprints to construct stage-gated realization transducers and sequential composition cascades.
- **Layer 5: `ParsingEngineBlueprint` (`parC/grammar/blueprints/parser.py`)**  
  Inverts inflection graphs into parse graphs and constructs fuzzy search lattices.

---

## Development

- **Validate Schemas**: `PYTHONPATH=. python -m parC.yaml_utils.schema_validation`
- **Run Tests**: `pytest`
