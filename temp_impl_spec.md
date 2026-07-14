## Architecture Overview

The morphological analyzer is decomposed into two discrete, uncomposed Finite State Transducers (FSTs) to isolate morphosyntax from phonological realization, eliminating compile-time state-space explosion. To achieve efficient parsing via inversion without runtime memory spikes, an intermediate static acceptor acts as a constraint filter.

---

## Implementation Pipeline & Phases

### 1. Definition of Spaces

- **Root Domain:** Modeled as an unconstrained language ($\Sigma^*$) matching phonotactic segments, preventing vocabulary-trie bloat.
- **Feature Tail:** All morphological and rule features are positioned in a strict, linear sequence appended _after_ the root and outside the end-of-word (`[EoW]`) boundary token.
- **Feature Preservation:** Morphological features are carried through the intermediate layer via explicit, un-dropped "discharge tags" (e.g., `[+Feature+Discharged]`), preserving full constraint information.

### 2. Compilation Phases

#### Phase 1: Morphological Mapping ($FST_1$)

- **Operation:** Maps underlying morphosyntactic feature bundles to phonological rule labels.
- **Topology:** Identity loops on the $\Sigma^*$ root space, branching into a shallow lookup tree post-`[EoW]`.

#### Phase 2: Filter Generation ($\text{Filter}$)

- **Operation:** $\text{Filter} = \text{Minimize}(\text{Determinize}(\text{Project}_{\text{output}}(FST_1)))$
- **Topology:** A single-state loop for the root phonemes leading to a highly compact, deterministic directed acyclic graph (DAG) representing valid rule-label combinations.

#### Phase 3: Phonological Cascades ($FST_2$)

- **Operation:** Applies the cascade of ~500 context-sensitive phonological rewriting rules to the root string.
- **Optimization:** Rules within the same structural stratum that exhibit no feeding/bleeding relationships are grouped via **union** ($\cup$) into parallel phase blocks before being sequentially composed ($\circ$) into the final $FST_2$ chain.
- **Tag Realization:** Rule labels post-`[EoW]` are mapped via identity or dropped to $\epsilon$ depending on whether the target parser variant requires feature retention on the surface form.

---

## Runtime Parsing Execution (Inversion)

To parse a surface string $w$, the evaluation pipeline executes via lazy composition or runtime intersection in the following sequence:

$$\text{Lexical Features} = w \circ \left( \text{Filter} \circ FST_2^{-1} \right) \circ FST_1^{-1}$$

> **Note on Optimization:** Pre-composing $\text{Filter} \circ FST_2^{-1}$ at compile time converts the unbounded epsilon-insertions of context-free feature dropping into a tightly constrained intersection, eliminating intermediate lattice bloat during runtime lookup.
