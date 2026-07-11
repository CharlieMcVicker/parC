from typing import NamedTuple
import pynini
import numpy as np
from dataclasses import dataclass
from parC.search.beam_search_jit import intersect_beam_jit
from parC.search.edit_graph import ascii_table


class WfsaCsr(NamedTuple):
    """
    Compressed Sparse Row representation of a WFSA with m states and n arcs.
    - `offsets`: n+1 array mapping state index to the first arc index
    - `final`: n array indicating final states
    - `next_states`: m array mapping arc index to target state index
    - `weights`: m array containing arc weights
    - `labels`: m array containing input labels for arcs
    """

    offsets: np.ndarray
    next_states: np.ndarray
    weights: np.ndarray
    labels: np.ndarray
    final: np.ndarray

    def to_dict(self) -> dict[str, np.ndarray]:
        return {
            "offsets": self.offsets,
            "next_states": self.next_states,
            "weights": self.weights,
            "labels": self.labels,
            "final": self.final,
        }

    @classmethod
    def from_pynini(cls, wfsa: pynini.Fst) -> "WfsaCsr":
        """
        Compute a Compressed Sparse Row representation for the given FST.

        Arguments:
        - `fst`: the input FST to convert
        Returns:
        A `CSR` named tuple containing the CSR representation of the FST.
        """

        # beam search will expect an epsilon-free deterministic arc-sorted WFSA
        wfsa = pynini.rmepsilon(wfsa)
        wfsa = pynini.determinize(wfsa)
        wfsa = pynini.arcsort(wfsa)

        n = wfsa.num_states()

        # store final states
        final = np.zeros(n, dtype=bool)
        zero_weight = pynini.Weight.zero(wfsa.weight_kind())

        # first compute offsets by counting the number of arcs leaving each state
        # check for final weights in same loop
        offsets = np.zeros(n + 1, dtype=np.int32)
        for s in wfsa.states():
            if wfsa.final(s) != zero_weight:
                final[s] = True
            offsets[s + 1] = wfsa.num_arcs(s)

        # then compute the prefix sum to get the starting index of arcs for each state
        np.cumsum(offsets, out=offsets)

        # the total number of arcs is the last value in offsets
        # (equivalent to the 'starting arc' of the final state, which has no outgoing arcs)
        total = int(offsets[-1])

        # arc arrays
        next_states = np.empty(total, dtype=np.int32)
        weights = np.empty(total, dtype=np.float32)
        labels = np.empty(total, dtype=np.int32)

        # populate the arc arrays by iterating over states and arcs
        for s in wfsa.states():
            i = offsets[s]
            for arc in wfsa.arcs(s):
                next_states[i] = arc.nextstate
                weights[i] = float(arc.weight)  # tropical -> float
                labels[i] = arc.ilabel
                i += 1

        return cls(
            offsets=offsets,
            final=final,
            next_states=next_states,
            weights=weights,
            labels=labels,
        )


@dataclass
class WfsaCsrBeam:
    """
    Dataclass representing a single beam search hypothesis.
    """

    left_state: int
    right_state: int
    path_weight: float
    final: bool
    labels: tuple[int, ...]

    def __str__(self) -> str:
        labels = [ascii_table.find(lbl) for lbl in self.labels]
        label_str = "".join(labels) or "<eps>"
        final = " final" if self.final else ""
        state_name = f"({self.left_state},{self.right_state})"
        return f"WfsaCsrBeam(Q={state_name}, w={self.path_weight}, label={label_str}{final})"

    def __repr__(self) -> str:
        return self.__str__()


def decode_beam(beam: WfsaCsrBeam) -> tuple[str, float]:
    decoded_labels = [ascii_table.find(label) for label in beam.labels]
    string = "".join(decoded_labels)
    return string, float(beam.path_weight)


def get_next_beams(
    beam: WfsaCsrBeam, left: WfsaCsr, right: WfsaCsr
) -> list[WfsaCsrBeam]:
    next_beams = []

    left_start_arc = left.offsets[beam.left_state]
    left_end_arc = left.offsets[beam.left_state + 1]

    right_start_arc = right.offsets[beam.right_state]
    right_end_arc = right.offsets[beam.right_state + 1]

    left_labels = left.labels[left_start_arc:left_end_arc]
    right_labels = right.labels[right_start_arc:right_end_arc]

    # since arcs are sorted, find matching labels
    # by checking for matches monotonically
    max_arcs = max(len(left_labels), len(right_labels))

    i = 0
    left_i = 0
    right_i = 0
    while (
        (i < max_arcs) and (left_i < len(left_labels)) and (right_i < len(right_labels))
    ):
        left_label = left_labels[left_i]
        right_label = right_labels[right_i]
        if left_label == right_label:
            left_next_state = left.next_states[left_start_arc + left_i]
            right_next_state = right.next_states[right_start_arc + right_i]

            left_weight = left.weights[left_start_arc + left_i]
            right_weight = right.weights[right_start_arc + right_i]

            is_final = left.final[left_next_state] and right.final[right_next_state]

            curr_beam = WfsaCsrBeam(
                left_state=left_next_state,
                right_state=right_next_state,
                path_weight=beam.path_weight + left_weight + right_weight,
                labels=beam.labels + (left_label.item(),),
                final=is_final,
            )
            next_beams.append(curr_beam)

            left_i += 1
            right_i += 1

        elif left_label < right_label:
            left_i += 1
        else:
            # right_label < left_label
            right_i += 1

    return next_beams


def get_next_beams_fuzzy(
    beam: WfsaCsrBeam,
    left: WfsaCsr,
    right: WfsaCsr,
    cost_matrix: np.ndarray | None = None,
) -> list[WfsaCsrBeam]:
    """
    Computes next beams allowing for inexact matches,
    weighted by naive Levenshtein edit distance or
    weighted distance when `cost_matrix` is provided.

    If `cost_matrix` is provided, it should be a 2D array where
    `cost_matrix[i, j]` represents the cost of substituting character
    `i` with character `j`, and the zero-th row and column represent the
    cost of insertions and deletions, respectively (i.e. substituting the
    blank token).
    """
    left_start_arc = left.offsets[beam.left_state]
    left_end_arc = left.offsets[beam.left_state + 1]

    right_start_arc = right.offsets[beam.right_state]
    right_end_arc = right.offsets[beam.right_state + 1]

    left_labels = left.labels[left_start_arc:left_end_arc]
    right_labels = right.labels[right_start_arc:right_end_arc]

    # consider all possible matches and substitutions

    matches_and_subs = matches_and_substitutions(
        beam,
        left,
        right,
        cost_matrix,
        left_start_arc,
        right_start_arc,
        left_labels,
        right_labels,
    )

    # consider deletions (of left language)
    delete_beams = deletions(
        beam, left, right, cost_matrix, left_start_arc, left_labels
    )

    # consider insertions (of left language)
    insert_beams = insertions(
        beam, left, right, cost_matrix, right_start_arc, right_labels
    )

    return matches_and_subs + delete_beams + insert_beams


def insertions(beam, left, right, cost_matrix, right_start_arc, right_labels):
    insert_beams = []
    for right_i, right_label in enumerate(right_labels):
        if cost_matrix is not None:
            # cost of substituting blank with right_label
            insert_weight = cost_matrix[0, right_label]
        else:
            insert_weight = 1

        right_next_state = right.next_states[right_start_arc + right_i]
        right_weight = right.weights[right_start_arc + right_i]

        is_final = left.final[beam.left_state] and right.final[right_next_state]
        hypothesis_weight = beam.path_weight + insert_weight + right_weight

        hypothesis = WfsaCsrBeam(
            left_state=beam.left_state,
            right_state=right_next_state,
            path_weight=hypothesis_weight,
            labels=beam.labels + (right_label.item(),),
            final=is_final,
        )
        insert_beams.append(hypothesis)
    return insert_beams


def deletions(
    beam: WfsaCsrBeam,
    left: WfsaCsr,
    right: WfsaCsr,
    cost_matrix: np.ndarray | None,
    left_start_arc: int,
    left_labels: np.ndarray,
) -> list[WfsaCsrBeam]:
    delete_beams = []
    for left_i, left_label in enumerate(left_labels):
        if cost_matrix is not None:
            # cost of substituting left_label with blank
            delete_weight = cost_matrix[left_label, 0]
        else:
            delete_weight = 1

        left_next_state = left.next_states[left_start_arc + left_i]
        left_weight = left.weights[left_start_arc + left_i]

        is_final = left.final[left_next_state] and right.final[beam.right_state]
        hypothesis_weight = beam.path_weight + delete_weight + left_weight

        hypothesis = WfsaCsrBeam(
            left_state=left_next_state,
            right_state=beam.right_state,
            path_weight=hypothesis_weight,
            labels=beam.labels,
            final=is_final,
        )
        delete_beams.append(hypothesis)
    return delete_beams


def matches_and_substitutions(
    beam: WfsaCsrBeam,
    left: WfsaCsr,
    right: WfsaCsr,
    cost_matrix: np.ndarray | None,
    left_start_arc: int,
    right_start_arc: int,
    left_labels: np.ndarray,
    right_labels: np.ndarray,
) -> list[WfsaCsrBeam]:
    match_and_sub_beams = []
    for left_i, left_label in enumerate(left_labels):
        for right_i, right_label in enumerate(right_labels):
            if cost_matrix is not None:
                edit_weight = cost_matrix[left_label, right_label]
            else:
                edit_weight = 1 if left_label != right_label else 0

            left_next_state = left.next_states[left_start_arc + left_i]
            left_weight = left.weights[left_start_arc + left_i]

            right_next_state = right.next_states[right_start_arc + right_i]
            right_weight = right.weights[right_start_arc + right_i]

            hypothesis_weight = (
                beam.path_weight + edit_weight + left_weight + right_weight
            )
            is_final = left.final[left_next_state] and right.final[right_next_state]

            hypothesis = WfsaCsrBeam(
                left_state=left_next_state,
                right_state=right_next_state,
                path_weight=hypothesis_weight,
                final=is_final,
                labels=beam.labels + (right_label.item(),),
            )
            match_and_sub_beams.append(hypothesis)
    return match_and_sub_beams


def filter_repeat_beams(beams: list[WfsaCsrBeam]) -> list[WfsaCsrBeam]:
    """
    Return filtered list of beams where for multiple beams containing the
    same label sequence, only the beam with lowest weight (highest probability)
    is kept.
    """

    # map label sequence to beam
    label2beam: dict[tuple[int, ...], WfsaCsrBeam] = {}

    for beam in beams:
        if beam.labels not in label2beam:
            label2beam[beam.labels] = beam
        elif label2beam[beam.labels].path_weight > beam.path_weight:
            # override previous beam if current has lower weight
            label2beam[beam.labels] = beam
        else:
            # previous beam has lower or equal weight, do nothing
            pass

    return list(label2beam.values())


def intersect_beam(
    left: WfsaCsr,
    right: WfsaCsr,
    num_beam: int = 5,
    fuzzy_search: bool = False,
    unique_only: bool = False,
    use_jit: bool = False,
    cost_matrix: np.ndarray | None = None,
) -> list[WfsaCsrBeam]:
    """
    Compose left FstCsr with right, pruning the top `num_beams`
    paths.
    """

    if use_jit:
        left = tuple(left)
        right = tuple(right)
        results = intersect_beam_jit(
            left=left,
            right=right,
            num_beam=num_beam,
            fuzzy_search=fuzzy_search,
            unique_only=unique_only,
        )
        beams = [WfsaCsrBeam(*result) for result in results]
        return beams

    # initialize w/ single beam starting at initial state

    start_state_is_final = left.final[0] and right.final[0]
    initial_beam = WfsaCsrBeam(
        left_state=0,
        right_state=0,
        path_weight=0.0,
        labels=tuple(),
        final=start_state_is_final,
    )
    beams: list[WfsaCsrBeam] = [initial_beam]
    successful_beams: list[WfsaCsrBeam] = []

    while beams:
        next_beams: list[WfsaCsrBeam] = []

        # get possible continuations per previous beam
        for beam in beams:
            if fuzzy_search:
                beams_from_current = get_next_beams_fuzzy(
                    beam, left, right, cost_matrix
                )
            else:
                beams_from_current = get_next_beams(beam, left, right)
            beams_from_current.sort(key=lambda b: b.path_weight)
            next_beams.extend(beams_from_current)

        if unique_only:
            # exclude beams with repeat labels
            next_beams = filter_repeat_beams(next_beams)

        # sort by path weight and trim to number of beams
        next_beams.sort(key=lambda b: b.path_weight)
        next_beams = next_beams[:num_beam]

        # extend successful_beams (if applicable)
        successful_beams.extend(beam for beam in next_beams if beam.final)
        beams = next_beams

    successful_beams.sort(key=lambda b: b.path_weight)
    if unique_only:
        # check again for repeats
        successful_beams = filter_repeat_beams(successful_beams)

    successful_beams = successful_beams[:num_beam]

    return successful_beams


def intersect_beam_forward_back(
    left_forward: WfsaCsr,
    right_forward: WfsaCsr,
    left_backward: WfsaCsr,
    right_backward: WfsaCsr,
    num_beam: int = 5,
    fuzzy_search: bool = False,
    unique_only: bool = False,
    use_jit: bool = False,
    cost_matrix: np.ndarray | None = None,
) -> list[WfsaCsrBeam]:
    """
    Wrapper over `intersect_beam` that performs beam search in both forward
    and backward directions, then merges results by intersecting the resulting
    beams. This allows for better recall of the top `num_beam` results,
    since a path that is pruned in the forward search might be kept in the backward search,
    and vice versa.
    """
    forward_hypotheses = intersect_beam(
        left=left_forward,
        right=right_forward,
        num_beam=num_beam,
        fuzzy_search=fuzzy_search,
        unique_only=unique_only,
        use_jit=use_jit,
        cost_matrix=cost_matrix,
    )
    backward_hypotheses = intersect_beam(
        left=left_backward,
        right=right_backward,
        num_beam=num_beam,
        fuzzy_search=fuzzy_search,
        unique_only=unique_only,
        use_jit=use_jit,
        cost_matrix=cost_matrix,
    )
    # reverse label sequences in backward hypotheses to match forward direction
    backward_hypotheses = [
        WfsaCsrBeam(
            left_state=beam.left_state,
            right_state=beam.right_state,
            path_weight=beam.path_weight,
            final=beam.final,
            labels=tuple(reversed(beam.labels)),
        )
        for beam in backward_hypotheses
    ]

    # merge forward and backward hypotheses by intersecting label sequences
    hypotheses_filtered = filter_repeat_beams(forward_hypotheses + backward_hypotheses)
    hypotheses_filtered.sort(key=lambda b: b.path_weight)
    return hypotheses_filtered[:num_beam]


if __name__ == "__main__":
    test = pynini.union(
        *[
            "foo",
            "for",
        ]
    )
    query = pynini.accep("fo")

    query_csr = WfsaCsr.from_pynini(query)
    test_csr = WfsaCsr.from_pynini(test)

    result = intersect_beam(
        left=query_csr,
        right=test_csr,
        fuzzy_search=True,
        unique_only=True,
    )
    decoded = [decode_beam(res) for res in result]

    result_jit = intersect_beam(
        left=query_csr,
        right=test_csr,
        fuzzy_search=True,
        unique_only=True,
        use_jit=True,
    )
    decoded_jit = [decode_beam(res) for res in result_jit]
    breakpoint()
