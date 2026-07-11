import numpy as np
import numba as nb

"""
Duplicate functions from `beam_search.py`
using only native Python types to enable JIT compilation.

WfsaCsr is represented as a 5-ple:
    (offsets, next_states, weights, labels, final)

WfsaCsrBeam is represented as a 5-ple:
    (left_state, right_state, path_weight, labels, final)
"""

# named tuple indices for readbility

# WfsaCsr only
offsets_index: int = 0
next_states_index: int = 1

# WfsaCsrBeam only
left_state_index: int = 0
right_state_index: int = 1

# Shared indices
weights_index: int = 2
labels_index: int = 3
final_index: int = 4

@nb.jit(nopython=True)
def get_next_beams_jit(
    beam: tuple, left: tuple, right: tuple
) -> list[tuple]:
    next_beams = []

    left_start_arc = left[offsets_index][beam[left_state_index]]
    left_end_arc = left[offsets_index][beam[left_state_index] + 1]

    right_start_arc = right[offsets_index][beam[right_state_index]]
    right_end_arc = right[offsets_index][beam[right_state_index] + 1]

    left_labels = left[labels_index][left_start_arc:left_end_arc]
    right_labels = right[labels_index][right_start_arc:right_end_arc]

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
            left_next_state = left[next_states_index][left_start_arc + left_i]
            right_next_state = right[next_states_index][right_start_arc + right_i]

            left_weight = left[weights_index][left_start_arc + left_i]
            right_weight = right[weights_index][right_start_arc + right_i]

            is_final = left[final_index][left_next_state] and right[final_index][right_next_state]

            curr_beam = tuple(
                left_next_state,
                right_next_state,
                beam[weights_index] + left_weight + right_weight,
                beam[labels_index] + (left_label.item(),),
                is_final,
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

@nb.jit(nopython=True)
def get_next_beams_fuzzy_jit(
    beam: tuple, left: tuple, right: tuple
) -> list[tuple]:
    """
    Computes next beams allowing for inexact matches,
    weighted by Levenshtein edit distance
    """
    next_beams = []

    left_start_arc = left[offsets_index][beam[left_state_index]]
    left_end_arc = left[offsets_index][beam[left_state_index] + 1]

    right_start_arc = right[offsets_index][beam[right_state_index]]
    right_end_arc = right[offsets_index][beam[right_state_index] + 1]

    left_labels = left[labels_index][left_start_arc:left_end_arc]
    right_labels = right[labels_index][right_start_arc:right_end_arc]

    # consider all possible matches and substitutions
    for left_i, left_label in enumerate(left_labels):
        for right_i, right_label in enumerate(right_labels):
            edit_weight = 1 if left_label != right_label else 0

            left_next_state = left[next_states_index][left_start_arc + left_i]
            left_weight = left[weights_index][left_start_arc + left_i]

            right_next_state = right[next_states_index][right_start_arc + right_i]
            right_weight = right[weights_index][right_start_arc + right_i]

            hypothesis_weight = (
                beam[weights_index] + edit_weight + left_weight + right_weight
            )
            is_final = left[final_index][left_next_state] and right[final_index][right_next_state]

            hypothesis = (
                left_next_state,
                right_next_state,
                hypothesis_weight,
                beam[labels_index] + (right_label.item(),),
                is_final,
            )
            next_beams.append(hypothesis)

    # consider deletions (of left language)
    for left_i, left_label in enumerate(left_labels):
        delete_weight = 1

        left_next_state = left[next_states_index][left_start_arc + left_i]
        left_weight = left[weights_index][left_start_arc + left_i]

        is_final = left[final_index][left_next_state] and right[final_index][beam[right_state_index]]
        hypothesis_weight = beam[weights_index] + delete_weight + left_weight

        hypothesis = (
            left_next_state,
            beam[right_state_index],
            hypothesis_weight,
            beam[labels_index],
            is_final,
        )
        next_beams.append(hypothesis)

    # consider insertions (of left language)
    for right_i, right_label in enumerate(right_labels):
        delete_weight = 1

        right_next_state = right[next_states_index][right_start_arc + right_i]
        right_weight = right[weights_index][right_start_arc + right_i]

        is_final = left[final_index][beam[left_state_index]] and right[final_index][right_next_state]
        hypothesis_weight = beam[weights_index] + delete_weight + right_weight

        hypothesis = (
            beam[left_state_index],
            right_next_state,
            hypothesis_weight,
            beam[labels_index] + (right_label.item(),),
            is_final,
        )
        next_beams.append(hypothesis)

    return next_beams

@nb.jit(nopython=True)
def filter_repeat_beams_jit(beams: list[tuple]) -> list[tuple]:
    """
    Return filtered list of beams where for multiple beams containing the
    same label sequence, only the beam with lowest weight (highest probability)
    is kept.
    """

    # map label sequence to beam
    label2beam: dict[tuple[int, ...], tuple] = {}

    for beam in beams:
        beam_labels = beam[labels_index]
        if beam_labels not in label2beam:
            label2beam[beam_labels] = beam
        elif label2beam[beam_labels][weights_index] > beam[weights_index]:
            # override previous beam if current has lower weight
            label2beam[beam_labels] = beam
        else:
            # previous beam has lower or equal weight, do nothing
            pass

    return list(label2beam.values())

@nb.jit(nopython=True, debug=True)
def intersect_beam_jit(
    left: tuple[np.ndarray],
    right: tuple[np.ndarray],
    num_beam: int = 5,
    fuzzy_search: bool = False,
    unique_only: bool = False,
) -> list[tuple]:
    """
    Replication of `intersect_beam` using only native Python types
    to enable JIT compilation.
    """
    # initialize w/ single beam starting at initial state

    start_state_is_final = left[final_index][0] and right[final_index][0]
    # shadow `WfsaCsrBeam` using a vanilla tuple
    initial_beam = tuple([
        0,                      # initial left state
        0,                      # initial right state
        0.0,                    # initial path weight
        tuple(),                # initial labels
        start_state_is_final,   # whether initial state is also final
    ])
    beams: list[tuple] = [initial_beam]
    successful_beams: list[tuple] = []

    while beams:
        next_beams: list[tuple] = []

        # get possible continuations per previous beam
        for beam in beams:
            if fuzzy_search:
                beams_from_current = get_next_beams_fuzzy_jit(beam, left, right)
            else:
                beams_from_current = get_next_beams_jit(beam, left, right)
            beams_from_current.sort(key=lambda b: b[weights_index])
            # avoid using 'list.extend' which uses a generator 
            for b in beams_from_current:
                next_beams.append(b)

        if unique_only:
            # exclude beams with repeat labels
            next_beams = filter_repeat_beams_jit(next_beams)

        # sort by path weight and trim to number of beams
        next_beams.sort(key=lambda b: b[weights_index])
        next_beams = next_beams[:num_beam]

        # extend successful_beams (if applicable)
        for beam in next_beams:
            # beam is successful if beam[final_index] is truthy
            if not beam[final_index]:
                continue
            successful_beams.append(beam)
        beams = next_beams

    successful_beams.sort(key=lambda b: b[weights_index])
    if unique_only:
        # check again for repeats
        successful_beams = filter_repeat_beams_jit(successful_beams)

    successful_beams = successful_beams[:num_beam]

    return successful_beams