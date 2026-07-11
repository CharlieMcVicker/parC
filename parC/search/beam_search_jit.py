"""
src/beam_search_jit.py

Numba JIT implementation of intersect_beam and its helpers.

Drop-in replacement: the public `intersect_beam_jit` function accepts and
returns the same types as the original, so the calling site in beam_search.py
needs no changes beyond the existing `tuple(left)` / `tuple(right)` conversion.

Key design decisions
--------------------
1.  WfsaCsr → 5 individual numpy arrays
    Numba @njit cannot accept Python NamedTuples as typed arguments; the 5
    CSR arrays are unpacked and passed separately.

2.  WfsaCsrBeam → struct-of-arrays (SoA)
    The dataclass has a variable-length `labels: tuple[int, ...]` field which
    has no Numba equivalent.  Instead, beams are represented as parallel 1-D
    arrays (left_states[i], right_states[i], weights[i], finals[i],
    label_lens[i]) and a 2-D labels[i, j] array.  This is the standard HPC
    "array of structs → struct of arrays" transformation.

3.  Fixed MAX_LABEL_LEN / MAX_BEAMS_BUFFER
    Numba requires statically-shaped in-JIT allocations.  128 chars covers all
    practical label sequences; 8192 slots comfortably fits the fan-out even in
    fuzzy mode (O(L×R + L + R) × num_beam per step).

4.  Double-buffered iteration — no dynamic allocation in the hot loop
    The original code appends to Python lists each iteration.  Here two sets of
    pre-allocated SoA buffers are ping-ponged: "current" beams → expand →
    "next" beams, then swap references.

5.  Partial selection sort (top-k)
    `list.sort()` is unavailable in @njit.  A partial O(n × k) selection sort
    finds only the top-k beams without fully sorting the candidate set.  Label
    rows are swapped element-by-element up to max(len_a, len_b), not the full
    MAX_LABEL_LEN.

6.  O(n²) deduplication
    A hash-map with variable-length tuple keys isn't safely JIT-able.  Because
    deduplication runs on already-pruned beams (≤ num_beam entries), O(n²)
    pairwise comparison amounts to at most ~2500 operations — negligible.

7.  Python wrapper (not JIT'd) converts output arrays back to tuples
    The JIT core writes results into pre-allocated output arrays and returns a
    count.  The thin Python wrapper reconstructs the
    (left_state, right_state, path_weight, final, labels_tuple) tuples that
    `WfsaCsrBeam(*result)` can unpack directly.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# ---------------------------------------------------------------------------
# Module-level constants visible to all @njit functions
# ---------------------------------------------------------------------------

MAX_LABEL_LEN: int = 128   # max length of any label path
MAX_BEAMS_BUFFER: int = 32748 #8192  # pre-allocated slot count for beam buffers

# ---------------------------------------------------------------------------
# Module-level working buffers — allocated once at import time.
#
# Passing these into _intersect_beam_core instead of allocating inside it
# avoids ~30 MB of malloc+memset on every call.
#
# NOTE: not thread-safe.  If you need concurrent calls, allocate a separate
# buffer set per thread and pass the appropriate one to intersect_beam_jit.
# ---------------------------------------------------------------------------
_N = MAX_BEAMS_BUFFER

# Ping-pong buffer A (holds "current" beams each iteration)
_a_left      = np.empty(_N, dtype=np.int32)
_a_right     = np.empty(_N, dtype=np.int32)
_a_weight    = np.empty(_N, dtype=np.float32)
_a_final     = np.empty(_N, dtype=np.bool_)
_a_labels    = np.empty((_N, MAX_LABEL_LEN), dtype=np.int32)
_a_label_len = np.empty(_N, dtype=np.int32)

# Ping-pong buffer B (holds "next" beams each iteration)
_b_left      = np.empty(_N, dtype=np.int32)
_b_right     = np.empty(_N, dtype=np.int32)
_b_weight    = np.empty(_N, dtype=np.float32)
_b_final     = np.empty(_N, dtype=np.bool_)
_b_labels    = np.empty((_N, MAX_LABEL_LEN), dtype=np.int32)
_b_label_len = np.empty(_N, dtype=np.int32)

# Accumulator for beams that have reached a final state
_suc_left      = np.empty(_N, dtype=np.int32)
_suc_right     = np.empty(_N, dtype=np.int32)
_suc_weight    = np.empty(_N, dtype=np.float32)
_suc_final     = np.empty(_N, dtype=np.bool_)
_suc_labels    = np.empty((_N, MAX_LABEL_LEN), dtype=np.int32)
_suc_label_len = np.empty(_N, dtype=np.int32)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _labels_equal(
    a_labels: np.ndarray, a_len: int,
    b_labels: np.ndarray, b_len: int,
) -> bool:
    """Return True iff two label rows represent the same sequence."""
    if a_len != b_len:
        return False
    for i in range(a_len):
        if a_labels[i] != b_labels[i]:
            return False
    return True


@njit(cache=True)
def _filter_repeat_beams(
    bs_left:      np.ndarray,
    bs_right:     np.ndarray,
    bs_weight:    np.ndarray,
    bs_final:     np.ndarray,
    bs_labels:    np.ndarray,   # shape (N, MAX_LABEL_LEN)
    bs_label_len: np.ndarray,
    count:        int,
) -> int:
    """
    In-place deduplication of a beam buffer.

    For beams sharing the same label sequence keep only the one with the
    lowest path weight (highest probability).  Uses O(n²) pairwise comparison
    which is fine because n ≤ num_beam (typically single digits).

    Returns the new logical count (entries 0..new_count-1 are valid).
    """
    keep = np.ones(count, dtype=np.bool_)

    for i in range(count):
        if not keep[i]:
            continue
        for j in range(i + 1, count):
            if not keep[j]:
                continue
            if _labels_equal(
                bs_labels[i], bs_label_len[i],
                bs_labels[j], bs_label_len[j],
            ):
                if bs_weight[i] <= bs_weight[j]:
                    keep[j] = False
                else:
                    keep[i] = False
                    break  # i is dead; stop scanning for i's duplicates

    # Compact: shift surviving entries to the front
    new_count = 0
    for i in range(count):
        if keep[i]:
            if new_count != i:
                bs_left[new_count]      = bs_left[i]
                bs_right[new_count]     = bs_right[i]
                bs_weight[new_count]    = bs_weight[i]
                bs_final[new_count]     = bs_final[i]
                bs_label_len[new_count] = bs_label_len[i]
                llen = bs_label_len[i]
                for k in range(llen):
                    bs_labels[new_count, k] = bs_labels[i, k]
            new_count += 1

    return new_count


@njit(cache=True)
def _partial_sort_inplace(
    bs_left:      np.ndarray,
    bs_right:     np.ndarray,
    bs_weight:    np.ndarray,
    bs_final:     np.ndarray,
    bs_labels:    np.ndarray,
    bs_label_len: np.ndarray,
    count:        int,
    top_k:        int,
) -> int:
    """
    Partially sort a beam SoA buffer in-place by ascending path weight,
    placing the `top_k` lightest beams in positions 0..top_k-1.

    Uses selection sort over only the first `top_k` positions — O(count × top_k).
    Label rows are swapped element-by-element up to max(len_a, len_b) rather
    than the full MAX_LABEL_LEN, saving unnecessary work.

    Returns min(count, top_k).
    """
    k = min(count, top_k)

    for i in range(k):
        min_idx = i
        for j in range(i + 1, count):
            if bs_weight[j] < bs_weight[min_idx]:
                min_idx = j

        if min_idx != i:
            # Swap scalar fields
            tmp_l             = bs_left[i]
            bs_left[i]        = bs_left[min_idx]
            bs_left[min_idx]  = tmp_l

            tmp_r             = bs_right[i]
            bs_right[i]       = bs_right[min_idx]
            bs_right[min_idx] = tmp_r

            tmp_w             = bs_weight[i]
            bs_weight[i]      = bs_weight[min_idx]
            bs_weight[min_idx]= tmp_w

            tmp_f             = bs_final[i]
            bs_final[i]       = bs_final[min_idx]
            bs_final[min_idx] = tmp_f

            tmp_ll            = bs_label_len[i]
            bs_label_len[i]   = bs_label_len[min_idx]
            bs_label_len[min_idx] = tmp_ll

            # Swap only the occupied portion of each label row
            swap_len = bs_label_len[i]  # after the scalar swap this is min_idx's original len
            if bs_label_len[min_idx] > swap_len:
                swap_len = bs_label_len[min_idx]
            for k2 in range(swap_len):
                tmp_lbl              = bs_labels[i, k2]
                bs_labels[i, k2]     = bs_labels[min_idx, k2]
                bs_labels[min_idx, k2] = tmp_lbl

    return k


# ---------------------------------------------------------------------------
# Arc-expansion helpers  (one per search mode)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _expand_exact(
    # Current beam
    left_state:    int,
    right_state:   int,
    path_weight:   float,
    beam_labels:   np.ndarray,   # 1-D view of the current beam's label row
    beam_label_len: int,
    # Left FST
    l_offsets: np.ndarray, l_next: np.ndarray,
    l_weights: np.ndarray, l_labels: np.ndarray, l_final: np.ndarray,
    # Right FST
    r_offsets: np.ndarray, r_next: np.ndarray,
    r_weights: np.ndarray, r_labels: np.ndarray, r_final: np.ndarray,
    # Output SoA (write from out_offset onwards)
    out_left:      np.ndarray,
    out_right:     np.ndarray,
    out_weight:    np.ndarray,
    out_final:     np.ndarray,
    out_labels:    np.ndarray,
    out_label_len: np.ndarray,
    out_offset:    int,
) -> int:
    """
    Exact intersection: match arcs with equal labels (arc-sorted assumption
    allows a monotone two-pointer merge instead of O(L×R) nested loops).
    Returns the number of new beams written.
    """
    ls = l_offsets[left_state];   le = l_offsets[left_state + 1]
    rs = r_offsets[right_state];  re = r_offsets[right_state + 1]

    written = 0
    li = 0;  ri = 0
    nl = le - ls;  nr = re - rs

    while li < nl and ri < nr:
        ll = l_labels[ls + li]
        rl = r_labels[rs + ri]

        if ll == rl:
            l_ns = l_next[ls + li];  r_ns = r_next[rs + ri]
            lw   = l_weights[ls + li]; rw = r_weights[rs + ri]
            is_final = l_final[l_ns] and r_final[r_ns]

            idx = out_offset + written
            out_left[idx]      = l_ns
            out_right[idx]     = r_ns
            out_weight[idx]    = path_weight + lw + rw
            out_final[idx]     = is_final
            new_len            = beam_label_len + 1
            out_label_len[idx] = new_len
            for k in range(beam_label_len):
                out_labels[idx, k] = beam_labels[k]
            out_labels[idx, beam_label_len] = ll

            written += 1
            li += 1
            ri += 1

        elif ll < rl:
            li += 1
        else:
            ri += 1

    return written


@njit(cache=True)
def _expand_fuzzy(
    left_state:    int,
    right_state:   int,
    path_weight:   float,
    beam_labels:   np.ndarray,
    beam_label_len: int,
    l_offsets: np.ndarray, l_next: np.ndarray,
    l_weights: np.ndarray, l_labels: np.ndarray, l_final: np.ndarray,
    r_offsets: np.ndarray, r_next: np.ndarray,
    r_weights: np.ndarray, r_labels: np.ndarray, r_final: np.ndarray,
    out_left:      np.ndarray,
    out_right:     np.ndarray,
    out_weight:    np.ndarray,
    out_final:     np.ndarray,
    out_labels:    np.ndarray,
    out_label_len: np.ndarray,
    out_offset:    int,
) -> int:
    """
    Fuzzy intersection weighted by Levenshtein edit distance:
      - substitutions / exact matches  (advance both pointers)
      - deletions                       (advance left only)
      - insertions                      (advance right only)
    Returns the number of new beams written.
    """
    ls = l_offsets[left_state];   le = l_offsets[left_state + 1]
    rs = r_offsets[right_state];  re = r_offsets[right_state + 1]
    nl = le - ls;  nr = re - rs

    written = 0

    # --- substitutions / exact matches ---
    for li in range(nl):
        ll   = l_labels[ls + li]
        l_ns = l_next[ls + li]
        lw   = l_weights[ls + li]
        for ri in range(nr):
            rl   = r_labels[rs + ri]
            r_ns = r_next[rs + ri]
            rw   = r_weights[rs + ri]

            edit_w   = np.float32(0.0) if ll == rl else np.float32(1.0)
            is_final = l_final[l_ns] and r_final[r_ns]

            idx = out_offset + written
            out_left[idx]      = l_ns
            out_right[idx]     = r_ns
            out_weight[idx]    = path_weight + edit_w + lw + rw
            out_final[idx]     = is_final
            new_len            = beam_label_len + 1
            out_label_len[idx] = new_len
            for k in range(beam_label_len):
                out_labels[idx, k] = beam_labels[k]
            out_labels[idx, beam_label_len] = rl
            written += 1

    # --- deletions (advance left, hold right) ---
    for li in range(nl):
        l_ns = l_next[ls + li]
        lw   = l_weights[ls + li]
        is_final = l_final[l_ns] and r_final[right_state]

        idx = out_offset + written
        out_left[idx]      = l_ns
        out_right[idx]     = right_state
        out_weight[idx]    = path_weight + np.float32(1.0) + lw
        out_final[idx]     = is_final
        out_label_len[idx] = beam_label_len
        for k in range(beam_label_len):
            out_labels[idx, k] = beam_labels[k]
        written += 1

    # --- insertions (hold left, advance right) ---
    for ri in range(nr):
        rl   = r_labels[rs + ri]
        r_ns = r_next[rs + ri]
        rw   = r_weights[rs + ri]
        is_final = l_final[left_state] and r_final[r_ns]

        idx = out_offset + written
        out_left[idx]      = left_state
        out_right[idx]     = r_ns
        out_weight[idx]    = path_weight + np.float32(1.0) + rw
        out_final[idx]     = is_final
        new_len            = beam_label_len + 1
        out_label_len[idx] = new_len
        for k in range(beam_label_len):
            out_labels[idx, k] = beam_labels[k]
        out_labels[idx, beam_label_len] = rl
        written += 1

    return written


# ---------------------------------------------------------------------------
# Core JIT loop
# ---------------------------------------------------------------------------


@njit(cache=True)
def _intersect_beam_core(
    # Left FST (CSR arrays in NamedTuple field order: offsets, next, weights, labels, final)
    l_offsets: np.ndarray, l_next: np.ndarray,
    l_weights: np.ndarray, l_labels: np.ndarray, l_final: np.ndarray,
    # Right FST
    r_offsets: np.ndarray, r_next: np.ndarray,
    r_weights: np.ndarray, r_labels: np.ndarray, r_final: np.ndarray,
    # Search parameters
    num_beam:     int,
    fuzzy_search: bool,
    unique_only:  bool,
    # Output SoA (pre-allocated by the Python wrapper, size = num_beam)
    out_left:      np.ndarray,
    out_right:     np.ndarray,
    out_weight:    np.ndarray,
    out_final:     np.ndarray,
    out_labels:    np.ndarray,
    out_label_len: np.ndarray,
    # Working buffers — passed in from module-level to avoid per-call allocation.
    # Ping-pong A:
    a_left: np.ndarray, a_right: np.ndarray, a_weight: np.ndarray,
    a_final: np.ndarray, a_labels: np.ndarray, a_label_len: np.ndarray,
    # Ping-pong B:
    b_left: np.ndarray, b_right: np.ndarray, b_weight: np.ndarray,
    b_final: np.ndarray, b_labels: np.ndarray, b_label_len: np.ndarray,
    # Successful-beams accumulator:
    suc_left: np.ndarray, suc_right: np.ndarray, suc_weight: np.ndarray,
    suc_final: np.ndarray, suc_labels: np.ndarray, suc_label_len: np.ndarray,
) -> int:
    """
    Double-buffered beam search intersection over two WFSA CSR representations.

    Working buffers (a_*, b_*, suc_*) are passed in from module-level
    pre-allocations rather than being created here, eliminating ~30 MB of
    malloc+memset on every call.

    Returns the number of successful beams written (≤ num_beam).
    """
    suc_count = 0

    # --- initialise with the single start beam at (state=0, state=0) ---
    a_left[0]      = 0
    a_right[0]     = 0
    a_weight[0]    = np.float32(0.0)
    a_final[0]     = l_final[0] and r_final[0]
    a_label_len[0] = 0
    cur_count      = 1

    # Use mutable references so we can swap without data copies
    cur_left      = a_left;   nxt_left      = b_left
    cur_right     = a_right;  nxt_right     = b_right
    cur_weight    = a_weight; nxt_weight    = b_weight
    cur_final     = a_final;  nxt_final     = b_final
    cur_labels    = a_labels; nxt_labels    = b_labels
    cur_label_len = a_label_len; nxt_label_len = b_label_len

    while cur_count > 0:
        nxt_count = 0

        # Expand every current beam
        for bi in range(cur_count):
            if fuzzy_search:
                written = _expand_fuzzy(
                    cur_left[bi], cur_right[bi], cur_weight[bi],
                    cur_labels[bi], cur_label_len[bi],
                    l_offsets, l_next, l_weights, l_labels, l_final,
                    r_offsets, r_next, r_weights, r_labels, r_final,
                    nxt_left, nxt_right, nxt_weight, nxt_final,
                    nxt_labels, nxt_label_len, nxt_count,
                )
            else:
                written = _expand_exact(
                    cur_left[bi], cur_right[bi], cur_weight[bi],
                    cur_labels[bi], cur_label_len[bi],
                    l_offsets, l_next, l_weights, l_labels, l_final,
                    r_offsets, r_next, r_weights, r_labels, r_final,
                    nxt_left, nxt_right, nxt_weight, nxt_final,
                    nxt_labels, nxt_label_len, nxt_count,
                )
            nxt_count += written

        if unique_only:
            nxt_count = _filter_repeat_beams(
                nxt_left, nxt_right, nxt_weight, nxt_final,
                nxt_labels, nxt_label_len, nxt_count,
            )

        nxt_count = _partial_sort_inplace(
            nxt_left, nxt_right, nxt_weight, nxt_final,
            nxt_labels, nxt_label_len, nxt_count, num_beam,
        )

        # Accumulate accepting beams
        for bi in range(nxt_count):
            if nxt_final[bi] and suc_count < _N:
                suc_left[suc_count]      = nxt_left[bi]
                suc_right[suc_count]     = nxt_right[bi]
                suc_weight[suc_count]    = nxt_weight[bi]
                suc_final[suc_count]     = nxt_final[bi]
                suc_label_len[suc_count] = nxt_label_len[bi]
                llen = nxt_label_len[bi]
                for k in range(llen):
                    suc_labels[suc_count, k] = nxt_labels[bi, k]
                suc_count += 1

        # Ping-pong: swap buffer references (no data copy)
        cur_left,      nxt_left      = nxt_left,      cur_left
        cur_right,     nxt_right     = nxt_right,     cur_right
        cur_weight,    nxt_weight    = nxt_weight,     cur_weight
        cur_final,     nxt_final     = nxt_final,      cur_final
        cur_labels,    nxt_labels    = nxt_labels,     cur_labels
        cur_label_len, nxt_label_len = nxt_label_len,  cur_label_len
        cur_count = nxt_count

    # --- finalise successful-beam list ---
    suc_count = _partial_sort_inplace(
        suc_left, suc_right, suc_weight, suc_final,
        suc_labels, suc_label_len, suc_count, num_beam,
    )
    if unique_only:
        suc_count = _filter_repeat_beams(
            suc_left, suc_right, suc_weight, suc_final,
            suc_labels, suc_label_len, suc_count,
        )
        if suc_count > num_beam:
            suc_count = num_beam

    result_count = min(suc_count, num_beam)

    for i in range(result_count):
        out_left[i]      = suc_left[i]
        out_right[i]     = suc_right[i]
        out_weight[i]    = suc_weight[i]
        out_final[i]     = suc_final[i]
        out_label_len[i] = suc_label_len[i]
        llen = suc_label_len[i]
        for k in range(llen):
            out_labels[i, k] = suc_labels[i, k]

    return result_count


# ---------------------------------------------------------------------------
# Public Python wrapper  (NOT @njit — handles object conversion)
# ---------------------------------------------------------------------------


def intersect_beam_jit(
    left:         tuple,
    right:        tuple,
    num_beam:     int  = 5,
    fuzzy_search: bool = False,
    unique_only:  bool = False,
) -> list[tuple]:
    """
    JIT-accelerated beam search intersection.

    Parameters
    ----------
    left, right
        Plain 5-tuples of numpy arrays produced by ``tuple(WfsaCsr(...))``.
        Field order must match WfsaCsr: (offsets, next_states, weights, labels, final).
    num_beam
        Maximum number of hypotheses to keep at each step and to return.
    fuzzy_search
        If True, use Levenshtein-weighted expansion instead of exact matching.
    unique_only
        If True, deduplicate hypotheses with identical label sequences.

    Returns
    -------
    list of tuples
        Each element is ``(left_state, right_state, path_weight, final, labels_tuple)``
        — directly unpackable into ``WfsaCsrBeam(*result)``.
    """
    # Unpack CSR tuples (field order: offsets, next_states, weights, labels, final)
    l_offsets, l_next, l_weights, l_labels, l_final = left
    r_offsets, r_next, r_weights, r_labels, r_final = right

    # Ensure dtypes expected by the JIT core
    l_offsets  = np.asarray(l_offsets,  dtype=np.int32)
    l_next     = np.asarray(l_next,     dtype=np.int32)
    l_weights  = np.asarray(l_weights,  dtype=np.float32)
    l_labels   = np.asarray(l_labels,   dtype=np.int32)
    l_final    = np.asarray(l_final,    dtype=np.bool_)

    r_offsets  = np.asarray(r_offsets,  dtype=np.int32)
    r_next     = np.asarray(r_next,     dtype=np.int32)
    r_weights  = np.asarray(r_weights,  dtype=np.float32)
    r_labels   = np.asarray(r_labels,   dtype=np.int32)
    r_final    = np.asarray(r_final,    dtype=np.bool_)

    # Pre-allocate output buffers
    out_left      = np.empty(num_beam, dtype=np.int32)
    out_right     = np.empty(num_beam, dtype=np.int32)
    out_weight    = np.empty(num_beam, dtype=np.float32)
    out_final     = np.empty(num_beam, dtype=np.bool_)
    out_labels    = np.zeros((num_beam, MAX_LABEL_LEN), dtype=np.int32)
    out_label_len = np.zeros(num_beam, dtype=np.int32)

    count = _intersect_beam_core(
        l_offsets, l_next, l_weights, l_labels, l_final,
        r_offsets, r_next, r_weights, r_labels, r_final,
        num_beam, fuzzy_search, unique_only,
        out_left, out_right, out_weight, out_final, out_labels, out_label_len,
        # Module-level working buffers — no allocation on this call path
        _a_left, _a_right, _a_weight, _a_final, _a_labels, _a_label_len,
        _b_left, _b_right, _b_weight, _b_final, _b_labels, _b_label_len,
        _suc_left, _suc_right, _suc_weight, _suc_final, _suc_labels, _suc_label_len,
    )

    # Reconstruct Python tuples — WfsaCsrBeam(*result) unpacks cleanly
    results: list[tuple] = []
    for i in range(count):
        llen         = int(out_label_len[i])
        labels_tuple = tuple(int(out_labels[i, j]) for j in range(llen))
        results.append((
            int(out_left[i]),
            int(out_right[i]),
            float(out_weight[i]),
            bool(out_final[i]),
            labels_tuple,
        ))

    return results