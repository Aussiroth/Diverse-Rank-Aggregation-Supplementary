"""
Shared utilities for diverse footrule rank aggregation.

Public API is unchanged with respect to the original `utils.py`:

    find_tight_pairs, max_weight_perfect_matching, footrule_distance,
    sum_dispersion, min_dispersion, is_prime, next_prime,
    invert_matrix_mod_p, determinant_mod_p

Additions (all optional, nothing existing was renamed):

    prepare_instance, build_cost_matrix, rankings_to_matrix,
    hungarian_potentials, matchings_to_ranking, is_permutation,
    determinants_mod_p_batch, mod_matmul, INT64_SAFE_PRIME_LIMIT
"""

from __future__ import annotations

import math
from itertools import combinations

import numpy as np
from scipy.optimize import linprog, linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "find_tight_pairs",
    "max_weight_perfect_matching",
    "footrule_distance",
    "sum_dispersion",
    "min_dispersion",
    "is_prime",
    "next_prime",
    "invert_matrix_mod_p",
    "determinant_mod_p",
    "prepare_instance",
    "build_cost_matrix",
    "rankings_to_matrix",
    "hungarian_potentials",
    "prune_to_matching_covered",
    "matchings_to_ranking",
    "is_permutation",
    "determinants_mod_p_batch",
    "mod_matmul",
    "INT64_SAFE_PRIME_LIMIT",
]


# ---------------------------------------------------------------------------
# Instance construction
# ---------------------------------------------------------------------------

def rankings_to_matrix(rankings, candidates):
    """Stack a list of ``{candidate: rank}`` dicts into a ``(t, n)`` float array."""
    return np.asarray(
        [[r[c] for c in candidates] for r in rankings], dtype=float
    )


def build_cost_matrix(rankings, candidates, positions, block=256):
    """
    c[i, p] = sum_j |rank_j(candidate_i) - position_p|.

    Identical values to the original triple ``for`` loop, but computed with
    broadcasting.  Voters are processed in blocks so peak memory stays at
    ``block * n * n`` instead of ``t * n * n``.
    """
    R = rankings_to_matrix(rankings, candidates)            # (t, n)
    P = np.asarray(positions, dtype=float)                  # (m,)
    c = np.zeros((len(candidates), P.size), dtype=float)
    for s in range(0, R.shape[0], block):
        chunk = R[s:s + block]                              # (b, n)
        c += np.abs(chunk[:, :, None] - P[None, None, :]).sum(axis=0)
    return c


def prepare_instance(rankings, positions=None):
    """
    Convenience wrapper used by every ``*_rankings`` entry point.

    Returns ``(candidates, positions, c_matrix)`` where ``positions`` defaults
    to ``[1, 2, ..., n]`` and ``candidates`` preserves the key order of
    ``rankings[0]`` (exactly what the original code did).
    """
    candidates = list(rankings[0].keys())
    n = len(candidates)
    if positions is None:
        positions = [p + 1 for p in range(n)]
    c_matrix = build_cost_matrix(rankings, candidates, positions)
    return candidates, positions, c_matrix


def matchings_to_ranking(matching):
    """``[(candidate, position, weight?), ...] -> {candidate: position}``."""
    return {edge[0]: edge[1] for edge in matching}


def is_permutation(ranking, candidates, positions=None):
    """True iff ``ranking`` assigns every candidate a distinct position."""
    if set(ranking.keys()) != set(candidates):
        return False
    used = sorted(ranking.values())
    if positions is None:
        positions = list(range(1, len(candidates) + 1))
    return used == sorted(positions)


# ---------------------------------------------------------------------------
# Assignment problem: primal matching and optimal dual potentials
# ---------------------------------------------------------------------------

def hungarian_potentials(cost):
    """
    Jonker-Volgenant shortest augmenting path for the *minimisation* assignment
    problem.

    Returns ``(u, v, col_of_row)`` with

        u[i] + v[j] <= cost[i, j]          for every (i, j)
        u[i] + v[j] == cost[i, j]          for every matched (i, j)

    i.e. an optimal solution of the dual LP that ``find_tight_pairs`` used to
    solve with ``linprog``.  Runs in O(n^3) time and O(n^2) memory instead of
    building an LP with n^2 constraints and n^2 * 2n dense coefficients.
    """
    cost = np.asarray(cost, dtype=float)
    n, m = cost.shape
    if n > m:
        raise ValueError("hungarian_potentials expects n_rows <= n_cols")

    INF = np.inf
    u = np.zeros(n + 1, dtype=float)
    v = np.zeros(m + 1, dtype=float)
    p = np.zeros(m + 1, dtype=np.int64)      # p[j] = 1-based row matched to col j
    way = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, INF, dtype=float)
        used = np.zeros(m + 1, dtype=bool)

        while True:
            used[j0] = True
            i0 = int(p[j0])
            free = ~used[1:]

            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0

            masked = np.where(free, minv[1:], INF)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]

            # Reweight: used columns keep tightness, free columns shrink.
            u[p[used]] += delta
            v[used] -= delta
            minv[~used] -= delta

            j0 = j1
            if p[j0] == 0:
                break

        while j0:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1

    col_of_row = np.full(n, -1, dtype=np.int64)
    for j in range(1, m + 1):
        if p[j]:
            col_of_row[p[j] - 1] = j - 1
    return u[1:].copy(), v[1:].copy(), col_of_row


def prune_to_matching_covered(tight_mask, col_of_row):
    """
    Drop tight edges that lie in *no* perfect matching of the tight subgraph.

    An edge (i, j) survives iff it is in the reference matching M, or i and
    M^-1(j) share a strongly connected component of the digraph with an arc
    i -> M^-1(j) for every non-matching edge (i, j).  This is the standard
    Dulmage-Mendelsohn characterisation, so the set of perfect matchings - and
    therefore the set of optimal consensus rankings - is exactly preserved.

    Worth doing: optimal duals are highly degenerate on structured instances,
    and the raw tight set can be an order of magnitude larger than the union of
    the optimal matchings (5100 vs 200 edges on a 100-candidate polarised
    instance).  Every downstream solver is sized by |E_0|.
    """
    mask = np.asarray(tight_mask, dtype=bool)
    n = mask.shape[0]
    if mask.shape[0] != mask.shape[1]:
        return mask

    col_of_row = np.asarray(col_of_row, dtype=np.int64)
    row_of_col = np.empty(n, dtype=np.int64)
    row_of_col[col_of_row] = np.arange(n)

    rows, cols = np.nonzero(mask)
    is_matched = cols == col_of_row[rows]

    src = rows[~is_matched]
    dst = row_of_col[cols[~is_matched]]
    if src.size:
        graph = csr_matrix(
            (np.ones(src.size, dtype=np.int8), (src, dst)), shape=(n, n)
        )
        _ncomp, label = connected_components(graph, directed=True, connection="strong")
        keep = is_matched | (label[rows] == label[row_of_col[cols]])
    else:
        keep = is_matched

    pruned = np.zeros_like(mask)
    pruned[rows[keep], cols[keep]] = True
    return pruned


def find_tight_pairs(C_labels, P_labels, c_matrix, tol=1e-5, method="lp", prune=False):
    """
    Solves the dual LP

        max sum(u_i) + sum(v_p)  s.t.  u_i + v_p <= c_{i,p}

    and returns every pair (i, p) that is tight, together with the optimal
    duals.  Signature and return value match the original implementation.

    THE DEFAULTS REPRODUCE THE ORIGINAL IMPLEMENTATION EXACTLY.  Read the
    warning below before changing them.

    ``method="lp"`` (default) is the original ``scipy.linprog`` dual solve.
    ``method="dual"`` uses Hungarian potentials instead: O(n^3) time and O(n^2)
    memory rather than an LP with n^2 constraints and an n^2 x 2n dense matrix
    (~430 MB at n=300).  Both return *an* optimal dual, but not the same one.

    ``prune=True`` additionally removes tight edges that belong to no perfect
    matching - see ``prune_to_matching_covered`` - which can shrink E_0 by an
    order of magnitude on degenerate instances.

    WARNING.  Neither option changes the *set of optimal consensus rankings*
    inside E_0: that set is invariant, and there is a test asserting it.  What
    they do change is the dead edges E_0 carries around, and downstream solvers
    break ties between equal-weight matchings by looking at the whole cost
    matrix, dead entries included.  On instances with many exact ties this
    silently selects a different (equally optimal for the round's linear
    objective, but not equally useful) matching.  Measured effect: on
    two_blocks(8), MWU returns min-dispersion 2.0 with the defaults and 0.0
    with either option enabled, because the tie-break flips it onto a ranking
    it had already produced.  Enable them for scale, and re-check any
    tie-sensitive method when you do.

    Complementary slackness guarantees that *every* optimal assignment uses
    only tight edges, and conversely every perfect matching inside the tight
    subgraph has exactly the optimal cost - independently of which optimal dual
    solution is used.  So the tight subgraph E_0 is the arena of all optimal
    consensus rankings.
    """
    c = np.asarray(c_matrix, dtype=float)
    n_C, n_P = len(C_labels), len(P_labels)
    if c.shape != (n_C, n_P):
        raise ValueError(f"c_matrix has shape {c.shape}, expected {(n_C, n_P)}")

    if method == "lp":
        u_opt, v_opt = _tight_pairs_lp_duals(c)
        col_of_row = None
    elif method == "dual":
        u_opt, v_opt, col_of_row = hungarian_potentials(c)
    else:
        raise ValueError("method must be 'dual' or 'lp'")

    tight = np.isclose(u_opt[:, None] + v_opt[None, :], c, atol=tol)

    if prune and n_C == n_P:
        if col_of_row is None:
            col_of_row = linear_sum_assignment(c)[1]
        tight = prune_to_matching_covered(tight, col_of_row)

    rows, cols = np.nonzero(tight)
    tight_pairs = [(C_labels[i], P_labels[j]) for i, j in zip(rows, cols)]
    return tight_pairs, u_opt, v_opt


def _tight_pairs_lp_duals(c):
    """Original dense-LP dual solve, kept for validation purposes."""
    n_C, n_P = c.shape
    n_vars = n_C + n_P
    c_obj = -np.ones(n_vars)

    A_ub = np.zeros((n_C * n_P, n_vars))
    idx = np.arange(n_C * n_P)
    A_ub[idx, np.repeat(np.arange(n_C), n_P)] = 1.0
    A_ub[idx, n_C + np.tile(np.arange(n_P), n_C)] = 1.0
    b_ub = c.reshape(-1)

    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=(None, None), method="highs")
    if not res.success:
        raise RuntimeError(f"LP optimization failed: {res.message}")
    return res.x[:n_C], res.x[n_C:]


def max_weight_perfect_matching(adj_matrix, row_labels, col_labels):
    """
    Maximum weight perfect matching via the Hungarian algorithm.

    Returns ``(total_weight, [(row_label, col_label, weight), ...])``.

    Unlike the original, ``-inf`` entries (used to mark forbidden edges) are
    handled: SciPy rejects non-finite cost matrices outright, so they are
    replaced by a sentinel that can never appear in an optimal solution unless
    no perfect matching exists - in which case ValueError is raised, as before.
    """
    A = np.asarray(adj_matrix, dtype=float)
    forbidden = ~np.isfinite(A)

    if forbidden.any():
        finite = A[~forbidden]
        if finite.size == 0:
            raise ValueError("No perfect matching exists for this graph structure.")
        span = float(np.abs(finite).max())
        sentinel = -(span * (A.shape[0] + 1.0) + 1.0)
        A = np.where(forbidden, sentinel, A)

    row_ind, col_ind = linear_sum_assignment(A, maximize=True)

    if forbidden.any() and forbidden[row_ind, col_ind].any():
        raise ValueError("No perfect matching exists for this graph structure.")

    weights = np.asarray(adj_matrix, dtype=float)[row_ind, col_ind]
    matching = [
        (row_labels[r], col_labels[c], w)
        for r, c, w in zip(row_ind, col_ind, weights)
    ]
    return float(weights.sum()), matching


# ---------------------------------------------------------------------------
# Distances and dispersion
# ---------------------------------------------------------------------------

def footrule_distance(tau, sigma, candidates):
    """Spearman's footrule distance between two rankings."""
    return sum(abs(tau[i] - sigma[i]) for i in candidates)


def _pairwise_footrule(sigmas, candidates):
    """All pairwise footrule distances as a dense (k, k) array."""
    R = rankings_to_matrix(sigmas, candidates)
    return np.abs(R[:, None, :] - R[None, :, :]).sum(axis=-1)


def sum_dispersion(sigmas, candidates, distance=footrule_distance):
    """Sum of pairwise distances over a set of rankings."""
    if len(sigmas) < 2:
        return 0.0
    if distance is footrule_distance:
        D = _pairwise_footrule(sigmas, candidates)
        return float(D[np.triu_indices(len(sigmas), k=1)].sum())
    return float(
        sum(distance(a, b, candidates) for a, b in combinations(sigmas, 2))
    )


def min_dispersion(sigmas, candidates, distance=footrule_distance):
    """Minimum pairwise distance over a set of rankings."""
    if len(sigmas) < 2:
        return np.inf
    if distance is footrule_distance:
        D = _pairwise_footrule(sigmas, candidates)
        iu = np.triu_indices(len(sigmas), k=1)
        return float(D[iu].min())
    return float(
        min(distance(a, b, candidates) for a, b in combinations(sigmas, 2))
    )

def max_dispersion(sigmas, candidates, distance=footrule_distance):
    """Maximum pairwise distance over a set of rankings."""
    if len(sigmas) < 2:
        return np.inf
    if distance is footrule_distance:
        D = _pairwise_footrule(sigmas, candidates)
        iu = np.triu_indices(len(sigmas), k=1)
        return float(D[iu].max())
    return float(
        max(distance(a, b, candidates) for a, b in combinations(sigmas, 2))
    )

def dispersion_kt(sigmas, candidates):
    if len(sigmas) < 2:
        return 0.0, 0.0, 0.0
    sum_disp = 0
    min_disp = 1e8
    for a, b in combinations(sigmas, 2):
        dist = kendall_tau_distance(a, b, candidates)
        sum_disp += dist
        min_disp = min(min_disp, dist)
    return sum_disp, min_disp

def kendall_tau_distance(tau, sigma, candidates):
    """Computes Kendall's tau distance between two permutations."""
    distance = 0
    for i, j in combinations(candidates, 2):
        if (tau[i] - tau[j]) * (sigma[i] - sigma[j]) < 0:
            distance += 1
    return distance

# ---------------------------------------------------------------------------
# Modular arithmetic (used by the algebraic min-dispersion algorithm)
# ---------------------------------------------------------------------------

# Products of two residues must fit in int64; keep a little headroom for the
# accumulation performed inside mod_matmul.
INT64_SAFE_PRIME_LIMIT = 1 << 31


def is_prime(num):
    """Deterministic Miller-Rabin for 64-bit integers."""
    num = int(num)
    if num < 2:
        return False
    for small in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if num % small == 0:
            return num == small
    d, r = num - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        x = pow(a, d, num)
        if x in (1, num - 1):
            continue
        for _ in range(r - 1):
            x = x * x % num
            if x == num - 1:
                break
        else:
            return False
    return True


def next_prime(n):
    """Smallest prime strictly greater than ``n``."""
    p = int(n) + 1
    if p <= 2:
        return 2
    if p % 2 == 0:
        p += 1 if p > 2 else 0
    while not is_prime(p):
        p += 1 if p == 2 else 2
    return p


def invert_matrix_mod_p(M, P):
    """Inverse of a square matrix modulo prime P (Gauss-Jordan)."""
    n = len(M)
    A = [[int(M[i][j]) % P for j in range(n)] for i in range(n)]
    I = [[1 if i == j else 0 for j in range(n)] for i in range(n)]

    for i in range(n):
        pivot = i
        while pivot < n and A[pivot][i] == 0:
            pivot += 1
        if pivot == n:
            raise ValueError("Matrix is singular")
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            I[i], I[pivot] = I[pivot], I[i]

        inv_p = pow(A[i][i], P - 2, P)
        Ai, Ii = A[i], I[i]
        A[i] = [x * inv_p % P for x in Ai]
        I[i] = [x * inv_p % P for x in Ii]
        Ai, Ii = A[i], I[i]

        for k in range(n):
            if k == i:
                continue
            factor = A[k][i]
            if factor:
                Ak, Ik = A[k], I[k]
                for j in range(n):
                    Ak[j] = (Ak[j] - factor * Ai[j]) % P
                    Ik[j] = (Ik[j] - factor * Ii[j]) % P
    return I


def determinant_mod_p(M, P):
    """Determinant of a single matrix modulo prime P."""
    n = len(M)
    A = [[int(M[i][j]) % P for j in range(n)] for i in range(n)]
    det = 1
    for i in range(n):
        pivot = i
        while pivot < n and A[pivot][i] == 0:
            pivot += 1
        if pivot == n:
            return 0
        if pivot != i:
            A[i], A[pivot] = A[pivot], A[i]
            det = (-det) % P

        Ai = A[i]
        det = det * Ai[i] % P
        inv_p = pow(Ai[i], P - 2, P)
        for k in range(i + 1, n):
            factor = A[k][i]
            if factor:
                f = factor * inv_p % P
                Ak = A[k]
                for j in range(i, n):
                    Ak[j] = (Ak[j] - f * Ai[j]) % P
    return det % P


def _pow_mod_array(base, exp, P):
    """Element-wise modular exponentiation on an int64 array."""
    result = np.ones_like(base)
    b = base % P
    e = int(exp)
    while e:
        if e & 1:
            result = result * b % P
        b = b * b % P
        e >>= 1
    return result


def determinants_mod_p_batch(mats, P):
    """
    Determinants of a stack of matrices modulo prime P.

    ``mats`` has shape ``(B, n, n)``; the return value has shape ``(B,)``.
    Gaussian elimination is executed once, vectorised across the batch, which
    replaces B separate pure-Python eliminations.  Requires
    ``P < INT64_SAFE_PRIME_LIMIT``.
    """
    A = np.asarray(mats, dtype=np.int64) % P
    if A.ndim != 3 or A.shape[1] != A.shape[2]:
        raise ValueError("mats must have shape (B, n, n)")
    B, n, _ = A.shape
    if n == 0:
        return np.ones(B, dtype=np.int64)

    A = A.copy()
    idx = np.arange(B)
    det = np.ones(B, dtype=np.int64)
    sign = np.ones(B, dtype=np.int64)
    alive = np.ones(B, dtype=bool)

    for i in range(n):
        nz = A[:, i:, i] != 0
        has_pivot = nz.any(axis=1)
        det = np.where(alive & ~has_pivot, 0, det)
        alive &= has_pivot
        if not alive.any():
            break

        piv = np.argmax(nz, axis=1) + i
        swap = piv != i
        row_i = A[idx, i, :].copy()
        A[idx, i, :] = np.where(swap[:, None], A[idx, piv, :], row_i)
        A[idx, piv, :] = np.where(swap[:, None], row_i, A[idx, piv, :])
        sign = np.where(swap & alive, -sign % P, sign)

        pivots = np.where(alive, A[:, i, i], 1)
        det = np.where(alive, det * pivots % P, det)

        if i + 1 < n:
            inv = _pow_mod_array(pivots, P - 2, P)
            factors = A[:, i + 1:, i] * inv[:, None] % P
            A[:, i + 1:, i:] = (
                A[:, i + 1:, i:] - factors[:, :, None] * A[:, i, None, i:]
            ) % P

    return det * sign % P


def mod_matmul(A, B, P):
    """
    ``A @ B (mod P)`` for int64 arrays, reducing often enough that the
    accumulator never overflows.  ``A`` may have any number of leading batch
    dimensions; contraction is over the last axis of ``A``.
    """
    A = np.asarray(A, dtype=np.int64) % P
    B = np.asarray(B, dtype=np.int64) % P
    k = A.shape[-1]
    if B.shape[0] != k:
        raise ValueError("shape mismatch in mod_matmul")

    max_terms = max(1, int((np.iinfo(np.int64).max) // (int(P - 1) ** 2 or 1)))
    if max_terms >= k:
        return A @ B % P

    out = np.zeros(A.shape[:-1] + B.shape[1:], dtype=np.int64)
    for s in range(0, k, max_terms):
        out = (out + A[..., s:s + max_terms] @ B[s:s + max_terms]) % P
    return out