import math
import random

import numpy as np
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

import networkx as nx
import itertools
from itertools import combinations

from utils import find_tight_pairs
from utils import max_weight_perfect_matching
from utils import footrule_distance

def find_tau_mwu(sigmas, E0, eta, candidates, positions):
    """
    Executes the MWU algorithm to find tau.

    :param n: Integer, size of permutations (0 to n-1).
    :param sigmas: List of dictionaries or arrays representing center permutations.
                   sigmas[j][i] gives the mapped value p.
    :param E0: List of tuples (i, p) representing valid edges in G_0.
    :param eta: Float, additive accuracy (0 < eta <= B_n).
    :return: A randomly chosen permutation tau from the MWU rounds.
    """
    n = len(candidates)
    t = len(sigmas)
    candidate_to_idx = {candidate: i for i, candidate in enumerate(candidates)}
    position_to_idx = {position: p for p, position in enumerate(positions)}
    Bn = (n ** 2) // 2  # Max possible footrule distance

    #precompute the distances from each tight edge rank, to the sigmas
    # Pre-allocate array of shape (len(E0), t)
    edge_distances = np.empty((len(E0), t))

    for i, (c, pos) in enumerate(E0):
        # Assign the list comprehension directly to the row index
        edge_distances[i, :] = [abs(pos - sigmas[j][c]) for j in range(t)]

    #to avoid looping when assigning weights to the cost matrix
    tight_candidates = np.fromiter(
        (candidate_to_idx[c] for c, _ in E0),
        dtype=np.intp,
        count=len(E0),
    )
    tight_positions = np.fromiter(
        (position_to_idx[p] for _, p in E0),
        dtype=np.intp,
        count=len(E0),
    )

    sigma_positions = np.asarray(
        [[sigma[c] for c in candidates] for sigma in sigmas],
        dtype=np.int64,
    )

    position_values = np.asarray(positions, dtype=np.int64)

    # Handle the trivial case t = 1
    if t == 1:
        cost_matrix = np.full((n, n), -n**2)
        for i, (c, pos) in enumerate(E0):
            # Weight sum over all t centers
            weight = edge_distances[i][0]
            cost_matrix[candidate_to_idx[c], position_to_idx[pos]] = weight
        total_weight, matching = max_weight_perfect_matching(cost_matrix, candidates, positions)
        return {i: p for i, p, w in matching}
    
    # Initialization for MWU (t >= 2)
    T = math.ceil((4 * (Bn ** 2) * math.log(t)) / (eta ** 2))
    alpha = eta / (2.0 * Bn)
    beta = np.full(t, 1.0 / t)
    tau_s = []

    #sample ahead of time which round will actually be returned and stop there
    selected_round = random.randrange(T)

    #preallocate the cost matrix for the matching problem, since non-tight edges weight will never change
    cost_matrix = np.full((n, n), -np.inf)
    for s in range(selected_round + 1):
        # Maximum weight perfect matching on G_0
        #compute the weights for each tight edge using matrix-vector multiplication
        weights = edge_distances @ beta
        cost_matrix[tight_candidates, tight_positions] = weights
        # Solve matching
        total_weight, matching = max_weight_perfect_matching(cost_matrix, candidates, positions)
        # Reconstruct tau^s
        tau_s = {i: p for i, p, w in matching}

        # Update beta
        for j in range(t):
            dist = footrule_distance(tau_s, sigmas[j], candidates)
            beta[j] = beta[j] * (1.0 - alpha * (dist / Bn))

        #always keep it normalized to sum to 1
        beta /= beta.sum()

    return tau_s

def min_dispersion_approx_rankings_MWU(rankings, k, epsilon=0.05):

    nCandidates = len(rankings[0])
    nPositions = len(rankings[0])
    
    candidates = list(rankings[0].keys())
    positions = [p+1 for p in range(nPositions)]
    c_matrix = np.zeros((nCandidates, nPositions))

    # Populating the c_matrix map.
    for i in range(nCandidates):
      for p in range(nPositions):
        for rank in rankings:
          c_matrix[i, p] += abs(rank[candidates[i]] - positions[p])

    # Finding the tight pairs of edges
    tight_pairs, u_opt, v_opt = find_tight_pairs(candidates, positions, c_matrix)

    # Finding the min cost perfect matching to find the initial ranking by using -c_matrix.
    total_weight, matching = max_weight_perfect_matching(-c_matrix, candidates, positions)
    eta = epsilon* (-total_weight)/len(rankings)
    # Using the multiplicaitve weight update rule to generate k diverse solutions.
    diverse_rankings = [{i: p for i, p, w in matching}]
    for i in range(1,k):
      matching = find_tau_mwu(diverse_rankings, tight_pairs, eta, candidates, positions)
      diverse_rankings.append({i: int(matching[i]) for i in matching})
    return diverse_rankings


def find_tau_mwu_sparse(sigmas, E0, eta, candidates, positions):
    """Run MWU using SciPy's sparse full bipartite matching oracle.

    This is the sparse counterpart of :func:`find_tau_mwu`.  The rows of the
    sparse biadjacency matrix correspond to ``candidates`` and its columns to
    ``positions``.  Only edges in ``E0`` are stored, so an absent sparse entry
    represents a forbidden candidate-position assignment.

    The CSR sparsity pattern is constructed once.  During an MWU round, only
    the stored edge weights in ``sparse_graph.data`` are updated.

    :param sigmas: Previously selected rankings. ``sigmas[j][c]`` is the
                   position of candidate ``c`` in ranking ``j``.
    :param E0: Tight edges, represented as ``(candidate, position)`` pairs.
    :param eta: Additive accuracy parameter, with ``0 < eta <= B_n``.
    :param candidates: Candidate labels, in matrix-row order.
    :param positions: Position labels, in matrix-column order.
    :return: A randomly selected MWU ranking as a candidate-position dict.
    """
    n = len(candidates)
    t = len(sigmas)

    if t == 0:
        raise ValueError("sigmas must contain at least one ranking")
    if len(positions) != n:
        raise ValueError("candidates and positions must have the same length")

    candidate_to_idx = {
        candidate: i for i, candidate in enumerate(candidates)
    }
    position_to_idx = {
        position: p for p, position in enumerate(positions)
    }

    # Dense representations are used only for the small MWU weight vector and
    # for computing all t footrule distances at once.
    sigma_positions = np.asarray(
        [[sigma[c] for c in candidates] for sigma in sigmas],
        dtype=np.int64,
    )
    position_values = np.asarray(positions, dtype=np.int64)

    tight_candidates = np.fromiter(
        (candidate_to_idx[c] for c, _ in E0),
        dtype=np.intp,
        count=len(E0),
    )
    tight_positions = np.fromiter(
        (position_to_idx[p] for _, p in E0),
        dtype=np.intp,
        count=len(E0),
    )

    # Duplicate sparse coordinates would be interpreted as parallel entries
    # and may be combined by sparse-matrix operations.  E0 should be a set.
    edge_keys = tight_candidates * n + tight_positions
    if np.unique(edge_keys).size != len(E0):
        raise ValueError("E0 contains duplicate candidate-position edges")

    tight_position_values = position_values[tight_positions]
    edge_distances = np.abs(
        tight_position_values[:, None]
        - sigma_positions[:, tight_candidates].T
    ).astype(np.float64)

    # Construct canonical CSR storage explicitly.  This lets us update
    # sparse_graph.data in place without rebuilding or sorting a sparse matrix
    # in every MWU round.
    edge_order = np.lexsort((tight_positions, tight_candidates))
    sorted_rows = tight_candidates[edge_order]
    sorted_cols = tight_positions[edge_order]
    row_counts = np.bincount(sorted_rows, minlength=n)
    indptr = np.empty(n + 1, dtype=np.intp)
    indptr[0] = 0
    np.cumsum(row_counts, out=indptr[1:])

    sparse_graph = csr_matrix(
        (
            np.ones(len(E0), dtype=np.float64),
            sorted_cols,
            indptr,
        ),
        shape=(n, n),
    )

    edge_weights = np.empty(len(E0), dtype=np.float64)
    ordered_weights = np.empty(len(E0), dtype=np.float64)
    tau_positions = np.empty(n, dtype=np.int64)

    # SciPy's sparse matching routine requires stored edges to have nonzero
    # weights.  All MWU edge weights are nonnegative, so adding one makes every
    # stored edge nonzero.  Every full matching has exactly n edges, hence this
    # common shift does not change which matching is optimal.
    weight_shift = 1.0

    def solve_current_matching(beta):
        np.dot(edge_distances, beta, out=edge_weights)
        np.take(edge_weights, edge_order, out=ordered_weights)
        np.add(ordered_weights, weight_shift, out=ordered_weights)
        sparse_graph.data[:] = ordered_weights

        row_ind, col_ind = min_weight_full_bipartite_matching(
            sparse_graph,
            maximize=True,
        )
        tau_positions[row_ind] = position_values[col_ind]

    if t == 1:
        solve_current_matching(np.ones(1, dtype=np.float64))
        return {
            candidate: int(tau_positions[i])
            for i, candidate in enumerate(candidates)
        }

    Bn = (n ** 2) // 2
    T = math.ceil((4 * (Bn ** 2) * math.log(t)) / (eta ** 2))
    alpha = eta / (2.0 * Bn)
    update_scale = alpha / Bn
    beta = np.full(t, 1.0 / t)
    selected_round = random.randrange(T)

    # Reuse these arrays when computing all footrule distances and the MWU
    # update, avoiding t Python calls and per-round dictionary construction.
    distance_work = np.empty_like(sigma_positions)
    distances = np.empty(t, dtype=np.int64)
    decay = np.empty(t, dtype=np.float64)

    for s in range(selected_round + 1):
        solve_current_matching(beta)

        # The final update cannot affect the ranking being returned.
        if s == selected_round:
            break

        np.subtract(sigma_positions, tau_positions, out=distance_work)
        np.abs(distance_work, out=distance_work)
        np.sum(distance_work, axis=1, out=distances)

        np.multiply(distances, -update_scale, out=decay)
        decay += 1.0
        beta *= decay
        beta /= beta.sum()

    return {
        candidate: int(tau_positions[i])
        for i, candidate in enumerate(candidates)
    }


def min_dispersion_approx_rankings_MWU_sparse(rankings, k, epsilon=0.05):
    """Sparse-matching version of ``min_dispersion_approx_rankings_MWU``."""
    nCandidates = len(rankings[0])
    nPositions = len(rankings[0])

    candidates = list(rankings[0].keys())
    positions = [p + 1 for p in range(nPositions)]
    c_matrix = np.zeros((nCandidates, nPositions))

    for i in range(nCandidates):
        for p in range(nPositions):
            for rank in rankings:
                c_matrix[i, p] += abs(
                    rank[candidates[i]] - positions[p]
                )

    tight_pairs, u_opt, v_opt = find_tight_pairs(
        candidates,
        positions,
        c_matrix,
    )

    total_weight, matching = max_weight_perfect_matching(
        -c_matrix,
        candidates,
        positions,
    )
    eta = epsilon * (-total_weight) / len(rankings)

    diverse_rankings = [{i: p for i, p, w in matching}]
    for i in range(1, k):
        matching = find_tau_mwu_sparse(
            diverse_rankings,
            tight_pairs,
            eta,
            candidates,
            positions,
        )
        diverse_rankings.append(
            {candidate: int(matching[candidate]) for candidate in matching}
        )

    return diverse_rankings
