from ortools.sat.python import cp_model
import numpy as np

from utils import find_tight_pairs
from utils import max_weight_perfect_matching
from utils import footrule_distance


DEFAULT_TIME_LIMIT = 30
DEFAULT_WORKERS = 16


def _valid_columns_by_candidate(candidates, positions, edge_set):
    """Map each candidate row to its sorted valid column indices.

    Building this map directly from ``edge_set`` is O(|E|), whereas scanning
    every candidate-position pair can be O(n^2 |E|) when ``edge_set`` is a
    list. Edges containing labels outside the supplied partitions are ignored,
    matching the behavior of the previous membership-test implementation.
    """
    candidate_to_index = {
        candidate: i for i, candidate in enumerate(candidates)
    }
    position_to_index = {
        position: j for j, position in enumerate(positions)
    }
    valid_columns = [set() for _ in candidates]

    for candidate, position in edge_set:
        i = candidate_to_index.get(candidate)
        j = position_to_index.get(position)
        if i is not None and j is not None:
            valid_columns[i].add(j)

    result = []
    for i, columns in enumerate(valid_columns):
        if not columns:
            raise ValueError(
                f"Row label '{candidates[i]}' has no valid edges."
            )
        result.append(sorted(columns))

    return result


def _create_matching_variables(model, candidates, positions, edge_set, k):
    """Create the k domain-restricted permutation vectors."""
    valid_columns = _valid_columns_by_candidate(
        candidates,
        positions,
        edge_set,
    )

    # A domain is identical across matchings for a fixed candidate, so build it
    # once and reuse it for all k copies.
    domains = [
        cp_model.Domain.FromValues(columns) for columns in valid_columns
    ]
    row_spans = [columns[-1] - columns[0] for columns in valid_columns]

    X = [
        [
            model.NewIntVarFromDomain(
                domains[i],
                f"X_m{m}_i{i}",
            )
            for i in range(len(candidates))
        ]
        for m in range(k)
    ]

    for matching_vars in X:
        model.AddAllDifferent(matching_vars)

    return X, row_spans


def _pairwise_distance_expressions(model, X, row_spans):
    """Create compact expressions for every pairwise footrule distance.

    CP-SAT accepts a linear expression directly in AddAbsEquality, so no
    signed ``diff`` variable or defining equality is needed. Likewise, the
    sum of the absolute differences can be used directly without introducing
    a separate total-distance variable for every pair of matchings.
    """
    pairwise_distances = []
    k = len(X)

    for m1 in range(k):
        for m2 in range(m1 + 1, k):
            abs_differences = []

            for i, max_difference in enumerate(row_spans):
                # If this candidate has only one allowed position, its
                # contribution is identically zero and needs no model object.
                if max_difference == 0:
                    continue

                abs_difference = model.NewIntVar(
                    0,
                    max_difference,
                    f"abs_diff_{m1}_{m2}_{i}",
                )
                model.AddAbsEquality(
                    abs_difference,
                    X[m1][i] - X[m2][i],
                )
                abs_differences.append(abs_difference)

            if abs_differences:
                pairwise_distances.append(
                    cp_model.LinearExpr.Sum(abs_differences)
                )
            else:
                pairwise_distances.append(0)

    return pairwise_distances


def _add_matching_symmetry_breaking(model, X):
    """Preserve the original ordering symmetry break on the first row."""
    for m in range(len(X) - 1):
        model.Add(X[m][0] <= X[m + 1][0])


def _new_solver():
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = int(DEFAULT_WORKERS)
    if DEFAULT_TIME_LIMIT is not None:
        solver.parameters.max_time_in_seconds = float(DEFAULT_TIME_LIMIT)
    return solver


def _extract_matchings(
    solver,
    X,
    candidates,
    positions,
    weight_matrix,
):
    """Convert the solver's column indices back to the original labels."""
    all_matchings = []

    for matching_vars in X:
        current_matching = []
        for i, variable in enumerate(matching_vars):
            assigned_col_index = solver.Value(variable)
            current_matching.append(
                (
                    candidates[i],
                    positions[assigned_col_index],
                    weight_matrix[i][assigned_col_index],
                )
            )
        all_matchings.append(current_matching)

    return all_matchings


def _build_dispersion_model(candidates, positions, edge_set, k):
    """Build the shared portion of the sum and max-min CP-SAT models."""
    model = cp_model.CpModel()
    X, row_spans = _create_matching_variables(
        model,
        candidates,
        positions,
        edge_set,
        k,
    )
    pairwise_distances = _pairwise_distance_expressions(
        model,
        X,
        row_spans,
    )
    _add_matching_symmetry_breaking(model, X)

    # This is a safe pair-distance bound for arbitrary restricted domains. For
    # square permutation instances, the Spearman footrule bound floor(n^2 / 2)
    # is tighter and can improve propagation of the max-min objective.
    pair_distance_upper_bound = sum(row_spans)
    if len(candidates) == len(positions):
        pair_distance_upper_bound = min(
            pair_distance_upper_bound,
            (len(positions) ** 2) // 2,
        )
    if len(X) < 2:
        # Preserve the previous (otherwise vacuous) k < 2 objective behavior.
        pair_distance_upper_bound = len(candidates) * max(
            0,
            len(positions) - 1,
        )

    return model, X, pairwise_distances, pair_distance_upper_bound


def CP_SAT_sum_dispersion(candidates, positions, edge_set, weight_matrix, k):
    """
    Finds K perfect matchings maximizing the SUM of index distances across all pairs.
    """
    model, X, pairwise_distances, _ = _build_dispersion_model(
        candidates,
        positions,
        edge_set,
        k,
    )

    if pairwise_distances:
        model.Maximize(cp_model.LinearExpr.Sum(pairwise_distances))
    else:
        model.Maximize(0)

    solver = _new_solver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return (
            solver.ObjectiveValue(),
            _extract_matchings(
                solver,
                X,
                candidates,
                positions,
                weight_matrix,
            ),
        )

    return None, []


def CP_SAT(candidates, positions, edge_set, weight_matrix, k):
    """
    Finds K perfect matchings maximizing the minimum index distance between any pair.

    :param candidates: List of labels for the left partition (e.g., ['u1', 'u2', ...])
    :param positions: List of labels for the right partition (e.g., ['v1', 'v2', ...])
    :param weight_matrix: 2D array where weight_matrix[i][j] is the weight of edge (i, j).
    :param k: Number of perfect matchings to find.
    """
    model, X, pairwise_distances, distance_upper_bound = (
        _build_dispersion_model(
            candidates,
            positions,
            edge_set,
            k,
        )
    )

    min_dist = model.NewIntVar(0, distance_upper_bound, "min_dist")
    for pairwise_distance in pairwise_distances:
        model.Add(min_dist <= pairwise_distance)

    model.Maximize(min_dist)

    solver = _new_solver()
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return (
            solver.ObjectiveValue(),
            _extract_matchings(
                solver,
                X,
                candidates,
                positions,
                weight_matrix,
            ),
        )

    return None, []


def _aggregate_footrule_cost_matrix(rankings, candidates):
    """Compute all aggregate footrule edge costs using rank histograms.

    For every candidate, let h[q] count input rankings placing it at position
    q. Prefix counts and prefix weighted sums then evaluate

        sum_q h[q] * abs(q - p)

    for every target position p. This takes O(r*n + n^2) time and O(n^2)
    memory, rather than three nested Python loops taking O(r*n^2) time.
    """
    number_of_rankings = len(rankings)
    n = len(candidates)

    raw_rank_values = np.asarray(
        [
            [ranking[candidate] for candidate in candidates]
            for ranking in rankings
        ]
    )

    try:
        rank_values = raw_rank_values.astype(np.int64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Every ranking position must be an integer.") from error

    if (
        np.any(raw_rank_values != rank_values)
        or np.any(rank_values < 1)
        or np.any(rank_values > n)
    ):
        raise ValueError(
            "Every ranking position must be an integer between 1 and n."
        )

    # Flatten (candidate, observed-position) coordinates into histogram bins.
    bin_indices = (
        (np.arange(n, dtype=np.int64) * n)[None, :]
        + rank_values
        - 1
    )
    counts = np.bincount(
        bin_indices.ravel(),
        minlength=n * n,
    ).reshape(n, n)

    target_positions = np.arange(1, n + 1, dtype=np.int64)[None, :]
    prefix_counts = np.cumsum(counts, axis=1)
    prefix_weighted_positions = np.cumsum(
        counts * target_positions,
        axis=1,
    )
    total_weighted_positions = prefix_weighted_positions[:, -1, None]

    cost_matrix = (
        target_positions * prefix_counts
        - prefix_weighted_positions
        + total_weighted_positions
        - prefix_weighted_positions
        - target_positions * (number_of_rankings - prefix_counts)
    )

    # Preserve the previous floating-point matrix type expected by downstream
    # code, while doing the computation exactly with integers.
    return cost_matrix.astype(np.float64)


def _tight_ranking_instance(rankings):
    candidates = list(rankings[0].keys())
    positions = list(range(1, len(candidates) + 1))
    c_matrix = _aggregate_footrule_cost_matrix(rankings, candidates)
    tight_pairs, _, _ = find_tight_pairs(
        candidates,
        positions,
        c_matrix,
    )
    return candidates, positions, c_matrix, tight_pairs


def CP_SAT_tight_min_ranking(rankings, k):
    candidates, positions, c_matrix, tight_pairs = _tight_ranking_instance(
        rankings
    )
    _, perfect_matchings = CP_SAT(
        candidates,
        positions,
        tight_pairs,
        c_matrix,
        k,
    )
    return [
        {candidate: position for candidate, position, _ in matching}
        for matching in perfect_matchings
    ]


def CP_SAT_tight_sum_ranking(rankings, k):
    candidates, positions, c_matrix, tight_pairs = _tight_ranking_instance(
        rankings
    )
    _, perfect_matchings = CP_SAT_sum_dispersion(
        candidates,
        positions,
        tight_pairs,
        c_matrix,
        k,
    )
    return [
        {candidate: position for candidate, position, _ in matching}
        for matching in perfect_matchings
    ]