"""Max-sum diversity among (1 + alpha)-approximate footrule medians.

Implements Theorem 5.4.

* oracle='milp' (default): SciPy/HiGHS solver is used instead of the determinant oracle.
  It is an implementation of the same oracle, but MILP has no polynomial time worst-case guarantee.
* oracle='determinant': randomized edge coefficients and bivariate polynomial
  determinants/self-reduction, using python-flint (pip install python-flint).
  Native sparse polynomial arithmetic replaces object arrays, dense evaluation
  grids, and inverse Vandermonde matrices. It is intended for SMALL instances.
  Fraction-free Bareiss intermediates can grow much larger than the result.
  Size/term limits raise OracleLimitError instead of silently switching methods.

ITERATION/GUARANTEE DISTINCTION
The iteration count R=4*k**3*floor(d**2/2) required for the theorem is large.
To make the algorithm practical, the max_iterations=100 caps the value of R significantly. 
This cap means the (1-1/k) guarantee does not hold; but should give a good idea for the tradeoffs.

MILP 'certificates' rely on floating-point HiGHS bounds/optimality, with a
conservative rounding margin; and small rounding issues may occur.

For determinant answers, certificates are conditional on successful polynomial
identity testing. With independent nonzero coefficients mod p=2**61-1, each
oracle call misses its true optimal objective with probability at most d/(p-1).

Optional diagnostics:
    result, info = sum_dispersion_approx_rankings(
        rankings, K, alpha=0.05, max_iterations=50,
        oracle_time_limit=30, return_info=True)
    print(info['target_certified'], info['ratio_lower_bound'])

The finite-run upper bound follows from negative type. For the current
distribution mu, f=E[F(X,Y)]/2, and any upper bound U on max_s E[F(s,X)],
f* <= U-f and OPT_k <= k**2*f*. A separate integer cut bound for all rankings
is also used, as is the triangle bound OPT_k<=k*(k-1)*U. This is a posteriori
certification, not an assertion that every
capped run attains the paper's theorem.
"""

from __future__ import annotations

import math
import random
import time
import warnings
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix


class OracleLimitError(RuntimeError):
    """An oracle exceeded an explicit resource limit or failed to solve."""


class UncertifiedResultError(RuntimeError):
    """The outer iteration cap was reached without the requested certificate."""


def _integer_sum(values):
    return sum(map(int, values))


def _alpha_fraction(alpha):
    try:
        value = alpha if isinstance(alpha, Fraction) else Fraction(str(alpha))
    except (ValueError, TypeError, ZeroDivisionError) as exc:
        raise ValueError("alpha must be a finite nonnegative number") from exc
    if value < 0:
        raise ValueError("alpha must be nonnegative")
    return value


def _read_rankings(rankings):
    rankings = list(rankings)
    if not rankings or not isinstance(rankings[0], Mapping) or not rankings[0]:
        raise ValueError("rankings must contain nonempty candidate-to-position dictionaries")
    candidates = list(rankings[0])
    d = len(candidates)
    rows = []
    expected = set(candidates)
    offset = None
    for ranking in rankings:
        if not isinstance(ranking, Mapping) or set(ranking) != expected:
            raise ValueError("all rankings must contain the same candidates")
        values = []
        for candidate in candidates:
            value = ranking[candidate]
            try:
                integer = int(value)
            except (ValueError, TypeError, OverflowError) as exc:
                raise ValueError("positions must be integers") from exc
            if integer != value:
                raise ValueError("positions must be integers")
            values.append(integer)
        if offset is None:
            if sorted(values) == list(range(d)):
                offset = 0
            elif sorted(values) == list(range(1, d + 1)):
                offset = 1
            else:
                raise ValueError("positions must be a permutation of 0..d-1 or 1..d")
        if sorted(values) != list(range(offset, d + offset)):
            raise ValueError("rankings use inconsistent positions or are not permutations")
        rows.append([value - offset for value in values])
    # Keep all integer model data comfortably inside exact double-integer range.
    if len(rows) * d * d >= 2**50:
        raise ValueError("instance is too large for the supported exact-integer range")
    return candidates, offset, np.asarray(rows, dtype=np.int64)


def _aggregation_costs(rows):
    """O(nd+d^2), rather than Python loops over all n*d^2 triples."""
    n, d = rows.shape
    counts = np.zeros((d, d), dtype=np.int64)
    items = np.arange(d)
    for ranking in rows:
        counts[items, ranking] += 1
    costs = np.empty((d, d), dtype=np.int64)
    costs[:, 0] = counts @ np.arange(d, dtype=np.int64)
    if d > 1:
        differences = 2 * np.cumsum(counts, axis=1)[:, :-1] - n
        costs[:, 1:] = costs[:, [0]] + np.cumsum(differences, axis=1)
    return costs


def _reduced_costs(costs, median):
    """Integer optimal assignment duals from difference-constraint relaxation.

    v[p] <= v[median[i]] + c[i,p]-c[i,median[i]]. No negative cycles exist
    because median is optimal. Row duals then force matched edges to be tight.
    """
    d = len(median)
    matched = costs[np.arange(d), median]
    delta = costs - matched[:, None]
    v = np.zeros(d, dtype=np.int64)
    for _ in range(d):
        new_v = np.minimum(v, np.min(delta + v[median, None], axis=0))
        if np.array_equal(new_v, v):
            break
        v = new_v
    u = matched - v[median]
    reduced = costs - u[:, None] - v[None, :]
    if np.any(reduced < 0) or np.any(reduced[np.arange(d), median]):
        raise RuntimeError("failed to construct nonnegative integer reduced costs")
    return reduced


def _global_diversity_bound(d, k):
    """Integer upper bound via balanced occupancies at each rank threshold."""
    bound = 0
    for threshold in range(1, d):
        q, r = divmod(k * (d - threshold), d)
        bound += (d - r) * q * (k - q) + r * (q + 1) * (k - q - 1)
    return bound


@dataclass
class _OracleAnswer:
    ranking: np.ndarray
    value: int
    upper_bound: int
    optimal: bool
    method: str


class _BudgetedAssignment:
    def __init__(self, costs, alpha, backend, rng, time_limit,
                 allow_inexact, determinant_max_d, determinant_max_terms,
                 milp_backend='scipy'):
        self.costs = costs
        self.d = len(costs)
        self.items = np.arange(self.d)
        _, self.median = linear_sum_assignment(costs)
        self.optimal_cost = self.cost(self.median)
        self.budget = self.optimal_cost + (alpha.numerator * self.optimal_cost) // alpha.denominator
        reduced = _reduced_costs(costs, self.median)
        divisor = int(np.gcd.reduce(reduced.ravel())) or 1
        self.reduced = reduced // divisor
        self.slack = (self.budget - self.optimal_cost) // divisor
        self.allowed = self.reduced <= self.slack
        self.edge_i, self.edge_p = np.nonzero(self.allowed)
        self.backend = backend
        self.rng = rng
        self.time_limit = time_limit
        self.allow_inexact = allow_inexact
        self.determinant_max_d = determinant_max_d
        self.determinant_max_terms = determinant_max_terms
        self.calls = Counter()
        self._constraints = None
        self.milp_backend = milp_backend
        self._highs = None
        self._last_ranking = self.median.copy()

    def cost(self, ranking):
        return _integer_sum(self.costs[self.items, ranking])

    def _check(self, ranking):
        if sorted(map(int, ranking)) != list(range(self.d)) or self.cost(ranking) > self.budget:
            raise RuntimeError("oracle produced an invalid or over-budget ranking")

    def solve(self, labels):
        # Integer objectives remain exactly representable for ordinary assignment.
        if self.d * int(labels.max()) >= 2**50:
            raise OracleLimitError("label objective exceeded supported integer range; use fewer iterations")
        assignment_cost = np.where(self.allowed, -labels.astype(float), np.inf)
        _, unconstrained = linear_sum_assignment(assignment_cost)
        penalty = self.d * int(labels.max()) + 1
        integer_objective = np.where(self.allowed, -labels, penalty)
        # Verify the ordinary assignment's optimality with exact integer duals;
        # this also identifies its complete optimal face for tie-breaking.
        weight_reduced = _reduced_costs(integer_objective, unconstrained)
        upper = _integer_sum(labels[self.items, unconstrained])
        if self.cost(unconstrained) <= self.budget:
            self.calls['assignment'] += 1
            return _OracleAnswer(unconstrained, upper, upper, True, 'assignment')
        # An arbitrary max-weight assignment can fail the budget even though
        # another max-weight assignment fits. Minimize aggregation cost over
        # that entire optimal face using integer duals and a second assignment.
        tight = self.allowed & (weight_reduced == 0)
        _, cheapest = linear_sum_assignment(np.where(tight, self.costs.astype(float), np.inf))
        if self.cost(cheapest) <= self.budget:
            self.calls['assignment-tiebreak'] += 1
            return _OracleAnswer(cheapest, upper, upper, True, 'assignment-tiebreak')
        if self.backend == 'milp':
            result = self._milp(labels, upper)
        else:
            result = self._determinant(labels)
        self._check(result.ranking)
        self._last_ranking = result.ranking.copy()
        self.calls[result.method] += 1
        return result

    def _milp(self, labels, unconstrained_upper):
        i, p = self.edge_i, self.edge_p
        m = len(i)
        # Row shifts/gcd shrink objective coefficients without changing optima.
        row_min = np.min(np.where(self.allowed, labels, np.iinfo(np.int64).max), axis=1)
        shifted = labels[i, p] - row_min[i]
        divisor = int(np.gcd.reduce(shifted)) or 1
        objective = -(shifted // divisor).astype(float)
        offset = _integer_sum(row_min)
        if self._constraints is None:
            columns = np.tile(np.arange(m), 3)
            constraint_rows = np.concatenate([i, self.d + p, np.full(m, 2 * self.d)])
            data = np.concatenate([np.ones(2 * m), self.reduced[i, p].astype(float)])
            matrix = coo_matrix((data, (constraint_rows, columns)), shape=(2 * self.d + 1, m)).tocsc()
            lower = np.r_[np.ones(2 * self.d), -np.inf]
            upper = np.r_[np.ones(2 * self.d), float(self.slack)]
            self._constraints = LinearConstraint(matrix, lower, upper)
        options = {'mip_rel_gap': 0.0, 'presolve': True}
        if self.time_limit is not None:
            options['time_limit'] = self.time_limit
        if self.milp_backend == 'highspy':
            result = self._highspy(objective, labels)
        else:
            result = milp(objective, integrality=np.ones(m, dtype=np.uint8),
                          bounds=Bounds(0, 1), constraints=self._constraints, options=options)
        if result.status not in (0, 1):
            raise OracleLimitError(f"HiGHS oracle failed: {result.message}")
        if result.status != 0 and not self.allow_inexact:
            raise OracleLimitError(f"HiGHS oracle did not finish optimally: {result.message}. "
                                   "Increase oracle_time_limit or explicitly allow inexact oracles.")
        ranking = self.median.copy()
        feasible_incumbent = False
        if result.x is not None:
            chosen = np.flatnonzero(result.x > 0.5)
            if len(chosen) == self.d and len(set(map(int, i[chosen]))) == self.d and len(set(map(int, p[chosen]))) == self.d:
                candidate = np.empty(self.d, dtype=np.int64)
                candidate[i[chosen]] = p[chosen]
                if self.cost(candidate) <= self.budget:
                    ranking = candidate
                    feasible_incumbent = True
        if result.status == 0 and not feasible_incumbent:
            raise OracleLimitError('HiGHS reported optimality but no valid integer-budget witness was recovered')
        value = _integer_sum(labels[self.items, ranking])
        if result.status == 0:
            reported_value = offset - divisor * float(result.fun)
            tolerance = max(1e-6, abs(reported_value) * 1e-12)
            if abs(value - reported_value) > tolerance:
                raise OracleLimitError('HiGHS objective does not match the recovered integer witness')
        upper = unconstrained_upper
        dual = getattr(result, 'mip_dual_bound', None)
        if dual is not None and math.isfinite(dual):
            # Integer optimum <= floor(real upper bound). The added margin is
            # deliberately conservative; this remains a numerical certificate.
            raw_bound = offset - divisor * float(dual)
            margin = max(1e-5, abs(raw_bound) * 1e-8)
            if raw_bound + margin >= value:
                upper = min(upper, math.floor(raw_bound + margin))
        # Solver optimality and our deliberately inflated upper bound are
        # separate diagnostics: the latter need not round to the incumbent.
        optimal = result.status == 0
        method = 'highspy' if self.milp_backend == 'highspy' else 'milp'
        return _OracleAnswer(ranking, value, upper, optimal, method)

    def _highspy(self, objective, labels):
        try:
            import highspy
        except ImportError as exc:
            raise ImportError("milp_backend='highspy' requires: pip install highspy") from exc
        m = len(objective)
        indexes = np.arange(m, dtype=np.int32)
        if self._highs is None:
            h = highspy.Highs()
            h.setOptionValue('output_flag', False)
            h.setOptionValue('threads', 1)
            h.setOptionValue('mip_rel_gap', 0.0)
            h.setOptionValue('mip_abs_gap', 0.0)
            h.setOptionValue('random_seed', 0)
            lp = highspy.HighsLp()
            matrix = self._constraints.A
            lp.num_col_, lp.num_row_ = m, matrix.shape[0]
            lp.col_cost_ = objective
            lp.col_lower_, lp.col_upper_ = np.zeros(m), np.ones(m)
            lp.row_lower_, lp.row_upper_ = self._constraints.lb, self._constraints.ub
            lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
            lp.a_matrix_.start_ = matrix.indptr
            lp.a_matrix_.index_ = matrix.indices
            lp.a_matrix_.value_ = matrix.data
            if h.passModel(lp) != highspy.HighsStatus.kOk:
                raise OracleLimitError('HiGHS could not load the assignment model')
            h.changeColsIntegrality(m, indexes, np.ones(m, dtype=np.uint8))
            self._highs = h
        else:
            h = self._highs
            h.changeColsCost(m, indexes, objective)
        # Supply a fully feasible permutation, even for the first solve.
        start = max((self.median, self._last_ranking),
                    key=lambda r: _integer_sum(labels[self.items, r]))
        flat_edges = self.edge_i * self.d + self.edge_p
        selected = np.searchsorted(flat_edges, self.items * self.d + start)
        x = np.zeros(m)
        x[selected] = 1.0
        h.setSolution(m, indexes, x)
        if self.time_limit is not None:
            # HiGHS accumulates solve time when a model is reused.
            h.setOptionValue('time_limit', h.getRunTime() + self.time_limit)
        h.run()
        status = h.getModelStatus()
        if status == highspy.HighsModelStatus.kOptimal:
            code = 0
        elif status in (highspy.HighsModelStatus.kTimeLimit,
                        highspy.HighsModelStatus.kIterationLimit,
                        highspy.HighsModelStatus.kSolutionLimit):
            code = 1
        else:
            code = 4
        solution = h.getSolution()
        return SimpleNamespace(status=code, message=h.modelStatusToString(status),
                               x=np.asarray(solution.col_value) if solution.value_valid else None,
                               fun=h.getInfo().objective_function_value,
                               mip_dual_bound=h.getInfo().mip_dual_bound)

    def _determinant(self, labels):
        if self.d > self.determinant_max_d:
            raise OracleLimitError(f"determinant backend is limited to d<={self.determinant_max_d}; "
                                   "raise determinant_max_d explicitly or use oracle='milp'")
        try:
            from flint import nmod_mpoly_ctx
        except ImportError as exc:
            raise ImportError("oracle='determinant' requires python-flint: pip install python-flint") from exc
        prime = 2**61 - 1
        ctx = nmod_mpoly_ctx.get(('cost', 'label'), prime)
        zero, one = ctx.constant(0), ctx.constant(1)
        row_min = np.min(np.where(self.allowed, labels, np.iinfo(np.int64).max), axis=1)
        shifted = labels - row_min[:, None]
        divisor = int(np.gcd.reduce(shifted[self.allowed])) or 1
        shifted = shifted // divisor
        entries = [[zero for _ in range(self.d)] for _ in range(self.d)]
        for i, p in zip(self.edge_i, self.edge_p):
            entries[int(i)][int(p)] = ctx.from_dict({
                (int(self.reduced[i, p]), int(shifted[i, p])): self.rng.randrange(1, prime)})
        deadline = None if self.time_limit is None else time.perf_counter() + self.time_limit

        def determinant(rows, columns):
            matrix = [[entries[i][p] for p in columns] for i in rows]
            size = len(rows)
            if size == 0:
                return one
            previous, sign = one, 1
            for col in range(size - 1):
                if not matrix[col][col]:
                    pivot_row = next((i for i in range(col + 1, size) if matrix[i][col]), None)
                    if pivot_row is None:
                        return zero
                    matrix[col], matrix[pivot_row] = matrix[pivot_row], matrix[col]
                    sign = -sign
                pivot = matrix[col][col]
                for i in range(col + 1, size):
                    for p in range(col + 1, size):
                        value = pivot * matrix[i][p] - matrix[i][col] * matrix[col][p]
                        if col:
                            value, remainder = divmod(value, previous)
                            if remainder:
                                raise RuntimeError("nonexact Bareiss polynomial division")
                        if len(value) > self.determinant_max_terms:
                            raise OracleLimitError("determinant term limit reached; use oracle='milp'")
                        if deadline is not None and time.perf_counter() > deadline:
                            raise OracleLimitError("determinant soft time limit reached; use oracle='milp'")
                        matrix[i][p] = value
                    matrix[i][col] = zero
                previous = pivot
            return sign * matrix[-1][-1]

        rows = list(range(self.d))
        columns = list(range(self.d))
        poly = determinant(rows, columns)
        attainable = [(int(c), int(w)) for (c, w), coefficient in poly.terms()
                      if coefficient and c <= self.slack]
        if not attainable:
            # A polynomial identity-test failure can only hide feasible terms.
            median = self.median.copy()
            value = _integer_sum(labels[self.items, median])
            # Do not use a detected lower bound as an upper bound in this case.
            relaxed = np.where(self.allowed, -labels.astype(float), np.inf)
            _, permutation = linear_sum_assignment(relaxed)
            upper = _integer_sum(labels[self.items, permutation])
            return _OracleAnswer(median, value, upper, False, 'determinant-zero')
        target_c, target_w = max(attainable, key=lambda cw: (cw[1], -cw[0]))
        original_target = target_w
        ranking = np.empty(self.d, dtype=np.int64)
        for i in range(self.d):
            rows.remove(i)
            for p in columns:
                if not self.allowed[i, p]:
                    continue
                c = target_c - int(self.reduced[i, p])
                w = target_w - int(shifted[i, p])
                if c < 0 or w < 0:
                    continue
                remainder_columns = [q for q in columns if q != p]
                minor = determinant(rows, remainder_columns)
                if minor[(c, w)]:
                    ranking[i] = p
                    columns.remove(p)
                    target_c, target_w = c, w
                    break
            else:
                raise RuntimeError("determinant witness recovery failed")
        if target_c or target_w:
            raise RuntimeError("determinant witness has incorrect final exponents")
        value = _integer_sum(labels[self.items, ranking])
        if value != _integer_sum(row_min) + divisor * original_target:
            raise RuntimeError("determinant witness has incorrect objective")
        return _OracleAnswer(ranking, value, value, True, 'determinant')


class _Support:
    """Exact integer-weight distribution; duplicate vertices are merged."""
    def __init__(self, median):
        self.rankings = [median.copy()]
        self.index = {median.tobytes(): 0}
        self.weights = [1]
        self.distances = [[0]]
        self.weighted_distances = [0]
        self.mass = 1
        self.energy = 0  # sum_{a<b} weight[a]*weight[b]*F(a,b)

    def add(self, ranking, weight):
        key = ranking.tobytes()
        index = self.index.get(key)
        if index is None:
            distances = [_integer_sum(np.abs(ranking - other)) for other in self.rankings]
            g = sum(w * distance for w, distance in zip(self.weights, distances))
            index = len(self.rankings)
            for row, distance in zip(self.distances, distances):
                row.append(distance)
            self.distances.append(distances + [0])
            self.rankings.append(ranking.copy())
            self.weights.append(0)
            self.weighted_distances.append(g)
            self.index[key] = index
        else:
            g = self.weighted_distances[index]
        self.energy += weight * g
        for i in range(len(self.rankings)):
            self.weighted_distances[i] += weight * self.distances[i][index]
        self.weights[index] += weight
        self.mass += weight

    def round(self, k):
        # Conditional expectation, using integer scores and allowing repeats.
        previous_distances = [0] * len(self.rankings)
        selected = []
        diversity = 0
        for a in range(k):
            remaining = k - a - 1
            scores = [self.mass * old + remaining * g
                      for old, g in zip(previous_distances, self.weighted_distances)]
            index = max(range(len(scores)), key=scores.__getitem__)
            diversity += previous_distances[index]
            selected.append(self.rankings[index].copy())
            for i in range(len(previous_distances)):
                previous_distances[i] += self.distances[i][index]
        # Rounding may not decrease the expected value of k independent draws.
        if diversity * self.mass**2 < k * (k - 1) * self.energy:
            raise RuntimeError("conditional-expectation rounding decreased the expectation")
        return selected, diversity


def sum_dispersion_approx_rankings(
    rankings, k, alpha, *, oracle='milp', max_iterations=100,
    oracle_time_limit=None, allow_inexact_oracle=False,
    require_certificate=False, return_info=False, seed=0,
    determinant_max_d=16, determinant_max_terms=200_000,
    milp_backend='scipy', verbose=False,
):
    """Return k candidate->position dicts, each costing <=(1+alpha)*C*.

    max_iterations counts oracle calls, not returned rankings. None requests
    the paper's full R-1 calls. require_certificate demands the paper's exact
    1-1/k approximation factor, certified by a posteriori upper bounds or the
    completed theoretical iteration count. See module docstring for caveats.
    """
    if isinstance(k, bool) or int(k) != k or k < 2:
        raise ValueError('k must be an integer >=2')
    k = int(k)
    if oracle not in {'milp', 'determinant'}:
        raise ValueError("oracle must be 'milp' or 'determinant'")
    if milp_backend not in {'scipy', 'highspy'}:
        raise ValueError("milp_backend must be 'scipy' or 'highspy'")
    if max_iterations is not None and (int(max_iterations) != max_iterations or max_iterations < 0):
        raise ValueError('max_iterations must be a nonnegative integer or None')
    if oracle_time_limit is not None and (not math.isfinite(oracle_time_limit) or oracle_time_limit <= 0):
        raise ValueError('oracle_time_limit must be positive and finite')
    if determinant_max_d < 1 or determinant_max_terms < 1:
        raise ValueError('determinant resource limits must be positive')
    alpha = _alpha_fraction(alpha)
    started = time.perf_counter()
    candidates, offset, input_rows = _read_rankings(rankings)
    n, d = input_rows.shape
    costs = _aggregation_costs(input_rows)
    assignment = _BudgetedAssignment(costs, alpha, oracle, random.Random(seed),
                                     oracle_time_limit, allow_inexact_oracle,
                                     determinant_max_d, determinant_max_terms, milp_backend)
    # Input rankings are no longer needed; retaining only the d*d cost matrix
    # avoids holding extra internal copies throughout optimization.
    del input_rows
    support = _Support(assignment.median)
    labels = np.abs(np.arange(d)[None, :] - assignment.median[:, None]).astype(np.int64)
    theoretical_rounds = 4 * k**3 * (d * d // 2)
    target = Fraction(k - 1, k)
    upper = min(_global_diversity_bound(d, k),
                (k * (k - 1) // 2) * (2 * assignment.budget // n))
    best, best_diversity = support.round(k)
    certified = upper == 0
    all_optimal = True
    history = []
    limit = max(0, theoretical_rounds - 1)
    if max_iterations is not None:
        limit = min(limit, int(max_iterations))
    completed = 0
    for t in range(1, limit + 1):
        if certified:
            break
        answer = assignment.solve(labels)
        all_optimal &= answer.optimal
        # f* <= max_s E[F(s,X)] - f(mu), using an oracle upper bound.
        f = Fraction(support.energy, support.mass**2)
        distribution_upper = Fraction(answer.upper_bound, support.mass) - f
        upper = min(upper, max(0, math.floor(k * k * distribution_upper)),
                    k * (k - 1) * answer.upper_bound // support.mass)
        support.add(answer.ranking, t + 1)
        labels += (t + 1) * np.abs(np.arange(d)[None, :] - answer.ranking[:, None])
        rounded, diversity = support.round(k)
        if diversity > best_diversity:
            best, best_diversity = rounded, diversity
        if upper < best_diversity:
            raise RuntimeError('numerical oracle bound is inconsistent with a feasible output')
        certified = k * best_diversity >= (k - 1) * upper
        completed = t
        entry = {'iteration': t, 'oracle_method': answer.method,
                 'oracle_optimal': answer.optimal, 'support_size': len(support.rankings),
                 'sum_diversity': best_diversity, 'optimum_upper_bound': upper,
                 'target_certified': certified, 'elapsed_seconds': time.perf_counter() - started}
        history.append(entry)
        if verbose:
            import sys
            print(entry, file=sys.stderr, flush=True)
    full_schedule = completed >= max(0, theoretical_rounds - 1)
    # The original theorem also applies after its full iteration schedule if
    # all oracle calls were exact (with Monte Carlo success for determinants).
    theorem_completed = full_schedule and all_optimal
    certified |= theorem_completed
    output_costs = [assignment.cost(ranking) for ranking in best]
    if any(cost > assignment.budget for cost in output_costs):
        raise RuntimeError('output violates the aggregation budget')
    output = [{candidate: int(ranking[i]) + offset for i, candidate in enumerate(candidates)}
              for ranking in best]
    ratio = 1.0 if upper == 0 else best_diversity / upper
    if theorem_completed:
        ratio = max(ratio, float(target))
    info = {
        'n': n, 'd': d, 'k': k, 'alpha': str(alpha), 'oracle': oracle, 'milp_backend': milp_backend,
        'optimal_aggregation_cost': assignment.optimal_cost,
        'aggregation_budget': assignment.budget, 'output_aggregation_costs': output_costs,
        'reduced_budget': assignment.slack, 'allowed_edges': len(assignment.edge_i),
        'oracle_calls': completed, 'oracle_methods': dict(assignment.calls),
        'all_oracles_optimal': all_optimal, 'support_size': len(support.rankings),
        'sum_diversity': best_diversity, 'optimum_upper_bound': upper,
        'ratio_lower_bound': ratio, 'target_ratio': float(target),
        'target_certified': bool(certified), 'theoretical_support_rounds': theoretical_rounds,
        'full_theoretical_schedule_completed': theorem_completed,
        'stopped_reason': 'target_certified' if certified else 'iteration_cap',
        'certificate_type': ('conditional_monte_carlo' if oracle == 'determinant' and assignment.calls['determinant']
                             else 'numerical_highs' if assignment.calls['milp'] or assignment.calls['highspy'] else 'integer_assignment'),
        # Use the predetermined cap, not the random adaptive stopping time,
        # for the union bound over polynomial identity tests.
        'polynomial_identity_failure_bound': (
            min(1.0, d * limit / (2**61 - 2))
            if oracle == 'determinant' and (assignment.calls['determinant'] or assignment.calls['determinant-zero'])
            else 0.0),
        'elapsed_seconds': time.perf_counter() - started, 'history': history,
    }
    if not certified:
        message = (f'Iteration cap reached: achieved ratio lower bound {ratio:.6g}, '
                   f'below the requested {float(target):.6g}. '
                   'All returned rankings satisfy the aggregation budget; '
                   'the full diversity approximation guarantee is not certified.')
        if require_certificate:
            raise UncertifiedResultError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return (output, info) if return_info else output
