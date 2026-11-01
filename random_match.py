import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import find_tight_pairs
from utils import max_weight_perfect_matching

def random_match(n, tight_edges, candidates, positions, k, time_budget=None, distinct_only=False):
    # 1. Extract unique labels and map them to safe internal indices 0..n-1
    # Sorting by string representation ensures deterministic internal indexing
    # 1. Map items to their list indices
    row_to_idx = {candidate: i for i, candidate in enumerate(candidates)}
    col_to_idx = {position: i for i, position in enumerate(positions)}

    # 2. Map indices back to the items
    idx_to_row = {i: candidate for candidate, i in row_to_idx.items()}
    idx_to_col = {i: position for position, i in col_to_idx.items()}
    
    # 2. Translate edges to internal coordinates for NumPy matrix slicing
    int_rows = [row_to_idx[u] for u, v in tight_edges]
    int_cols = [col_to_idx[v] for u, v in tight_edges]

    start_time = time.time()
    matchings = []
    seen = set()
    
    while len(matchings) < k:
        if time_budget is not None and (time.time() - start_time) >= time_budget:
            break
            
        weights = np.random.rand(len(tight_edges))
        
        # Matrix is strictly n x n, using internal indices
        W = np.full((n, n), -1e6)
        W[int_rows, int_cols] = weights
        
        row_ind, col_ind = linear_sum_assignment(W, maximize=True)
        
        if W[row_ind, col_ind].sum() < -1e5:
            continue
            
        # 3. Map back to the original row and column labels
        matching_mapping = tuple((idx_to_row[r], idx_to_col[c]) for r, c in zip(row_ind, col_ind))
        
        if distinct_only:
            if matching_mapping not in seen:
                seen.add(matching_mapping)
                matchings.append(matching_mapping)
        else:
            matchings.append(matching_mapping)
            
    return matchings

def random_match_rankings(rankings, k, distinct_only=False):
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

  # Extracting k perfect matchings
  perfect_matchings = random_match(nCandidates, tight_pairs, candidates, positions, k, time_budget=None, distinct_only=distinct_only)

  # Converting to rankings.
  diverse_rankings = []
  for matching in perfect_matchings:
    diverse_rankings.append({i: p for i, p in matching})
  return diverse_rankings