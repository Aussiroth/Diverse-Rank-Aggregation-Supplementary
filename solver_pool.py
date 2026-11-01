import cvxpy as cp
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from utils import find_tight_pairs
from utils import max_weight_perfect_matching

def solver_pool(n, tight_edges, k, candidates, positions, solver=cp.HIGHS):
    """
    Extracts up to k distinct perfect matchings using an ILP solver pool and no-good cuts.
    
    Args:
        n: Number of vertices in each partition.
        tight_edges: List of tuples (u, v) representing the tight edges.
        k: Maximum number of distinct matchings to return.
        solver: CVXPY mixed-integer solver to utilize.
    """
    num_edges = len(tight_edges)
    
    # Boolean variables for each tight edge
    x = cp.Variable(num_edges, boolean=True)
    
    constraints = []
    
    # Degree constraints: exactly 1 incident tight edge per vertex in U and V
    for u in range(n):
        incident_u = [i for i, edge in enumerate(tight_edges) if edge[0] ==  candidates[u]]
        constraints.append(cp.sum(x[incident_u]) == 1)
        
    for v in range(n):
        incident_v = [i for i, edge in enumerate(tight_edges) if edge[1] == positions[v]]
        constraints.append(cp.sum(x[incident_v]) == 1)
        
    matchings = []
    # Objective is arbitrary; we only care about feasibility inside the tight-edge subgraph
    objective = cp.Maximize(0)
    
    for _ in range(k):
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=solver)
        
        # Break if the solver exhausts all matchings or fails
        if prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            break
            
        # Extract the boolean assignments (rounding to handle floating point noise)
        sol = np.round(x.value)
        matched_edge_indices = np.where(sol == 1)[0]
        
        # Reconstruct the matching array where index u maps to node v
        m_arr = []
        for idx in matched_edge_indices:
            u, v = tight_edges[idx]
            m_arr.append((u,v))
            
        matchings.append(m_arr)
        
        # Add the no-good cut: prevent this exact set of edges from being chosen simultaneously
        cut_expr = cp.sum(x[matched_edge_indices]) <= n - 1
        constraints.append(cut_expr)
        
    return matchings

def solver_pool_rankings(rankings, k):
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
  perfect_matchings = solver_pool(nCandidates, tight_pairs, k, candidates, positions)

  # Converting to rankings.
  diverse_rankings = []
  for matching in perfect_matchings:
    diverse_rankings.append({i: p for i, p in matching})
  return diverse_rankings