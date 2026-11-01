import numpy as np
from scipy.optimize import linprog
from scipy.optimize import linear_sum_assignment
import networkx as nx
import random
import itertools
from itertools import combinations
import math
from utils import find_tight_pairs
from utils import max_weight_perfect_matching

def solve_via_network_flow(candidates, positions, E0, k, lambda_t):
    """
    Solves the sparse bipartite matching problem using Min-Cost Max-Flow.

    :param candidates: List of candidate IDs (e.g., [0, 1, 2])
    :param positions: List of position values (e.g., [10.5, 20.0, 15.5])
                      Note: The position ID is assumed to be its numerical value p.
    :param E0: Dictionary mapping candidate ID to a list of allowed position values.
    :param k: Integer, number of iterations/slots.
    :param lambda_t: List/Array of length k containing lambda multipliers.
    :return: Dictionary mapping (i, t, p) to flow value (0 or 1).
    """

    allowed_positions = {i: [] for i in candidates}
    for i, p in E0:
        if i in allowed_positions:
            allowed_positions[i].append(p)

    G = nx.DiGraph()
    S, T = "Source", "Sink"

    # Source to Candidates (Supply = 1)
    for i in candidates:
        for t in range(k):
            u_node = ("cand", i, t)
            G.add_edge(S, u_node, capacity=1, weight=0)

            # Candidates to Positions (Sparse edges from E0)
            for p in allowed_positions[i]:
                v_node = ("pos", p)
                # Maximize lambda_t * p -> Minimize -lambda_t * p
                cost = -lambda_t[t] * p
                G.add_edge(u_node, v_node, capacity=1, weight=cost)

    # Positions to Sink (Demand = k)
    for p in positions:
        v_node = ("pos", p)
        G.add_edge(v_node, T, capacity=k, weight=0)

    # Solve Minimum Cost Maximum Flow
    try:
        flow_dict = nx.max_flow_min_cost(G, S, T)
    except nx.NetworkXUnfeasible:
        raise ValueError("The problem is infeasible. Hall's condition is likely violated by E0.")

    # Verify that maximum flow was actually reached
    total_flow = sum(flow_dict.get(S, {}).get(("cand", i, t), 0) for i in candidates for t in range(k))
    if total_flow != len(candidates) * k:
        raise ValueError(f"Infeasible: Flow reached {total_flow}, expected {len(candidates) * k}")

    # Extract the non-zero assignments
    assignments = {}
    for i in candidates:
        for t in range(k):
            u_node = ("cand", i, t)
            for p in allowed_positions[i]:
                v_node = ("pos", p)
                # If flow is pushed along this edge, record it
                flow_val = flow_dict.get(u_node, {}).get(v_node, 0)
                if flow_val > 0:
                    assignments[(i, t, p)] = flow_val

    return assignments

def extract_k_perfect_matchings(assignments, candidates, k):
    """
    Decomposes the flow assignments into exactly k perfect matchings.
    """
    # Collapse the multiplicity index t to find total edge multiplicities x_{i,p}
    edge_counts = {}
    for (i, t, p), flow_val in assignments.items():
        if flow_val > 0:
            edge_counts[(i, p)] = edge_counts.get((i, p), 0) + int(flow_val)

    perfect_matchings = []

    # Iteratively extract perfect matchings
    for _ in range(k):
        # Build the residual graph for this iteration
        G = nx.Graph()
        top_nodes = []

        for (i, p), count in edge_counts.items():
            if count > 0:
                c_node = ("cand", i)
                p_node = ("pos", p)
                G.add_edge(c_node, p_node)

                if c_node not in top_nodes:
                    top_nodes.append(c_node)

        # Find a maximum matching using Hopcroft-Karp (default in nx)
        # Because the graph is regular, this is guaranteed to be a perfect matching.
        matching = nx.bipartite.maximum_matching(G, top_nodes=top_nodes)

        # Extract the edges of this matching and update the residual capacities
        current_matching = []
        for i in candidates:
            c_node = ("cand", i)
            if c_node in matching:
                # matching[c_node] returns the paired ("pos", p) node
                p_val = matching[c_node][1]
                current_matching.append((i, p_val))

                # Decrement the multiplicity for the next iteration
                edge_counts[(i, p_val)] -= 1

        perfect_matchings.append(current_matching)

    return perfect_matchings

def sum_dispersion_rankings(rankings, k):
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

  # Calculating the \lambda_t values
  lambda_t = np.array([2*(t+1)-k-1 for t in range(k)])

  # Solving the assignment problem using min cost max flow
  assignments = solve_via_network_flow(candidates, positions, tight_pairs, k, lambda_t)

  # Extracting k perfect matchings
  perfect_matchings = extract_k_perfect_matchings(assignments, candidates, k)

  # Converting to rankings.
  diverse_rankings = []
  for matching in perfect_matchings:
    diverse_rankings.append({i: p for i, p in matching})
  return diverse_rankings