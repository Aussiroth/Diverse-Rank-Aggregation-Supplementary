from utils import min_dispersion, sum_dispersion, max_dispersion, dispersion_kt, kendall_tau_distance, footrule_distance
from dataset_to_rank import getDataset
from sum_dispersion_approx import sum_dispersion_approx_rankings
import numpy as np
import random

import sys

import time

def getKendallTauMetrics(diverse_ranks, candidates, rankings):
    dispersion_metrics_kt = dispersion_kt(diverse_ranks, candidates)
    kt_costs = []
    for rank in diverse_ranks:
        kt_cost = np.sum([kendall_tau_distance(rank, rank_, candidates) for rank_ in rankings])
        kt_costs.append(kt_cost)
    return dispersion_metrics_kt, kt_costs

def getSpearmanFootruleMetrics(diverse_ranks, candidates, rankings):
    sum_disp = sum_dispersion(diverse_ranks, candidates)
    min_disp = min_dispersion(diverse_ranks, candidates)
    max_disp = max_dispersion(diverse_ranks, candidates)
    sf_costs = []
    for rank in diverse_ranks:
        sf_cost = np.sum([footrule_distance(rank, rank_, candidates) for rank_ in rankings])
        sf_costs.append(sf_cost)
    return sum_disp, min_disp, max_disp, sf_costs

if __name__ == "__main__":

    K = int(sys.argv[1])
    if K < 2:
        exit(1)

    np.random.seed(0)
    random.seed(0)

    dataset = input().rstrip()
    dataset_file = input().rstrip()
    rankings = getDataset(dataset_file)
    candidates = list(rankings[0].keys())

    for alpha in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        sTime = time.time()
        diverse_ranks = sum_dispersion_approx_rankings(rankings, K, alpha=alpha)
        time_taken = time.time() - sTime
        sum_disp, min_disp, max_disp, sf_costs = getSpearmanFootruleMetrics(diverse_ranks, candidates, rankings)
        sf_costs_str = ', '.join([str(cost) for cost in sf_costs])
        print(f"{dataset}, {alpha}, Approx-Sum, {K}, {sum_disp}, {min_disp}, {time_taken}, {sf_costs_str}")
        dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks, candidates, rankings)
        kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
        print(f"Kendall-Tau, {dataset}, {alpha}, Approx-Sum, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )