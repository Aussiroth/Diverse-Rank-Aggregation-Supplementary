from utils import min_dispersion, sum_dispersion, max_dispersion, dispersion_kt, kendall_tau_distance
from sum_dispersion import sum_dispersion_rankings 
from min_dispersion_mwu import min_dispersion_approx_rankings_MWU
from dataset_to_rank import getDataset
from random_match import random_match_rankings
from solver_pool import solver_pool_rankings
from cp_sat import CP_SAT_tight_min_ranking,CP_SAT_tight_sum_ranking
from hand_kwiksort import generate_diverse_kendall_tau_rankings
import numpy as np
import random

import sys

import time
np.random.seed(0)
random.seed(0)

def getKendallTauMetrics(diverse_ranks, candidates, rankings):
    dispersion_metrics_kt = dispersion_kt(diverse_ranks, candidates)
    kt_costs = []
    for rank in diverse_ranks:
        kt_cost = np.sum([kendall_tau_distance(rank, rank_, candidates) for rank_ in rankings])
        kt_costs.append(kt_cost)
    return dispersion_metrics_kt, kt_costs

def getSpearmanFootruleMetrics(diverse_ranks, candidates, rankings):
    sum_disp = int(sum_dispersion(diverse_ranks, candidates))
    min_disp = int(min_dispersion(diverse_ranks, candidates))
    return sum_disp, min_disp

if __name__ == "__main__":

    K = int(sys.argv[1])
    if K < 2:
        exit(1)

    dataset_file = input().rstrip()
    rankings = getDataset(dataset_file)
    candidates = list(rankings[0].keys())

    print("Dataset, Algorithm, K, Sum Dispersion, Min Dispersion, Time Taken (s), Kendall-Tau Dispersion, Kendall-Tau Costs")

    sTime = time.time()
    diverse_ranks = CP_SAT_tight_min_ranking(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks, candidates, rankings)
    print(f"{dataset_file},CP SAT Tight - Min, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file},CP SAT Tight - Min, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )

    sTime = time.time()
    diverse_ranks = CP_SAT_tight_sum_ranking(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks, candidates, rankings)
    print(f"{dataset_file},CP SAT Tight - Sum, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file},CP SAT Tight - Sum, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )
    
    sTime = time.time()
    diverse_ranks_sum = sum_dispersion_rankings(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks_sum, candidates, rankings)
    print(f"{dataset_file},MaxSumDiversity, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks_sum, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file},MaxSumDiversity, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )

    sTime = time.time()
    diverse_ranks_slp = solver_pool_rankings(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks_slp, candidates, rankings)
    print(f"{dataset_file},Solver Pool, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks_slp, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file},Solver Pool, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )


    sTime = time.time()
    diverse_ranks_rnd = random_match_rankings(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks_rnd, candidates, rankings)
    print(f"{dataset_file},Random Match, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks_rnd, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file},Random Match, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )

    sTime = time.time()
    diverse_ranks_min_mwu = min_dispersion_approx_rankings_MWU(rankings, K)
    time_taken = time.time() - sTime
    sum_disp, min_disp = getSpearmanFootruleMetrics(diverse_ranks_min_mwu, candidates, rankings)
    print(f"{dataset_file}, MWU, {K}, {sum_disp}, {min_disp}, {time_taken}")
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks_min_mwu, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file}, MWU, {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )

    sTime = time.time()
    diverse_ranks_kwiksort_kt = generate_diverse_kendall_tau_rankings(rankings, K, 0)
    time_taken = time.time() - sTime
    dispersion_metrics_kt, kt_costs = getKendallTauMetrics(diverse_ranks_kwiksort_kt, candidates, rankings)
    kt_costs_str = ', '.join([str(cost) for cost in kt_costs])
    print(f"Kendall-Tau, {dataset_file}, Solver Pool (KT), {K}, {dispersion_metrics_kt[0]}, {dispersion_metrics_kt[1]}, {time_taken}, {kt_costs_str}" )
    