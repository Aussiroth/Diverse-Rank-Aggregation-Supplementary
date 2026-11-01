from utils import kendall_tau_distance
import numpy as np

def kt_objective(median, rankings):
    cost = 0
    candidates = [i for i in range(1, len(median)+1)]
    for ranking in rankings:
        cost += kendall_tau_distance(median, ranking, candidates)
    return cost

#assume candidates are labeled 1...d
def get_frac_tournament(rankings):
    element_count = len(rankings[0])
    frac_tournament = np.zeros((element_count+1, element_count+1))

    for ranking in rankings:
        for i in range(1, len(ranking)+1):
            ipos = ranking[i]
            for j in range(i+1, len(ranking)+1):
                jpos = ranking[j]
                if ipos < jpos:
                    frac_tournament[i][j] += 1
                else:
                    frac_tournament[j][i] += 1
    for i in range(element_count):
        for j in range(element_count):
            frac_tournament[i][j] = frac_tournament[i][j] / len(rankings)

    return frac_tournament

def find_one_ranking(rankings, rng, frac_tournament):
    candidates = [i+1 for i in range(len(rankings[0]))]
    result = ks(candidates, rng, frac_tournament)
    result_dict = {}
    for i in range(len(result)):
        result_dict[result[i]] = i+1
    result_obj = kt_objective(result_dict, rankings)
    pick_a_perm = rankings[rng.integers(len(rankings))]
    pick_obj = kt_objective(pick_a_perm, rankings)

    if result_obj < pick_obj:
        return result_dict
    return pick_a_perm

def ks(candidates, rng, frac_tournament):
    if len(candidates) <= 1:
        return candidates
    pivotidx = rng.integers(len(candidates))
    pivot = candidates[pivotidx]
    left = []
    right = []
    for c in candidates:
        if c != pivot:
            if frac_tournament[c][pivot] >= 0.5:
                left.append(c)
            else:
                right.append(c)
    return ks(left, rng, frac_tournament) + [pivot] + ks(right, rng, frac_tournament)

def generate_diverse_kendall_tau_rankings(
    rankings,
    k,
    seed=None,
    unique=True,
    max_attempts=None,
    pick_a_perm="random"
):
    output = []
    rng = np.random.default_rng(seed)
    frac_tournament = get_frac_tournament(rankings)
    while len(output) < k:
        ranking = find_one_ranking(rankings, rng, frac_tournament)
        #convert to dictionary
        if ranking not in output:
            output.append(ranking)
    return output