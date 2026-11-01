#!/usr/bin/env python3
"""Synthetic complete rankings for diverse Spearman-footrule aggregation.

STDOUT contains exactly n rows of d space-separated integers, without a header. 
The ith row is the ith ranking over the candidates 0...d-1.
We only use uniform in the paper for scalability tests.

Examples:
  python generate_rankings.py -n 1000 -d 500 --model kendall --q 0.95 > data.txt
  python generate_rankings.py -n 1000 -d 500 --model footrule --alpha 10 --steps 100 > data.txt
  python generate_rankings.py -n 1000 -d 500 --model uniform > data.txt

Models:
  uniform: independent, exactly uniform permutations.
  kendall: independent exact Mallows samples P(r) proportional to q**K(r,c).
    --q accepts [0,1]; 0 returns the center, 1 is uniform. Alternatively
    --alpha a gives q=exp(-a/d), so P(r) is proportional to exp(-a*K(r,c)/d).
    Uses truncated geometric inversion counts and Fenwick-tree decoding,
  footrule: APPROXIMATE Mallows samples P(r) proportional to exp(-a*F(r,c)/d).
    a=0 is exactly uniform. Uses a hit-and-run chain to sample, therefore increasing steps
    increases accuracy of the sampling. See Zhong (2021), Section 2.2, for the kernel;
    its logarithmic mixing bound has a restricted parameter range.

See:
  Zhong (2021), Mallows permutation models with L1 and L2 distances I,
    Section 2.2: https://arxiv.org/abs/2112.13456
"""

import argparse
import math
import random
import sys
from typing import Iterator, List

def inverse_permutation(order: List[int]) -> List[int]:
    ranks = [0] * len(order)
    for position, item in enumerate(order):
        ranks[item] = position
    return ranks


class AvailablePositions:
    """Fenwick tree initially containing 1 in every position."""

    def __init__(self, size: int):
        self.size = size
        self.tree = [0] + [i & -i for i in range(1, size + 1)]

    def pop_kth(self, k: int) -> int:
        """Remove and return the position of the zero-based kth remaining 1."""
        index = 0
        bit = 1 << (self.size.bit_length() - 1)
        while bit:
            nxt = index + bit
            if nxt <= self.size and self.tree[nxt] <= k:
                k -= self.tree[nxt]
                index = nxt
            bit >>= 1
        update = index + 1
        while update <= self.size:
            self.tree[update] -= 1
            update += update & -update
        return index


def kendall_ranks(d: int, log_q: float, rng: random.Random) -> List[int]:
    """Exact Mallows-Kendall rank vector around identity, up to float precision.

    At insertion stage m, V_m has probability q**v/sum(q**j, j=0..m-1).
    Decode stages d..1 into free slots, avoiding quadratic Python list inserts.
    """
    if log_q == -math.inf:
        return list(range(d))
    if log_q == 0:
        ranks = list(range(d))
        rng.shuffle(ranks)
        return ranks
    free = AvailablePositions(d)
    ranks = [0] * d
    for m in range(d, 0, -1):
        mass = -math.expm1(m * log_q)  # 1 - q**m, stable near q=1
        v = int(math.log1p(-rng.random() * mass) / log_q)
        v = min(v, m - 1)  # guard against endpoint floating-point rounding
        ranks[m - 1] = free.pop_kth(m - 1 - v)
    return ranks


def footrule_step(ranks: List[int], beta: float, rng: random.Random) -> List[int]:
    """Zhong's hit-and-run kernel for exp(-beta*F(r,identity)).

    Sample E_i~Exp(1), B_i=floor(max(i,r[i])+E_i/(2*beta)), capped at d-1.
    Assign positions d-1..0 uniformly among unassigned items with B_i>=position.
    The allowed sets are nested, so this produces a uniform feasible matching.
    """
    d = len(ranks)
    buckets: List[List[int]] = [[] for _ in range(d)]
    rate = 2 * beta
    for item, rank in enumerate(ranks):
        base = max(item, rank)
        # Avoid exponentiating tiny probabilities or forming unbounded integers.
        exponential = -math.log1p(-rng.random())
        slack = exponential / rate
        bound = d - 1 if slack >= d - 1 - base else base + int(slack)
        buckets[bound].append(item)
    available: List[int] = []
    result = [0] * d
    for position in range(d - 1, -1, -1):
        available.extend(buckets[position])
        selected = rng.randrange(len(available))
        item = available[selected]
        available[selected] = available[-1]
        available.pop()
        result[item] = position
    return result


def footrule_ranks(d: int, alpha: float, steps: int, initial: str,
                   rng: random.Random) -> List[int]:
    ranks = list(range(d))
    if alpha == 0 or initial == "random":
        rng.shuffle(ranks)
    elif initial == "reverse":
        ranks.reverse()
    if alpha == 0 or d == 1:
        return ranks
    beta = alpha / d
    if beta == 0:  # parameter underflow: indistinguishable from uniform here
        rng.shuffle(ranks)
        return ranks
    for _ in range(steps):
        ranks = footrule_step(ranks, beta, rng)
    return ranks


def balanced_blocks(n: int, d: int, b: int, center: List[int],
                    rng: random.Random) -> Iterator[List[int]]:
    full_end = (d // b) * b
    for _ in range(n // b):
        base = center[:]
        for start in range(0, full_end, b):
            block = base[start:start + b]
            rng.shuffle(block)
            base[start:start + b] = block
        for shift in range(b):
            ranks = [0] * d
            for start in range(0, full_end, b):
                for offset in range(b):
                    item = base[start + (offset + shift) % b]
                    ranks[item] = start + offset
            for position in range(full_end, d):
                ranks[base[position]] = position
            yield ranks


def generate(args: argparse.Namespace) -> Iterator[List[int]]:
    rng = random.Random(args.seed)
    center = list(range(args.d))  # item IDs in central preference order
    if args.random_center:
        random.Random(args.center_seed).shuffle(center)
    if args.model == "balanced-blocks":
        yield from balanced_blocks(args.n, args.d, args.block_size, center, rng)
        return
    if args.q is None:
        log_q = -args.alpha / args.d
    else:
        log_q = -math.inf if args.q == 0 else math.log(args.q)
    for _ in range(args.n):
        if args.model == "uniform":
            ranks = list(range(args.d))
            rng.shuffle(ranks)
            yield ranks
            continue
        reverse = args.centers == 2 and rng.random() >= args.weight
        if args.model == "kendall":
            relative = kendall_ranks(args.d, log_q, rng)
        else:
            relative = footrule_ranks(args.d, args.alpha, args.steps, args.initial, rng)
        # Relative coordinate i refers to the item in central position i.
        # Reversing the center is a relabeling, not a reversal of output ranks.
        ranks = [0] * args.d
        for i, rank in enumerate(relative):
            item = center[args.d - 1 - i] if reverse else center[i]
            ranks[item] = rank
        yield ranks


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print n complete rankings, each a permutation of 0..d-1.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="See the module docstring for models, sampling limits, and references.")
    parser.add_argument("-n", type=int, required=True, help="number of rankings")
    parser.add_argument("-d", type=int, required=True, help="number of items")
    parser.add_argument("--model", choices=["uniform", "kendall", "footrule", "balanced-blocks"],
                        default="kendall", help="generation model")
    dispersion = parser.add_mutually_exclusive_group()
    dispersion.add_argument("--alpha", type=float, default=None,
                            help="concentration in exp(-alpha*distance/d); default 5")
    dispersion.add_argument("--q", type=float, default=None,
                            help="Kendall Mallows q in [0,1]; excludes --alpha")
    parser.add_argument("--centers", type=int, choices=[1, 2], default=1,
                        help="one center or a mixture of a center and its reverse")
    parser.add_argument("--weight", type=float, default=0.5,
                        help="probability of first center in a two-center mixture")
    parser.add_argument("--steps", type=int, default=100,
                        help="hit-and-run transitions PER independent footrule sample")
    parser.add_argument("--initial", choices=["random", "identity", "reverse"], default="random",
                        help="initial relative state of EACH footrule chain")
    parser.add_argument("--block-size", type=int, default=10,
                        help="balanced block width; must divide n and be <=d")
    parser.add_argument("--format", choices=["ranks", "order"], default="ranks",
                        help="positions indexed by item, or item IDs in preference order")
    parser.add_argument("--seed", type=int, default=0, help="sampling random seed")
    parser.add_argument("--random-center", action="store_true",
                        help="use a random common item labeling/central order")
    parser.add_argument("--center-seed", type=int, default=0, help="common center random seed")
    args = parser.parse_args(argv)
    if args.n < 0 or args.d < 1:
        parser.error("require n >= 0 and d >= 1")
    if args.alpha is None:
        args.alpha = 5.0
    if not math.isfinite(args.alpha) or args.alpha < 0:
        parser.error("alpha must be finite and nonnegative")
    if args.q is not None:
        if args.model != "kendall":
            parser.error("--q is only supported for --model kendall")
        if not math.isfinite(args.q) or not 0 <= args.q <= 1:
            parser.error("q must lie in [0,1]")
    if not math.isfinite(args.weight) or not 0 <= args.weight <= 1:
        parser.error("weight must lie in [0,1]")
    if args.steps < 1:
        parser.error("steps must be at least 1")
    if args.centers != 1 and args.model not in ("kendall", "footrule"):
        parser.error("two centers are supported only for Mallows models")
    if args.model == "balanced-blocks":
        if not 1 <= args.block_size <= args.d:
            parser.error("block-size must lie between 1 and d")
        if args.n % args.block_size:
            parser.error("balanced-blocks requires n divisible by block-size")
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    for ranks in generate(args):
        row = ranks if args.format == "ranks" else inverse_permutation(ranks)
        sys.stdout.write(" ".join(map(str, row)) + "\n")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Allow piping into head without a traceback or extra stdout content.
        sys.exit(0)
