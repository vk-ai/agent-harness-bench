from __future__ import annotations

import math
import random


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return [math.nan, math.nan]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def bootstrap_mean(xs, iters: int = 2000, seed: int = 0):
    xs = list(xs)
    if not xs:
        return math.nan, [math.nan, math.nan]
    rng = random.Random(seed)
    n = len(xs)
    ms = sorted(sum(rng.choice(xs) for _ in range(n)) / n for _ in range(iters))
    return round(sum(xs) / n, 3), [round(ms[int(0.025 * iters)], 3), round(ms[int(0.975 * iters) - 1], 3)]
