"""Small local benchmark for repeated vs shared performance preparation."""

from time import perf_counter

import numpy as np

from rtichoke import prepare_performance_data


def run(n: int = 100_000, repeats: int = 5) -> None:
    rng = np.random.default_rng(2026)
    reals = rng.binomial(1, 0.25, n)
    probs = {"model": np.clip(0.1 + 0.65 * reals + rng.normal(0, 0.18, n), 0, 1)}

    start = perf_counter()
    for _ in range(repeats):
        prepare_performance_data(probs, reals)
    repeated = perf_counter() - start

    start = perf_counter()
    performance_data = prepare_performance_data(probs, reals)
    for _ in range(repeats):
        _ = performance_data
    shared = perf_counter() - start

    print(f"n={n:,}; repeated={repeated:.3f}s; shared={shared:.3f}s; ratio={repeated/shared:.2f}x")


if __name__ == "__main__":
    run()
