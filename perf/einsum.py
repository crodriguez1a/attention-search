import numpy as np
from tqdm import tqdm
import time


def optim_no(a, b):
    # Basic `einsum`: ~1520ms  (benchmarked on 3.1GHz Intel i5.)
    np.einsum(
        "ijkl,jkl->ikl", a, b,
    )


def optim_yes(a, b):
    # Sub-optimal `einsum` (due to repeated path calculation time): ~330ms
    np.einsum("ijkl,jkl->ikl", a, b, optimize="optimal")


def optim_greedy(a, b):
    # Greedy `einsum` (faster optimal path approximation): ~160ms
    np.einsum("ijkl,jkl->ikl", a, b, optimize="greedy")


def optim_path(a, b):
    # Optimal `einsum` (best usage pattern in some use cases): ~110ms
    path = np.einsum_path("ijkl,jkl->ikl", a, b, optimize="optimal")[0]
    np.einsum("ijkl,jkl->ikl", a, b, optimize=path)


if __name__ == "__main__":
    # TODO: document cProfiler
    """
    ```
    python -m cProfile -o out.profile perf/einsum.py __main__
    snakeviz out.profile
    ```
    """

    for i in tqdm(range(10000, 50000, 10000)):
        print(i)

        b = np.zeros((3, 100, 100))
        a = np.zeros((i, 3, 100, 100))

        tic = time.perf_counter()
        optim_no(a, b)
        toc = time.perf_counter()
        print(f"optim_no in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        optim_yes(a, b)
        toc = time.perf_counter()
        print(f"optim_yes in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        optim_greedy(a, b)
        toc = time.perf_counter()
        print(f"optim_greedy in {toc - tic:0.4f} seconds")

        tic = time.perf_counter()
        optim_path(a, b)
        toc = time.perf_counter()
        print(f"optim_path in {toc - tic:0.4f} seconds")

        print("")
