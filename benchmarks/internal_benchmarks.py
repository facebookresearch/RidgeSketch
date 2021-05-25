"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from scipy.sparse import random as sprandom
from timeit import default_timer
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
from itertools import product

from ridge_sketch import RidgeSketch


# Setup constants
ALPHA = 0.05
SMALL_DIMENSION = 2000
MEDIUM_DIMENSION = 10000
LARGE_DIMENSION = 50000


PROBLEMS = ["primal", "dual"]
OPERATOR_MODES = [True, False]
SOLVERS = ["gaussian", "subsample"]
SPARSE_FORMAT = "csr"
DENSITY = 0.01  # percentage of non-zeros in X


def run_benchmarks():
    """Returns pandas dataframe with benchmarks"""
    times = defaultdict(list)

    for problem in tqdm(PROBLEMS):
        if problem == "primal":
            n_samples = LARGE_DIMENSION
            n_features = MEDIUM_DIMENSION
        else:
            n_samples = MEDIUM_DIMENSION
            n_features = LARGE_DIMENSION

        X = sprandom(n_samples, n_features, density=DENSITY, format=SPARSE_FORMAT)
        y = np.random.rand(n_samples, 1)

        for solver, operator_mode in tqdm(product(SOLVERS, OPERATOR_MODES)):
            time_taken = compute_fit_time(
                times, X, y, SPARSE_FORMAT, problem, solver, operator_mode
            )
            benchmark = {
                "sparse_format": SPARSE_FORMAT,
                "problem": problem,
                "solver": solver,
                "operator_mode": operator_mode,
                "time": time_taken,
            }
            times = update_times(times, benchmark)

    df = pd.DataFrame(times)
    return df


def compute_fit_time(times, X, y, sparse_format, problem, solver, operator_mode):
    model = RidgeSketch(
        solver=solver, max_iter=10, operator_mode=operator_mode, alpha=ALPHA,
    )
    start = default_timer()
    model.fit(X, y)
    return default_timer() - start


def update_times(times, benchmark):
    """Updates times dict with new benchmark values"""
    for key, val in benchmark.items():
        times[key].append(val)
    return times


if __name__ == "__main__":
    df = run_benchmarks()
    print(df)
