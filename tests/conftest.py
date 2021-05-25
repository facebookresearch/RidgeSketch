"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

# conftest.py file to share fixtures between multiple test files
# https://docs.pytest.org/en/latest/fixture.html#conftest-py-sharing-fixture-functions
"""

import numpy as np
import pytest
from scipy.sparse import random as sprandom

from datasets.data_loaders import (
    BostonDataset,
    CaliforniaHousingDataset,
    YearPredictionDataset,
    Rcv1Dataset,
    TaxiDataset,
)


REAL_DATASETS = [
    BostonDataset,
    CaliforniaHousingDataset,
    YearPredictionDataset,
    Rcv1Dataset,
    TaxiDataset,
]


SEED = 0
np.random.seed(seed=SEED)


# skipping slow tests if --runslow not provided in cli
def pytest_addoption(parser):
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runslow"):
        # --runslow given in cli: do not skip slow tests
        return
    skip_slow = pytest.mark.skip(reason="need --runslow option to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


ALGO_MODES = ["auto", "mom", "accel"]

ALL_SOLVERS = [
    "direct",
    "cg",
    "subsample",
    "coordinate descent",
    "gaussian",
    "count",
    "subcount",
    "hadamard",
]

ALL_SOLVERS_EXCEPT_DIRECT = ALL_SOLVERS[1:]

SKETCH_SOLVERS = ALL_SOLVERS_EXCEPT_DIRECT[1:]

SKETCH_METHODS = [
    "SubsampleSketch",
    "CoordinateDescentSketch",
    "GaussianSketch",
    "CountSketch",
    "SubcountSketch",
    "HadamardSketch",
]


N_SAMPLES_SMALL = 100
N_FEATURES_SMALL = 20

N_SAMPLES_MIDSIZE = 500
N_FEATURES_MIDSIZE = 500

N_SAMPLES_LARGE = 10000
N_FEATURES_LARGE = 10000


@pytest.fixture(scope="session")
def X():
    n_samples = N_SAMPLES_SMALL
    n_features = N_FEATURES_SMALL
    return np.random.rand(n_samples, n_features)


@pytest.fixture(scope="session")
def y():
    n_samples = N_SAMPLES_SMALL
    return np.random.rand(n_samples, 1)


@pytest.fixture(scope="session", params=["dual", "primal"])
def X_dense(request):
    if request.param == "primal":
        n_samples = N_SAMPLES_LARGE
        n_features = N_FEATURES_MIDSIZE
    else:
        n_samples = N_SAMPLES_MIDSIZE
        n_features = N_FEATURES_LARGE
    return np.random.rand(n_samples, n_features)


@pytest.fixture(scope="session", params=["csc", "csr"])
def X_sparse(request):
    n_samples = N_SAMPLES_LARGE
    n_features = N_FEATURES_MIDSIZE
    sparse_format = request.param
    return sprandom(n_samples, n_features, density=0.1, format=sparse_format)


@pytest.fixture(scope="session", params=["primal", "dual"])
def data_dense(request):
    if request.param == "primal":
        n_samples = N_SAMPLES_LARGE
        n_features = N_FEATURES_MIDSIZE
    else:
        n_samples = N_SAMPLES_MIDSIZE
        n_features = N_FEATURES_LARGE
    return np.random.rand(n_samples, n_features), np.random.rand(n_samples, 1)


@pytest.fixture(scope="session", params=["primal_sparse", "dual_sparse"])
def data_sparse(request):
    if request.param == "primal_sparse":
        n_samples = N_SAMPLES_MIDSIZE
        n_features = N_FEATURES_SMALL
    else:
        n_samples = N_SAMPLES_SMALL
        n_features = N_FEATURES_MIDSIZE
    return (
        sprandom(n_samples, n_features, density=0.1, format="csr"),
        np.random.rand(n_samples, 1),
    )
