"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import pytest
import numpy as np
from scipy.sparse import issparse


from ridge_sketch import RidgeSketch
from datasets.data_loaders import (
    BostonDataset,
    Rcv1Dataset,
    TaxiDataset,
)
from datasets.generate_controlled_data import generate_data_cd


from tests.conftest import REAL_DATASETS


def test_boston():
    """
    Tests loading the boston dataset and its matrices.

    Checks the shape and the format.
    """
    test_small_load(BostonDataset)
    test_large_load(BostonDataset)


@pytest.mark.slow
@pytest.mark.parametrize("Dataset", REAL_DATASETS)
def test_small_load(Dataset):
    """
    Tests loading first 100 rows of a dataset and its matrices.

    Checks the shape and the format.
    """
    try:
        dataset = Dataset()
        X, y = dataset.load_X_y()
    except FileNotFoundError:
        pytest.skip("dataset file not found")

    dataset = Dataset(is_small=True)
    X, y = dataset.load_X_y()
    n_samples, n_features = dataset.get_dim()
    assert n_samples == 100
    assert n_samples == X.shape[0]
    assert n_features == X.shape[1]
    assert n_samples == y.shape[0]

    if issparse(X):
        assert dataset.get_sparse_format() == X.format
    else:
        assert dataset.get_sparse_format() == "dense"


@pytest.mark.slow
@pytest.mark.parametrize("Dataset", REAL_DATASETS)
def test_large_load(Dataset):
    """
    Tests loading of a large dataset and its matrices.

    Checks the shape and the format.
    """
    try:
        dataset = Dataset()
        X, y = dataset.load_X_y()
    except FileNotFoundError:
        pytest.skip("dataset file not found")
    n_samples, n_features = dataset.get_dim()
    assert n_samples == X.shape[0]
    assert n_features == X.shape[1]
    assert n_samples == y.shape[0]

    if issparse(X):
        assert dataset.get_sparse_format() == X.format
    else:
        assert dataset.get_sparse_format() == "dense"


@pytest.mark.ridgesolver
class TestRidgeSketchRealData:
    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ["direct"])
    @pytest.mark.parametrize("Dataset", REAL_DATASETS)
    def test_direct_solver_real_dataset(self, solver_name, Dataset):
        """Tests direct solver solutions on real data"""
        if Dataset in [Rcv1Dataset, TaxiDataset]:
            is_small = True
        else:
            is_small = False

        try:
            dataset = Dataset(is_small=is_small)
            X_t, y_t = dataset.load_X_y()
        except FileNotFoundError:
            pytest.skip("dataset file not found")

        X_t, y_t = dataset.load_X_y()
        self._verify_solution(solver_name, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ["subsample"])
    @pytest.mark.parametrize("Dataset", REAL_DATASETS)
    def test_all_sketch_solvers_real_dataset(self, solver_name, Dataset):
        """Tests sketch solvers solutions on real data"""
        if Dataset in [Rcv1Dataset, TaxiDataset]:
            # Use only the first 100 rows for the largest datasets
            is_small = True
        else:
            is_small = False

        try:
            dataset = Dataset(is_small=is_small)
            X_t, y_t = dataset.load_X_y()
        except FileNotFoundError:
            pytest.skip("dataset file not found")
        X_t, y_t = dataset.load_X_y()
        self._verify_solution(solver_name, X_t, y_t)

    def _verify_solution(self, solver_name, X_input, y_input):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        model = RidgeSketch(
            solver=solver_name, fit_intercept=True, tol=1e-3, max_iter=20000,
        )
        model.alpha = 0.1
        model.fit(X_input, y_input)
        assert model.residual_norms[-1] < model.tol


@pytest.mark.parametrize("shape", [(100, 10), (10, 100)])
@pytest.mark.parametrize("lmbda_min_target", [1e-1, 1e-4, 1e-8])
def test_generated_data_cd(shape, lmbda_min_target):
    """
    Tests the generated data for a given smallest eigenvalue of A.

    Checks the shape and the smallest eigenvalue for primal and dual forms.
    """
    eigen_scale = 100
    n_samples, n_features = shape
    X, y, lmbda = generate_data_cd(
        n_samples, n_features, lmbda_min_target, eigen_scale,
    )

    if n_features > n_samples:
        # dual system matrix
        A = X @ X.T + lmbda * np.eye(n_samples)
    else:
        # primal system matrix
        A = X.T @ X + lmbda * np.eye(n_features)
    eigen_val_A = np.sort(np.linalg.eigvals(A))
    lmbda_min = eigen_val_A[0]
    lmbda_max = eigen_val_A[-1]

    # Check that extreme eigenvalues of A corresponds to the targets
    assert lmbda_min == pytest.approx(lmbda_min_target, 0.01)
    assert lmbda_max == pytest.approx(eigen_scale * lmbda_min_target, 0.01)

    # Check shapes
    m = min(n_samples, n_features)
    assert n_samples == X.shape[0]
    assert n_features == X.shape[1]
    assert n_samples == y.shape[0]

    assert (m, m) == A.shape
