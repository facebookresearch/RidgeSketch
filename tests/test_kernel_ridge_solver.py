"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from numpy.testing import assert_allclose
import pytest

# This is just for constructing the Kernel matrix for testing the
from scipy.spatial.distance import pdist, squareform

# for testing RidgeSketch against sklearn
from kernel_ridge_sketch import KernelRidgeSketch
from tests.conftest import ALL_SOLVERS_EXCEPT_DIRECT


@pytest.mark.kernelridgesolver
class TestKernelRidgeSketch:
    def test_instantiation(self):
        """Confirms RidgeSolver can be instantiated"""
        _ = KernelRidgeSketch()

    def test_attributes(self):
        """Confirms class contains expected attributes"""
        model = KernelRidgeSketch()
        assert model.alpha == 1.0
        # check solver is set to sparse_cg by default
        assert model.solver == "cg"

    def test_set_sketch_size(self, X):
        """Confirms sketch_size is correctly set"""
        # check that a initialization it is None
        model = KernelRidgeSketch()
        assert model.sketch_size is None

        # check that default set value is square root of m
        m = min(X.shape)
        model.set_sketch_size(X)
        assert model.sketch_size == int(np.sqrt(m))

        # check that value given by user is correctly passed
        model = KernelRidgeSketch(sketch_size=7)
        assert model.sketch_size == 7
        model.set_sketch_size(X)
        assert model.sketch_size == 7

        # set to value larger than m should raise an error
        model = KernelRidgeSketch(sketch_size=23.2)
        with pytest.raises(TypeError):
            model.set_sketch_size(X)
        # set to value greater than m should raise an error
        model = KernelRidgeSketch(sketch_size=1000000000)
        with pytest.raises(ValueError):
            model.set_sketch_size(X)

    def test_fit_sets_coef(self, X, y):
        """Confirms fit with default parameters sets coef_ attribute"""
        model = KernelRidgeSketch()
        model.fit(X, y)
        assert hasattr(model, "coef_")

    @pytest.mark.parametrize("kernel_name", ["Matern", "RBF"])
    def test_build_kernel(self, kernel_name, X, y):
        """
        Testing the construction of the kernel matrix
        of sklearn against by hand construction
        """
        model = KernelRidgeSketch(kernel=kernel_name, kernel_nu=1.5)
        model.fit(X, y)
        # Build kernel using sklearn
        K = model.K_class(X=X)  # add the nu parameter to specify

        # Build the kernel matrix by hand
        s = model.kernel_sigma
        pairwise_dists = squareform(pdist(X, "euclidean"))
        if kernel_name == "Matern":
            # Building the Matern "Once differentiable" Kernel (nu =1.5)
            Ktest = (1 + pairwise_dists * np.sqrt(3) / s) * np.exp(
                -pairwise_dists * np.sqrt(3) / s
            )
        else:
            Ktest = np.exp(-(pairwise_dists ** 2) / (2 * s ** 2))

        assert_allclose(Ktest, K, rtol=1e-6)

    def test_fit_solution(self, X, y):
        """Compare iterative coef_ after fit to closed-form solution"""
        # cg's solution
        model = KernelRidgeSketch(tol=1e-12)
        model.fit(X, y)
        coef_sklearn = model.coef_

        # closed-form solution via 'linalg.solve'
        ds = KernelRidgeSketch(solver="direct")
        ds.fit(X, y)
        ds_coef = ds.coef_

        assert_allclose(coef_sklearn, ds_coef, rtol=1e-6)

    @pytest.mark.parametrize("kernel_name", ["Matern", "RBF"])
    def test_direct_kernel_solutions(self, kernel_name, X, y):
        """Tests different kernels"""
        self._verify_solution("direct", kernel_name, X, y)

    @pytest.mark.parametrize("solver_name", ["direct"])
    @pytest.mark.parametrize("data_dense", ["dual"], indirect=["data_dense"])
    def test_direct_solver_solutions(self, solver_name, data_dense):
        """Tests direct solver solutions using dense dual data"""
        X_t, y_t = data_dense
        self._verify_solution(solver_name, "RBF", X_t, y_t)

    @pytest.mark.parametrize("solver_name", ["direct"])
    @pytest.mark.parametrize("data_sparse", ["dual_sparse"], indirect=["data_sparse"])
    def test_direct_solver_solutions_sparse(self, solver_name, data_sparse):
        """Tests direct solver solutions using sparse dual data"""
        X_t, y_t = data_sparse
        self._verify_solution(solver_name, "RBF", X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ALL_SOLVERS_EXCEPT_DIRECT)
    @pytest.mark.parametrize("data_dense", ["dual"], indirect=["data_dense"])
    def test_all_sketch_solvers_solutions(self, solver_name, data_dense):
        """Tests sketch solvers solutions using dense dual data"""
        X_t, y_t = data_dense
        self._verify_solution(solver_name, "RBF", X_t, y_t)

    @pytest.mark.parametrize("solver_name", ALL_SOLVERS_EXCEPT_DIRECT)
    @pytest.mark.parametrize("data_sparse", ["dual_sparse"], indirect=["data_sparse"])
    def test_all_sketch_solvers_solutions_sparse(self, solver_name, data_sparse):
        """Tests sketch solvers solutions using sparse dual data"""
        X_t, y_t = data_sparse
        self._verify_solution(solver_name, "RBF", X_t, y_t)

    def _verify_solution(self, solver_name, kernel_name, X_input, y_input):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        model = KernelRidgeSketch(
            solver=solver_name, tol=1e-3, max_iter=5000, kernel=kernel_name,
        )
        model.alpha = 1.0
        model.fit(X_input, y_input)
        # FIXME: this condition is wrong for sklearn solvers
        # maybe it is like 'norm(residual) < tol * norm(y)'
        # need to look at scipy & sklearn codes closely
        assert model.residual_norms[-1] < model.tol

    def test_predict(self, X, y):
        """Tests prediction pre and post training"""
        model = KernelRidgeSketch()
        n_samples, n_features = X.shape

        # prediction without training should raise error
        with pytest.raises(ValueError):
            model.predict(X)

        model.fit(X, y)
        # tests prediction works after coefficients are set
        predicted = model.predict(X)
        assert predicted.shape == (n_samples, 1)

    def test_predict_accuracy_kernel(self, X, y):
        """
        Tests prediction with kernel over training set
        which should match labels

        For zero regularizer alpha=0, kernel ridge can fit anything.
        """
        model = KernelRidgeSketch(alpha=0, tol=1e-8)
        model.fit(X, y)
        # tests prediction works after coefficients are set
        predicted = model.predict(X)
        diff = predicted - y
        error_pred = np.dot(diff.T, diff) / np.dot(y.T, y)
        assert error_pred < 1e-6
