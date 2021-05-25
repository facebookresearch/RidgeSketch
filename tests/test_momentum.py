"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from numpy.testing import assert_allclose
import pytest

from ridge_sketch import RidgeSketch
from kernel_ridge_sketch import KernelRidgeSketch
from tests.conftest import SKETCH_SOLVERS


@pytest.mark.ridgesolver
class TestRidgeSketchMomentum:
    def test_instantiation(self):
        """Confirms RidgeSolver with momentum can be instantiated"""
        _ = RidgeSketch(algo_mode="mom", mom_beta=0.99, step_size=0.5)

    def test_attributes(self):
        """Confirms class contains expected attributes"""
        model = RidgeSketch(algo_mode="mom", step_size=0.5, mom_beta=0.999)
        assert model.step_size == 0.5
        assert model.mom_beta == 0.999
        assert model.mom_eta is None
        assert model.solver == "subsample"

        model = RidgeSketch(algo_mode="mom", step_size=0.5)
        assert model.step_size == 0.5
        assert model.mom_beta is None
        assert model.mom_eta == 0.99

        model = RidgeSketch(algo_mode="mom", mom_beta=0.999)
        assert model.step_size == 0.1
        assert model.mom_beta == 0.999
        assert model.mom_eta is None

        model = RidgeSketch(algo_mode="mom", mom_eta=0.47)
        assert model.step_size is None
        assert model.mom_beta is None
        assert model.mom_eta == 0.47

        model = RidgeSketch(algo_mode="mom")
        assert model.step_size is None
        assert model.mom_beta is None
        assert model.mom_eta == 0.99

        with pytest.raises(ValueError):
            _ = RidgeSketch(algo_mode="mom", mom_beta=-15.0)

        with pytest.raises(ValueError):
            _ = RidgeSketch(algo_mode="mom", mom_eta=2.0)

    def test_init_warnings(self):
        """
        Cheking that warning are correctly raised when a
        momentum parameter is set by default
        """
        with pytest.warns(UserWarning):
            _ = RidgeSketch(algo_mode="mom", step_size=0.5)

        with pytest.warns(UserWarning):
            _ = RidgeSketch(algo_mode="mom", mom_beta=0.999)

    def test_unsupported_solvers(self, X, y):
        with pytest.raises(ValueError):
            _ = RidgeSketch(solver="direct", algo_mode="mom")
        with pytest.raises(ValueError):
            _ = RidgeSketch(solver="cg", algo_mode="mom")

    def test_fit_sets_coef(self, X, y):
        """Confirms fit with default parameters sets coef_ attribute"""
        model = RidgeSketch(algo_mode="mom", mom_eta=0.995, verbose=1)
        model.fit(X, y)
        assert hasattr(model, "coef_")

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    def test_increasing_momentum_solutions_dense(self, solver_name, data_dense):
        """
        Tests all sketch solvers with momentum solutions using dense dual and
        primal
        """
        X_t, y_t = data_dense
        self._verify_solution_increasing_momentum(solver_name, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    def test_increasing_momentum_solutions_sparse(self, solver_name, data_sparse):
        """Tests sketch solvers solutions using sparse primal and dual"""
        X_t, y_t = data_sparse
        self._verify_solution_increasing_momentum(solver_name, X_t, y_t)

    def _verify_solution_increasing_momentum(self, solver_name, X, y):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        if solver_name in ["coordinate descent", "hadamard"]:
            tol = 1e-1
        else:
            tol = 1e-3

        model = RidgeSketch(
            alpha=0.1,
            solver=solver_name,
            fit_intercept=True,
            algo_mode="mom",
            mom_eta=0.995,
            tol=tol,
            max_iter=10000,
        )
        model.fit(X, y)
        assert model.residual_norms[-1] < model.tol

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    def test_constant_momentum_solutions_dense(self, solver_name, data_dense):
        """
        Tests all sketch solvers with momentum solutions using dense dual and
        primal
        """
        X_t, y_t = data_dense
        self._verify_solution_constant_momentum(solver_name, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    def test_constant_momentum_solutions_sparse(self, solver_name, data_sparse):
        """Tests sketch solvers solutions using sparse primal and dual"""
        X_t, y_t = data_sparse
        self._verify_solution_constant_momentum(solver_name, X_t, y_t)

    def _verify_solution_constant_momentum(self, solver_name, X, y):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        if solver_name in ["coordinate descent", "hadamard"]:
            tol = 1e-1
        else:
            tol = 1e-3

        model = RidgeSketch(
            alpha=0.1,
            solver=solver_name,
            fit_intercept=True,
            algo_mode="mom",
            step_size=1.0,
            mom_beta=0.5,
            tol=tol,
            max_iter=10000,
        )
        model.fit(X, y)
        assert model.residual_norms[-1] < model.tol

    def test_predict(self, X):
        """Tests prediction pre and post training"""
        model = RidgeSketch(algo_mode="mom", mom_eta=0.995)
        n_samples, n_features = X.shape

        # prediction without training should raise error
        with pytest.raises(ValueError):
            model.predict(X)

        # tests prediction works after coefficients are set
        model.coef_ = np.random.rand(n_features, 1)
        predicted = model.predict(X)
        assert predicted.shape == (n_samples, 1)

    def test_zero_momentum(self, X, y):
        """
        Check that we recover sketch-and-project when momentum has zero beta
        """
        model = RidgeSketch(
            solver="subsample", algo_mode="auto", max_iter=5, random_state=0,
        )
        model.fit(X, y)

        mom_model = RidgeSketch(
            solver="subsample",
            algo_mode="mom",
            step_size=1.0,
            mom_beta=0.0,
            max_iter=5,
            random_state=0,
        )
        mom_model.fit(X, y)

        assert_allclose(model.coef_, mom_model.coef_, rtol=1e-6)

    def test_increasing_momentum_theory(self, X, y):
        model = RidgeSketch(
            algo_mode="mom", tol=1e-16, mom_eta=0.99, use_heuristic=False,
        )
        assert model.step_size is None
        assert model.mom_beta is None
        model.fit(X, y)
        assert model.step_size >= 0.45
        assert model.mom_beta <= 0.55
        print(model.step_size, model.mom_beta)

    def test_increasing_momentum_heuristic(self, X, y):
        model = RidgeSketch(
            algo_mode="mom", tol=1e-16, sketch_size=1, mom_eta=0.99, use_heuristic=True,
        )
        assert model.step_size is None
        assert model.mom_beta is None
        model.fit(X, y)
        assert model.step_size == 1.0
        assert model.mom_beta <= 0.55
        print(model.step_size, model.mom_beta)


class TestKernelRidgeSketchMomentum:
    def test_instantiation(self):
        """Confirms RidgeSolver with momentum can be instantiated"""
        _ = KernelRidgeSketch(algo_mode="mom", mom_beta=0.99, step_size=0.5)

    def test_attributes(self):
        """Confirms class contains expected attributes"""
        model = KernelRidgeSketch(algo_mode="mom", step_size=0.5, mom_beta=0.999,)
        assert model.step_size == 0.5
        assert model.mom_beta == 0.999
        assert model.mom_eta is None
        assert model.solver == "subsample"

        model = KernelRidgeSketch(algo_mode="mom", step_size=0.5)
        assert model.step_size == 0.5
        assert model.mom_beta is None
        assert model.mom_eta == 0.99

        model = KernelRidgeSketch(algo_mode="mom", mom_beta=0.999)
        assert model.step_size == 0.1
        assert model.mom_beta == 0.999
        assert model.mom_eta is None

        model = KernelRidgeSketch(algo_mode="mom", mom_eta=0.47)
        assert model.step_size is None
        assert model.mom_beta is None
        assert model.mom_eta == 0.47

        model = KernelRidgeSketch(algo_mode="mom")
        assert model.step_size is None
        assert model.mom_beta is None
        assert model.mom_eta == 0.99

    def test_fit_sets_coef(self, X, y):
        """Confirms fit with default parameters sets coef_ attribute"""
        model = KernelRidgeSketch(algo_mode="mom", mom_eta=0.995)
        model.fit(X, y)
        assert hasattr(model, "coef_")

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("data_dense", ["dual"], indirect=["data_dense"])
    def test_increasing_momentum_solutions_dense(self, solver_name, data_dense):
        """Tests all sketch solvers with momentum solutions using dense dual"""
        X_t, y_t = data_dense
        self._verify_solution_increasing_momentum(solver_name, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("data_sparse", ["dual_sparse"], indirect=["data_sparse"])
    def test_increasing_momentum_solutions_sparse(self, solver_name, data_sparse):
        """Tests sketch solvers solutions using sparse dual"""
        X_t, y_t = data_sparse
        self._verify_solution_increasing_momentum(solver_name, X_t, y_t)

    def _verify_solution_increasing_momentum(self, solver_name, X, y):
        """Verifies residual is less than the expected tolerance"""
        # KernelRidgeSketch's solution for the given solver
        if solver_name in ["coordinate descent", "hadamard"]:
            tol = 1e-1
        else:
            tol = 1e-3

        model = KernelRidgeSketch(
            alpha=0.1,
            solver=solver_name,
            algo_mode="mom",
            mom_eta=0.99,
            tol=tol,
            max_iter=500000,
        )
        model.fit(X, y)
        assert model.residual_norms[-1] < model.tol

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("data_dense", ["dual"], indirect=["data_dense"])
    def test_constant_momentum_solutions_dense(self, solver_name, data_dense):
        """Tests all sketch solvers with momentum solutions using dense dual"""
        X_t, y_t = data_dense
        self._verify_solution_constant_momentum(solver_name, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("data_sparse", ["dual_sparse"], indirect=["data_sparse"])
    def test_constant_momentum_solutions_sparse(self, solver_name, data_sparse):
        """Tests sketch solvers solutions using sparse dual"""
        X_t, y_t = data_sparse
        self._verify_solution_constant_momentum(solver_name, X_t, y_t)

    def _verify_solution_constant_momentum(self, solver_name, X, y):
        """Verifies residual is less than the expected tolerance"""
        # KernelRidgeSketch's solution for the given solver
        if solver_name in ["coordinate descent", "hadamard"]:
            # coordinate descent too slow on these problems
            tol = 0.9
        else:
            tol = 1e-3

        model = KernelRidgeSketch(
            alpha=0.1,
            solver=solver_name,
            algo_mode="mom",
            step_size=1.0,
            mom_beta=0.5,
            tol=tol,
            max_iter=30000,
        )
        model.fit(X, y)
        assert model.residual_norms[-1] < model.tol

    # def test_predict(self, X):
    #     """Tests prediction pre and post training"""
    #     model = RidgeSketch(algo_mode="mom", mom_eta=.995)
    #     n_samples, n_features = X.shape

    #     # prediction without training should raise error
    #     with pytest.raises(ValueError):
    #         model.predict(X)

    #     # tests prediction works after coefficients are set
    #     model.coef_ = np.random.rand(n_features, 1)
    #     predicted = model.predict(X)
    #     assert predicted.shape == (n_samples, 1)

    # def test_zero_momentum(self, X, y):
    #     """
    #     Check that we recover sketch-and-project when momentum has zero beta
    #     """
    #     model = RidgeSketch(
    #         solver="subsample", algo_mode="auto", max_iter=5, random_state=0,
    #     )
    #     model.fit(X, y)

    #     mom_model = RidgeSketch(
    #         solver="subsample",
    #         algo_mode="mom",
    #         step_size=1,
    #         mom_beta=0.,
    #         max_iter=5,
    #         random_state=0,
    #     )
    #     mom_model.fit(X, y)

    #     assert_allclose(model.coef_, mom_model.coef_, rtol=1e-6)
