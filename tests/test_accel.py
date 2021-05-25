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


@pytest.mark.ridgesolver
class TestRidgeSketchAccel:
    def test_attributes(self):
        """
        Confirms class contains expected acceleration parameters
        and checks default setting
        """
        model = RidgeSketch(algo_mode="accel")
        assert model.mu == 0.0
        assert model.nu == 0.0
        assert hasattr(model, "algo_mode")
        assert model.algo_mode == "accel"

    def test_fit_solution(self, X, y):
        """Compare accell to direct solver coef_ """
        # Accelerated Gaussien sketch solution
        model = RidgeSketch(
            tol=1e-13, solver="subsample", verbose=True, alpha=10, algo_mode="accel",
        )
        model.fit(X, y)
        coef_iter = model.coef_
        print(
            f"Tolerance ({model.tol:.2e}) in {model.iterations:d} iterations, "
            f"relative residual norm = {model.residual_norms[-1]:.2e}"
        )
        # closed-form solution via 'linalg.solve'
        ds = RidgeSketch(solver="direct", alpha=10)
        ds.fit(X, y)
        ds_coef = ds.coef_

        assert_allclose(coef_iter, ds_coef, rtol=1e-6)

    def test_residuals_unaccelerated_comparison(self, X, y):
        """
        Compare acceleration with parameter settings
        that should recover the unaccelerated version
        """
        accel_mu = 1
        accel_nu = 1
        # Accelerated Gaussian sketch with parameters
        # that should recover unaccelerated version
        model = RidgeSketch(
            tol=1e-13,
            solver="gaussian",
            verbose=True,
            alpha=10,
            algo_mode="accel",
            accel_mu=accel_mu,
            accel_nu=accel_nu,
        )
        model.fit(X, y)

        print(
            f"Tolerance ({model.tol:.2e}) in {model.iterations:d} iterations, "
            f"relative residual norm = {model.residual_norms[-1]:.2e}"
        )
        # Solution without acceleration
        ds = model = RidgeSketch(
            solver="gaussian", alpha=10, tol=1e-13, algo_mode="auto", verbose=True,
        )
        ds.fit(X, y)

        assert len(model.residual_norms) == len(ds.residual_norms)

        assert_allclose(model.residual_norms, ds.residual_norms, rtol=1e-6)

        difference = []
        zip_object = zip(model.residual_norms, ds.residual_norms)
        for list1_i, list2_i in zip_object:
            difference.append(list1_i - list2_i)

        rtol = 1e-6
        assert sum(difference) < rtol

    @pytest.mark.slow
    # @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("solver_name", ["coordinate descent"])
    def test_all_sketch_solvers_solutions(self, solver_name, data_dense):
        """Tests sketch solvers solutions using primal and dual dense data"""
        X_t, y_t = data_dense
        self._verify_solution(solver_name, X_t, y_t)

    def _verify_solution(self, solver_name, X_input, y_input):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        model = RidgeSketch(
            solver=solver_name, alpha=0.1, tol=1e-3, max_iter=20000, algo_mode="accel",
        )
        model.fit(X_input, y_input)
        # FIXME: this condition is wrong for sklearn solvers
        # maybe it is like 'norm(residual) < tol * norm(y)'
        # need to look at scipy & sklearn codes closely
        assert model.residual_norms[-1] < model.tol

    def test_predict(self, X):
        """Tests prediction pre and post training"""
        model = RidgeSketch(algo_mode="accel")
        n_samples, n_features = X.shape

        # prediction without training should raise error
        with pytest.raises(ValueError):
            model.predict(X)

        # tests prediction works after coefficients are set
        model.coef_ = np.random.rand(n_features, 1)
        predicted = model.predict(X)
        assert predicted.shape == (n_samples, 1)

    def test_theory_dense_parameters(self, X, y):
        """Test convergence with mu and nu given by theory"""
        model = RidgeSketch(
            alpha=10,
            tol=1e-13,
            algo_mode="accel",
            solver="subsample",
            accel_mu="theorydense",
            accel_nu="theorydense",
        )
        model.fit(X, y)
