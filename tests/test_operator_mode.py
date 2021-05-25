"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import pytest
import numpy as np
from numpy.testing import assert_allclose

from ridge_sketch import RidgeSketch
from tests.conftest import ALGO_MODES, SKETCH_SOLVERS


@pytest.mark.operatormode
class TestOperatorMode:
    @pytest.mark.parametrize("solver_name", ["direct", "cg"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_operator_mode_direct_cg(self, solver_name, fit_intercept, data_sparse):
        """Tests operator mode for direct and CG solvers"""
        X_t, y_t = data_sparse
        self._verify_solution(solver_name, fit_intercept, "auto", X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", SKETCH_SOLVERS)
    @pytest.mark.parametrize("fit_intercept", [False, True])
    @pytest.mark.parametrize("algo_mode", ALGO_MODES)
    def test_operator_mode(self, solver_name, fit_intercept, algo_mode, data_sparse):
        """Tests operator mode for all algo modes"""
        X_t, y_t = data_sparse
        self._verify_solution(solver_name, fit_intercept, algo_mode, X_t, y_t)

    def _verify_solution(self, solver_name, fit_intercept, algo_mode, X_input, y_input):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        model = RidgeSketch(
            alpha=0.1,
            solver=solver_name,
            fit_intercept=fit_intercept,
            tol=1e-3,
            max_iter=50000,
            operator_mode=True,
            algo_mode=algo_mode,
        )
        model.fit(X_input, y_input)
        assert model.residual_norms[-1] < model.tol

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ["auto", "subsample", "coordinate descent"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_operator_mode_fit_intercept_coef(self, solver_name, fit_intercept):
        """
        Test if coef_ and intercept attributes are correctly fitted with
        operator mode
        """
        n_samples = 100000
        n_features = 4
        X = np.random.rand(n_samples, n_features)
        w_truth = np.array([-20, -6, 3, 14])
        y = np.matmul(X, w_truth)
        if fit_intercept:
            intercept_truth = 36
            y += intercept_truth
        else:
            intercept_truth = 0.0
        y = y.reshape(-1, 1)

        model = RidgeSketch(
            fit_intercept=fit_intercept,
            solver=solver_name,
            tol=1e-16,
            operator_mode=True,
        )
        model.fit(X, y)

        assert pytest.approx(model.intercept_, 0.1) == intercept_truth
        assert_allclose(model.coef_.flatten(), w_truth, atol=0.1)
