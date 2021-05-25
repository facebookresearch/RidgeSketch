"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import pytest
import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import random as sprandom

from ridge_sketch import RidgeSketch
from tests.conftest import ALL_SOLVERS_EXCEPT_DIRECT


@pytest.mark.ridgesolver
class TestRidgeSketch:
    def test_instantiation(self):
        """Confirms RidgeSolver can be instantiated"""
        _ = RidgeSketch()

    def test_attributes(self):
        """Confirms class contains expected attributes"""
        model = RidgeSketch()
        assert model.alpha == 1.0
        # check solver is set to sparse_cg by default
        assert model.solver == "cg"

    def test_set_step_size(self):
        """Confirms step_size is correctly set"""
        # auto and accel step size initialization
        model = RidgeSketch(algo_mode="auto")
        assert model.step_size == 1.0
        model = RidgeSketch(algo_mode="auto", step_size=0.7)
        assert model.step_size == 0.7

        model = RidgeSketch(algo_mode="accel")
        assert model.step_size == 1.0
        model = RidgeSketch(algo_mode="accel", step_size=0.7)
        assert model.step_size == 0.7

        with pytest.raises(ValueError):
            model = RidgeSketch(step_size=-1.0)

        # mom step size initialization
        model = RidgeSketch(algo_mode="mom")
        assert model.step_size is None

        model = RidgeSketch(algo_mode="mom", mom_beta=0.5)
        assert model.step_size == 0.1
        model = RidgeSketch(algo_mode="mom", mom_beta=0.5, step_size=0.2)
        assert model.step_size == 0.2

        model = RidgeSketch(algo_mode="mom", mom_eta=0.99)
        assert model.step_size is None

        with pytest.raises(ValueError):
            model = RidgeSketch(algo_mode="mom", mom_eta=0.99, step_size=0.5)

    def test_set_sketch_size(self, X):
        """Confirms sketch_size is correctly set"""
        # check that at initialization it is None
        model = RidgeSketch()
        assert model.sketch_size is None

        # check that default set value is square root of m
        m = min(X.shape)
        model.set_sketch_size(X)
        assert model.sketch_size == int(np.sqrt(m))

        # check that value given by user is correctly passed
        model = RidgeSketch(sketch_size=7)
        assert model.sketch_size == 7
        model.set_sketch_size(X)
        assert model.sketch_size == 7

        # set to value larger than m should raise an error
        model = RidgeSketch(sketch_size=23.2)
        with pytest.raises(TypeError):
            model.set_sketch_size(X)
        # set to value greater than m should raise an error
        model = RidgeSketch(sketch_size=1000000000)
        with pytest.raises(ValueError):
            model.set_sketch_size(X)

    def test_fit_y_type_shape(self, X, y):
        """Confirms fit works whatever the and type of shape of y"""
        model = RidgeSketch()

        # 1D
        y = y.flatten()
        model.fit(X, y)

        # 2D
        y = y.reshape(-1, 1)
        model.fit(X, y)
        y = y.reshape(1, -1)
        model.fit(X, y)

        # int
        y = y.astype(int)
        model.fit(X, y)

    def test_solver_instantiation(self, X, y):
        """Test instantiation for direct, sketch and sklearn solvers"""
        model = RidgeSketch(verbose=1)
        model.fit(X, y)

        model = RidgeSketch(solver="direct")
        model.fit(X, y)

        model = RidgeSketch(solver="svd")
        model.fit(X, y)

        wrong_model = RidgeSketch(solver="wrong_solver")
        with pytest.raises(NotImplementedError):
            wrong_model.fit(X, y)

    def test_algo_mode_instantiation(self, X, y):
        model = RidgeSketch(algo_mode="wrong_mode")
        with pytest.raises(ValueError):
            model.fit(X, y)

    def test_fit_sets_coef(self, X, y):
        """Confirms fit with default parameters sets coef_ attribute"""
        model = RidgeSketch()
        model.fit(X, y)
        assert hasattr(model, "coef_")

    def test_max_iter_reached(self, X, y):
        """Check that maximum iteration is reached"""
        max_iter = 5
        model = RidgeSketch(
            tol=1e-16, max_iter=max_iter, solver="subsample", verbose=1,
        )
        model.fit(X, y)
        assert len(model.residual_norms) == max_iter + 1

    def test_fit_solution(self, X, y):
        """Compare iterative coef_ after fit to closed-form solution"""
        fit_intercept = False  # because data are normalized when True

        # cg's solution
        model = RidgeSketch(tol=1e-16, fit_intercept=fit_intercept)
        model.fit(X, y)
        cg_coef = model.coef_

        # closed-form solution via 'linalg.solve'
        ds = RidgeSketch(solver="direct", fit_intercept=fit_intercept)
        ds.fit(X, y)
        ds_coef = ds.coef_

        assert_allclose(cg_coef, ds_coef, rtol=1e-6)

        # closed-form solution via sklearn solver
        sklearn = RidgeSketch(solver="svd", tol=1e-16, fit_intercept=fit_intercept)
        sklearn.fit(X, y)
        sklearn_coef = sklearn.coef_

        assert_allclose(cg_coef, sklearn_coef, rtol=1e-6)

    @pytest.mark.parametrize("solver_name", ["direct"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_direct_solver_solutions(self, solver_name, fit_intercept, data_dense):
        """Tests direct solver solutions using primal and dual dense data"""
        X_t, y_t = data_dense
        self._verify_solution(solver_name, fit_intercept, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ALL_SOLVERS_EXCEPT_DIRECT)
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_all_sketch_solvers_solutions(self, solver_name, fit_intercept, data_dense):
        """Tests sketch solvers solutions using primal and dual dense data"""
        X_t, y_t = data_dense
        self._verify_solution(solver_name, fit_intercept, X_t, y_t)

    @pytest.mark.slow
    @pytest.mark.parametrize("solver_name", ALL_SOLVERS_EXCEPT_DIRECT)
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_all_sketch_solvers_solutions_sparse(
        self, solver_name, fit_intercept, data_sparse
    ):
        """Tests sketch solvers solutions using primal and dual sparse data"""
        X_t, y_t = data_sparse
        self._verify_solution(solver_name, fit_intercept, X_t, y_t)

    def _verify_solution(self, solver_name, fit_intercept, X_input, y_input):
        """Verifies residual is less than the expected tolerance"""
        # RidgeSketch's solution for the given solver
        model = RidgeSketch(
            solver=solver_name,
            fit_intercept=fit_intercept,
            alpha=0.1,
            tol=1e-3,
            max_iter=20000,
        )
        model.fit(X_input, y_input)
        # FIXME: this condition is wrong for sklearn solvers
        # maybe it is like 'norm(residual) < tol * norm(y)'
        # need to look at scipy & sklearn codes closely
        assert model.residual_norms[-1] < model.tol

    @pytest.mark.parametrize("solver_name", ["saga"])
    def test_sklearn_solver_approx_tolerance(self, solver_name, data_dense):
        """Tests direct solver solutions using primal and dual dense data"""
        X_t, y_t = data_dense
        model = RidgeSketch(
            solver=solver_name,
            fit_intercept=False,
            alpha=0.1,
            tol=1e-4,
            max_iter=20000,
        )
        model.fit(X_t, y_t)
        # definition of tolerance differs from sklearn
        # so only approximative tolerance test
        assert model.residual_norms[-1] < model.tol * 10

    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_predict(self, X, y, fit_intercept):
        """
        Tests prediction pre and post training, with and without intercept
        """
        model = RidgeSketch(fit_intercept=fit_intercept)
        n_samples, n_features = X.shape

        # prediction without training should raise error
        with pytest.raises(ValueError):
            model.predict(X)

        # tests prediction works after coefficients are set
        model.coef_ = np.random.rand(n_features, 1)
        predicted = model.predict(X)
        assert predicted.shape == (n_samples, 1)

        model.fit(X, y)
        n_samples_test = 10
        X_test = np.random.rand(n_samples_test, n_features)
        predicted = model.predict(X_test)
        assert predicted.shape == (n_samples_test, 1)

    @pytest.mark.parametrize("solver_name", ["auto", "subsample", "coordinate descent"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_fit_intercept_coef_shape(self, X, y, solver_name, fit_intercept):
        """
        Test if coef_ and intercept attributes have the correct shape and
        default value
        """
        model = RidgeSketch(fit_intercept=fit_intercept, solver=solver_name)
        assert model.intercept_ == 0.0

        model.fit(X, y)

        n_features = X.shape[1]
        assert model.coef_.shape == (n_features, 1)
        assert isinstance(model.intercept_, float)

    @pytest.mark.parametrize("solver_name", ["auto", "subsample", "coordinate descent"])
    @pytest.mark.parametrize("fit_intercept", [False, True])
    def test_fit_intercept_coef(self, solver_name, fit_intercept):
        """Test if coef_ and intercept attributes are correctly fitted"""
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

        model = RidgeSketch(fit_intercept=fit_intercept, solver=solver_name, tol=1e-16)
        model.fit(X, y)

        assert pytest.approx(model.intercept_, 0.1) == intercept_truth
        assert_allclose(model.coef_.flatten(), w_truth, atol=0.1)

    # def test_singular_A(self):
    #     """
    #     Test if sketch-and-project uses linalg.lstsq if SAS is singular
    #     """
    #     n_samples, n_features = 2000, 1000
    #     X_t = sprandom(n_samples, n_features, density=.004, format="csr")
    #     y_t = np.random.rand(n_samples, 1)

    #     m = min(n_samples, n_features)
    #     tol = 1e-6
    #     model = RidgeSketch(
    #         alpha=0.,
    #         solver="subsample",
    #         sketch_size=int(m/2),
    #         tol=tol,
    #         max_iter=100,
    #         verbose=1,
    #     )
    #     model.fit(X_t, y_t)

    #     assert model.residual_norms[-1] <= tol
