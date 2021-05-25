"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from scipy.sparse import isspmatrix as issparse
from sklearn.gaussian_process.kernels import RBF  # for the kernels
from sklearn.gaussian_process.kernels import Matern

from a_matrix import AMatrix
from ridge_sketch import RidgeSketch


class KernelRidgeSketch(RidgeSketch):
    """
    Kernel Ridge Regression class. Inherits from RidgeSketch class

    Args:
    -----
        alpha (float)): regularization strength. Must be greater than 0.
            Default=1.0
        fit_intercept (bool): whether to use an intercept in model.
        tol (float): error tolerance for solution. Default = 1.e-3.
        sketch_size (int): defaults to min(sqrt(m), sqrt(n))
        solver (str): defaults to "auto". If dimensions smaller, use direct
            solver. Otherwise, randomized solver.
            options are "direct" or "ridgesketch" or "conjugate gradients".
        random_state (int): seed to use for pseudo random number generation.
        lambda (float): regularization parameter
        max_iter (int): maximum number of iterations
        algo_mode (str): defaults to 'auto' uses classical Ridge Sketch method.
            If set to 'mom', it uses the momentum version.
            Default is increasing momentum when neither
            'mom_eta','step_size' or 'mom_beta' is set.
            If set to 'accel', it uses the accelerated version.
        step_size (float): step_size parameter for momentum version.
            Default = 1.
        mom_beta (float): momentum parameter between 0 and 1.
            Default = .9.
        mom_eta (float): parameter between 0 and 1 (excluded).
            Default = .99 when "algo_mode" is set to 'mom',
            in this case the algorithm uses the decreasing
            'step_size' and increasing 'mom_beta' attributes.
        use_heuristic (bool): use the heuristic ruling increasing momentum
            If set to True, the code runs the momentum version with unitary
            step size and increasing momentum parameter beta,
            which is capped at 0.5.
            Default = False.
        kernel(string): determines the type of kernel. Default is "RBF" for
            radial basis functions (aka Gaussian kernels). You can
            alternatively choose "Matern" for Matern kernel
        kernel_sigma(float): default to 1.0. The sigma is the bandwidth
            parameter of the kernel.
        kernel_nu(float): default to 0.5. The nu is the parameter that
            determines how smooth and differential is the matern kernel.
            The larger nu is, the more differentiable it is

    Attributes:
    -----------
        coef_ (np.array): coefficients containing weight vectors
        intercept_ (float): 0.0, if fit_intercept is False
    """

    def __init__(
        self,
        alpha=1.0,
        tol=1e-3,
        max_iter=2000,  # default is None in sklearn
        sketch_size=None,
        solver="auto",
        random_state=0,
        algo_mode="auto",  # "auto", "accel", "mom"
        step_size=None,
        mom_beta=None,
        mom_eta=None,  # used to update step size and beta
        use_heuristic=False,  # use heuristic for increasing momentum
        accel_mu=0,
        accel_nu=0,
        verbose=0,
        kernel="RBF",
        kernel_sigma=1.0,
        kernel_nu=0.5,
    ):
        self.kernel = kernel
        self.kernel_sigma = kernel_sigma
        self.X_fit = np.array([])
        RidgeSketch.__init__(
            self,
            alpha=alpha,
            fit_intercept=False,
            tol=tol,
            max_iter=max_iter,
            sketch_size=sketch_size,
            solver=solver,
            random_state=random_state,
            operator_mode=False,
            algo_mode=algo_mode,
            step_size=step_size,
            mom_beta=mom_beta,
            mom_eta=mom_eta,
            use_heuristic=use_heuristic,
            accel_mu=accel_mu,
            accel_nu=accel_nu,
            verbose=verbose,
        )

        if self.kernel == "Matern":
            self.K_class = Matern(
                self.kernel_sigma, nu=kernel_nu
            )  # Should allow user to choose nu?
        else:
            self.K_class = RBF(self.kernel_sigma)

    def predict(self, X):
        """Generates prediction based on inputs"""
        if self.coef_ is None:
            raise ValueError("Train model first")

        if issparse(X):
            # Sparse matrices not supported for kernels
            X = X.toarray()
        K_test = self.K_class(
            X=self.X_fit, Y=X
        )  # Build kernel of train and test data K(X_i,Y_i)

        predicted = np.matmul(K_test, self.coef_)
        return predicted

    def fit(self, X, y):
        """Fit Kernel ridge regression model based on solver and data
        Sets coef_ and intercept_ attributes by solving

        min ||y - Kw||^2 + \lambda * w^T K w

        where K is the kernel built on W.

        Args:
            X (np.array): data matrix
            y (np.array): labels
        """
        if issparse(X):  # Sparse matrices not supported for kernels
            X = X.toarray()
        self.X_fit = X.copy()
        RidgeSketch.fit(self, X, y)

    def sketch_solver_setup(self, X, y):
        """
        Applies the sketch-and-project method to
        the kernel ridge regression problem
        """
        K = self.K_class(X=X)
        A = AMatrix(alpha=self.alpha, X=X, K=K)
        # Set the sketch_size to default or check its value
        self.set_sketch_size(A)

        # solve linear system
        b = K.dot(y)
        coef_ = self.sketch_solver(A, b)

        self.coef_ = coef_
        # self.intercept_ = intercept_
        return
