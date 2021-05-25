"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from scipy import linalg
from scipy import sparse
from sklearn import linear_model
from sklearn.utils.extmath import safe_sparse_dot
from warnings import warn

import sketching
from conjugate_gradients import conjugate_grad
from a_matrix import AMatrix, AOperator


class RidgeSketch:
    """
    Ridge Regression class for implementing Ridge Sketch.

    Args:
        alpha (float)): regularization strength.
            Must be greater than 0. Default=1.0
        fit_intercept (bool): whether to use an intercept in model.
            Default = True
        tol (float): error tolerance for solution.
            The solver stops when the relative residual norm becomes smaller
            than the tolerance.
            Default = 1e-3
        sketch_size (int): defaults to min(sqrt(m), sqrt(n))
        solver (str): defaults to 'auto'.
            If dimensions small, uses direct solver.
            Otherwise, randomized solver.
            Options are "direct" or "ridgesketch" or "conjugate gradients".
        random_state (int): seed to use for pseudo random number generation.
        lambda (float): regularization parameter
        max_iter (int): maximum number of iterations
        operator_mode (bool): default to True. Defines the covariance matrix
            as an operator (true) or as numpy matrix (false)
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
        mu (float): defaults to zero. The first acceleration parameter.
        nu (float): defaults to zero. The first acceleration parameter.
    Attributes:
        coef_ (np.array): coefficients containing weight vectors
        intercept_ (float): 0.0, if fit_intercept is False
    """

    SKLEARN_SOLVERS = {
        "svd",
        "cholesky",
        "lsqr",
        "sag",
        "saga",
    }

    SKETCH_SOLVERS = {
        "direct",
        "cg",
        "subsample",
        "coordinate descent",
        "gaussian",
        "count",
        "subcount",
        "hadamard",
    }

    def __init__(
        self,
        alpha=1.0,
        fit_intercept=True,  # default is True in sklearn
        tol=1e-3,
        max_iter=2000,  # default is None in sklearn
        sketch_size=None,
        solver="auto",
        random_state=0,
        operator_mode=False,
        algo_mode="auto",  # "auto", "accel", "mom"
        step_size=None,
        mom_beta=None,
        mom_eta=None,  # used to update step size and beta
        use_heuristic=False,  # use heuristic for increasing momentum
        accel_mu=0.0,
        accel_nu=0.0,
        verbose=0,
    ):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.tol = tol
        self.max_iter = max_iter
        self.sketch_size = sketch_size
        self.solver = self.set_solver(algo_mode, solver)
        self.random_state = random_state
        np.random.seed(seed=random_state)
        self.operator_mode = operator_mode
        self.algo_mode = algo_mode
        self.step_size = self.set_step_size(step_size, mom_beta)

        self.mom_beta, self.mom_eta, self.use_heuristic = self.set_momentum(
            mom_beta, mom_eta, use_heuristic
        )
        self.mu = accel_mu
        self.nu = accel_nu
        self.verbose = verbose

        self.coef_ = None
        self.intercept_ = 0.0
        self.residual_norms = []
        self.iterations = None

    def set_solver(self, algo_mode, solver):
        if algo_mode == "auto":
            if solver == "auto":
                solver = "cg"
        else:
            if solver in ["direct", "cg"]:
                raise ValueError(
                    f"Momentum or acceleration are not available for {solver} solver."
                )
            elif solver == "auto":
                solver = "subsample"  # default for accel and mom versions
        return solver

    def set_step_size(self, step_size, mom_beta):
        """Setting step size parameter depending on the algo mode"""
        if step_size is None:
            if self.algo_mode in ["auto", "accel"]:
                step_size = 1.0
            else:
                # momentum mode
                if isinstance(mom_beta, float):
                    # constant momentum
                    warn("Constant momentum automatically set with 'step_size' = .1 .")
                    step_size = 0.1
                    # step_size is kept as None if increasing momentum
        else:
            # Prevent out of range values
            if step_size < 0.0:
                raise ValueError("'step_size' must be a positive number.")

        return step_size

    def set_momentum(self, mom_beta, mom_eta, use_heuristic):
        """Setting momentum parameters depending on the inputs"""
        if mom_eta is None:
            use_heuristic = False
            # Default settings
            if mom_beta is None:
                if self.algo_mode == "mom":
                    warn("Increasing momentum automatically set with 'mom_eta' = .99 .")
                mom_eta = 0.99
                # step_size = None here
            else:
                # Prevent out of range values
                if mom_beta < 0.0 or mom_beta > 1.0:
                    raise ValueError("'mom_beta' must be in [0, 1].")
                # step_size = .1 here
        else:
            # Prevent out of range values
            if mom_eta <= 0.0 or mom_eta >= 1.0:
                raise ValueError("'mom_eta' must be in (0, 1).")

            # Prevent settings conflicts
            if any(isinstance(x, float) for x in [self.step_size, mom_beta]):
                raise ValueError(
                    "Cannot set at the same time 'step_size', 'mom_beta' and 'mom_eta'."
                )
            self.step_size = None

        return mom_beta, mom_eta, use_heuristic

    def set_sketch_size(self, X):
        """Sets sketching dimension based on data"""
        m = min(X.shape)
        if self.sketch_size is None:
            if self.solver == "coordinate descent":
                self.sketch_size = 1
            else:
                self.sketch_size = int(np.sqrt(m))
        # Sketch size must be an int
        elif not isinstance(self.sketch_size, int):
            raise TypeError(
                "sketch_size should be an int between 1 and m (smallest dimension of X)"
            )
        # Sketch size cannot be greater than m
        elif self.sketch_size > m:
            raise ValueError(
                "sketch_size must be smaller or equal than m (smallest dimension of X)"
            )

    def predict(self, X):
        """Generates prediction based on inputs"""
        if self.coef_ is None:
            raise ValueError("Train model first")

        predicted = np.matmul(X, self.coef_) + self.intercept_
        return predicted

    def fit(self, X, y):
        """
        Fit ridge regression model based on solver and data

        Sets coef_ and intercept_ attributes by solving

        min ||y - Xw||^2 + alpha * ||w||^2

        Args:
            X (2D numpy.ndarray): data matrix (m, m)
            y (2D numpy.ndarray): labels (m, 1)
        """

        if any([isinstance(z, (int, np.integer)) for z in y.flatten()]):
            y = y.astype(np.float64)
            warn(
                "Vector of labels must be an array of floats, it has been automatically converted."
            )
        if y.ndim == 1 or y.shape[0] == 1:
            y = y.reshape(-1, 1)
            warn(
                f"Vector of labels must be a numpy column 2D array, it has been reshaped into a {y.shape} array."
            )

        if self.solver in RidgeSketch.SKLEARN_SOLVERS:
            # # Pass attributes of the RigdeSketch object to the sklearn model
            model = linear_model.Ridge(
                alpha=self.alpha,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                solver=self.solver,
            )
            model.fit(X, y)
            self.coef_ = model.coef_.reshape(-1, 1)

            A = AMatrix(alpha=self.alpha, X=X)
            n_samples, n_features = X.shape
            if n_features > n_samples:
                # dual system
                b = y
                w, _, _, _ = linalg.lstsq(X.T, self.coef_)  # dual coef
            else:
                # primal system
                b = safe_sparse_dot(X.T, y, dense_output=True)
                w = self.coef_
            residual_norm = np.linalg.norm(A @ w - b) / np.linalg.norm(b)
            self.residual_norms.append(residual_norm)
            # self.residual_norms.append(np.linalg.norm(np.matmul(X, model.coef_.T) - y)) # should be relative
            self.intercept_ = model.intercept_
        elif self.solver in RidgeSketch.SKETCH_SOLVERS:
            # if m <= self.direct_solver_th and not self.solver == "direct":
            #     # If dimension of A is small, use direct solver instead of sketching method
            #     warn(
            #         f"Using direct solver since dimension of the problem is smaller than {self.direct_solver_th:d}"
            #     )
            #     self.solver = "direct"
            if self.fit_intercept:
                X = RidgeSketch.add_col_for_intercept(X)
            self.sketch_solver_setup(X, y)
        else:
            raise NotImplementedError(f"{self.solver} solver not implemented")

    @staticmethod
    def add_col_for_intercept(X):
        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1))
        if sparse.issparse(X):
            # if X is a scipy array
            X = sparse.hstack([X, ones], format=X.format)
        else:
            # if X is a numpy array
            X = np.append(X, ones, axis=1)
        return X

    def sketch_solver_setup(self, X, y):
        """
        Chooses between primal or dual formulation
        Solves resulting linear system Ax=b using a sketch-and-project method
        """
        # Instantiate the covariance matrix of ridge regression
        if self.operator_mode:
            A = AOperator(alpha=self.alpha, X=X)
        else:
            A = AMatrix(alpha=self.alpha, X=X)

        # Set the sketch_size to default or check its value
        self.set_sketch_size(X)

        n_samples, n_features = X.shape
        if n_features > n_samples:
            # dual system, solves (X X.T + alpha I) z = b, then w = X.T z
            b = y
            dualcoef_ = self.sketch_solver(A, b)
            coef_ = safe_sparse_dot(X.T, dualcoef_, dense_output=True)
        else:
            # primal system, solves (X.T X + alpha I) w = X.T b
            b = safe_sparse_dot(X.T, y, dense_output=True)
            coef_ = self.sketch_solver(A, b)

        self.coef_ = coef_
        if self.fit_intercept:
            self.intercept_ = coef_[-1, :][0]
            self.coef_ = coef_[:-1, :]
        else:
            self.intercept_ = 0.0

    def solve_system(self, A, b):
        """Solves linear system Ax=b, where A is a sparse or dense matrix"""
        if sparse.issparse(A):
            B = A.todense()
        else:
            B = A

        try:
            # When B is invertible, computing the only solution
            w = linalg.solve(B, b, sym_pos=True, overwrite_a=False)
        except np.linalg.LinAlgError:
            # When B is singular, computing least-squares solution
            w, res, rnk, s = linalg.lstsq(B, b, overwrite_a=False)
        return w

    def direct_solver(self, A, b):
        """Solves linear system and computes residual"""
        w = self.solve_system(A, b)
        residual_norm = np.linalg.norm(A @ w - b) / np.linalg.norm(b)
        self.residual_norms.append(residual_norm)
        return w

    def sketch_solver(self, A, b):
        """Solver using a subsample sketching matrix"""
        if self.solver == "direct":
            return self.direct_solver(A.get_matrix(), b)
        elif self.solver == "cg":
            return self.cg_solver(A, b)

        m = A.shape[0]

        # Initialize sketching object
        sketch_class_name = self.solver.title().replace(" ", "") + "Sketch"
        SketchClass = getattr(sketching, sketch_class_name)
        sketch_method = SketchClass(A, b, self.sketch_size)

        # Initialize weights to 0
        w = np.zeros((m, 1))

        # Run the iterations of sketch project method until relative residual norm below tol
        if self.algo_mode == "accel":  # Accelerated option
            w = self.iterations_accel_sketch_proj(b, w, sketch_method, m)
        elif self.algo_mode == "mom":  # Momentum option
            w = self.iterations_mom_sketch_proj(b, w, sketch_method, m)
        elif self.algo_mode == "auto":  # Non-accelerated option, no momentum
            w = self.iterations_sketch_proj(b, w, sketch_method, m)
        else:
            raise ValueError(
                "Unsupported algo_mode. It should be 'auto', 'accel' or 'mom'."
            )

        if self.verbose:
            current_residual = self.residual_norms[-1]
            if self.iterations != (self.max_iter - 1):
                print(
                    f"Tolerance ({self.tol:.2e}) in {self.iterations:d}"
                    f" iterations, relative residual norm = {current_residual:.2e}"
                )
            else:
                print(
                    f"Max iteration ({self.max_iter:d}) reached, "
                    f"relative residual norm = {current_residual:.2e}"
                )
        return w

    def iterations_sketch_proj(self, b, w, sketch_method, m):
        # Initialize residual to A @ w_0 - b
        r = -b.copy()
        r_norm_initial = np.linalg.norm(r)
        r_norm = r_norm_initial
        self.residual_norms.append(1.0)

        if self.verbose:
            print()
        for i in range(self.max_iter):
            if self.verbose and (i % 50 == 0):
                print(f"iter:{i:^8d} |  rel res norm: {(r_norm / r_norm_initial):.2e}")

            SA, SAS, rs = sketch_method.sketch(r)
            lmbda = self.solve_system(SAS, rs)
            sketch_method.update_iterate(w, lmbda)
            r -= safe_sparse_dot(SA.T, lmbda)
            r_norm = np.linalg.norm(r)
            err = r_norm / r_norm_initial
            self.residual_norms.append(err)
            if err < self.tol:
                break
        self.iterations = i
        return w

    def iterations_mom_sketch_proj(self, b, w, sketch_method, m):
        # Initialize residual to A @ w_0 - b
        w_previous = w.copy()
        r = -b.copy()
        r_previous = r.copy()
        r_norm_initial = np.linalg.norm(r)
        r_norm = r_norm_initial
        self.residual_norms.append(1.0)

        if self.verbose:
            print()

        mom_eta_cst = self.mom_eta
        mom_eta_k = mom_eta_cst
        mom_lmbda_k = 0.0
        switch_flag = False  # for sequence of etas
        for i in range(self.max_iter):
            if self.verbose and (i % 1000 == 0):
                print(f"iter:{i:^8d} |  rel res norm: {(r_norm / r_norm_initial):.2e}")

            if self.mom_eta is not None:
                # if self.eta set by user,
                # update momentum parameters using our settings

                if self.use_heuristic:
                    # # HEURISTIC: unitary step size and capped beta to 1/2
                    # self.step_size = 1.
                    # self.mom_beta = min(1 - ((2 - self.mom_eta)/mom_zeta),.5)

                    # HEURISTIC: sequence of eta and unitary step size
                    if self.step_size is not None and self.mom_beta is not None:
                        if self.mom_beta >= 0.5 and not switch_flag:
                            switch_flag = True
                            mom_eta_cst = 1.0

                    # Update eta to its current constant value
                    mom_eta_k_plus_1 = mom_eta_cst

                    # Update lmbda
                    mom_lmbda_k_plus_1 = (
                        mom_eta_k * (1.0 + mom_lmbda_k - mom_eta_k) / mom_eta_k_plus_1
                    )

                    # Update step size and beta
                    self.step_size = 1.0  # enforce unitary step size
                    self.mom_beta = mom_lmbda_k / (mom_lmbda_k_plus_1 + 1.0)

                    mom_eta_k = mom_eta_k_plus_1
                    mom_lmbda_k = mom_lmbda_k_plus_1
                else:
                    # THEORY: constant eta
                    # mom_zeta_plus_1 = (i + 1) * (1 - self.mom_eta) + 1
                    # self.step_size = self.mom_eta / mom_zeta_plus_1
                    # self.mom_beta = 1 - ((2 - self.mom_eta) / mom_zeta_plus_1)

                    # THEORY: sequence of etas (constant value < 1, then 1)
                    if self.step_size is not None and self.mom_beta is not None:
                        if self.step_size <= 0.5 and not switch_flag:
                            switch_flag = True
                            mom_eta_cst = 1.0

                    # Update eta to its current constant value
                    mom_eta_k_plus_1 = mom_eta_cst

                    # Update lmbda
                    mom_lmbda_k_plus_1 = (
                        mom_eta_k * (1.0 + mom_lmbda_k - mom_eta_k) / mom_eta_k_plus_1
                    )

                    # Update step size and beta
                    self.step_size = mom_eta_k / (mom_lmbda_k_plus_1 + 1.0)
                    self.mom_beta = mom_lmbda_k / (mom_lmbda_k_plus_1 + 1.0)

                    mom_eta_k = mom_eta_k_plus_1
                    mom_lmbda_k = mom_lmbda_k_plus_1

            SA, SAS, rs = sketch_method.sketch(r)
            lmbda = self.solve_system(SAS, rs)
            # updating the iterates
            diff_iterates = w - w_previous
            w_previous = w.copy()  # update w^{k-1} <- w^k
            sketch_method.update_iterate(w, lmbda, step_size=self.step_size)
            w += self.mom_beta * diff_iterates  # update w^{k+1} <- w^k
            # updating the residuals
            r_tmp = r.copy()
            r *= 1.0 + self.mom_beta
            r -= self.mom_beta * r_previous
            r -= self.step_size * safe_sparse_dot(SA.T, lmbda)
            r_previous = r_tmp.copy()
            r_norm = np.linalg.norm(r)
            err = r_norm / r_norm_initial
            self.residual_norms.append(err)
            if err < self.tol:
                break
        self.iterations = i
        return w

    def iterations_accel_sketch_proj(self, b, w, sketch_method, m):
        if self.mu == 0 or self.nu == 0:
            # srt = self.sketch_size / m
            # mu, nu = (srt  *0.25 , 4.0 / srt)
            #  mu, nu = (srt ** (4 / 5), 1.0 / (srt ** (5 / 4)))
            mu, nu = 0.1, 5.0
        elif self.mu == "theorydense" or self.nu == "theorydense":
            diags = sketch_method.A.get_diag()
            eigs = np.linalg.eigvalsh(sketch_method.A._A)
            lmin = np.min(eigs)
            trA = np.sum(diags)
            minAii = np.min(diags)
            mu = lmin / trA  # (self.alpha+minAii)/(trA+minAii)
            nu = trA / minAii
        else:
            mu = self.mu
            nu = self.nu

        # mu, nu = self.set_aggressive_accel_parameters(A)
        gamma = np.sqrt(1 / (mu * nu))  # step size
        a = 1 / (1 + gamma * nu)  # momentum parameter alpha
        beta = 1 - np.sqrt(mu / nu)  # momentum parameter

        v = w.copy()  # averaged estimate
        z = w.copy()  # extrapolated estimate
        rw = -b.copy()  # initialize residual when w=0, which is -b
        rz = -b.copy()  # residual of z
        rv = -b.copy()  # residual of v

        self.residual_norms.append(1.0)
        r_norm_initial = np.linalg.norm(rw)
        rw_norm = r_norm_initial
        for i in range(self.max_iter):
            if self.verbose and (i % 50 == 0):
                print(f"iter:{i:^8d} |  rel res norm: {(rw_norm / r_norm_initial):.2e}")

            SA, SAS, rs = sketch_method.sketch(rz)  # Sketch A, AS and rz
            lmbda = self.solve_system(SAS, rs)
            z = a * v + (1 - a) * w  # 1st momentum step
            w = z.copy()  # Every other workaround failed :( I can explain
            sketch_method.update_iterate(w, lmbda)  # update w (SGD step)
            v = beta * v + (1 - beta) * z + gamma * (w - z)  # 2nd mom step
            rw = rz - safe_sparse_dot(SA.T, lmbda)  # updating the residuals
            rv = (
                beta * rv + (1 - beta) * rz + gamma * (rw - rz)
            )  # updating the residuals
            rz = a * rv + (1 - a) * rw  # updating the residuals
            # COMMENT. Both z and w converge.
            # Choosing the one with minimum residual:
            rz_norm = np.linalg.norm(rz)
            rw_norm = np.linalg.norm(rw)
            # err = np.minimum(rw_norm,  rz_norm)/r_norm_initial
            err = rw_norm / r_norm_initial
            self.residual_norms.append(err)
            if err < self.tol:
                break
        self.iterations = i
        if rz_norm < rw_norm:
            return z
        else:
            return w

    def cg_solver(self, A, b):
        w, self.residual_norms, self.iterations = conjugate_grad(
            A, b, self.tol, self.max_iter, verbose=self.verbose
        )

        if self.verbose:
            current_residual = self.residual_norms[-1]
            if self.iterations != (self.max_iter - 1):
                print(
                    f"Tolerance ({self.tol:.2e}) in {self.iterations:d}"
                    f" iterations, relative residual norm = {current_residual:.2e}"
                )
            else:
                print(
                    f"Max iteration ({self.max_iter:d}) reached, "
                    f"relative residual norm = {current_residual:.2e}"
                )

        return w
