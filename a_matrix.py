"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from abc import ABC, abstractmethod
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse
from scipy.sparse import isspmatrix_coo


class AAbstract(ABC):
    r"""
    Matrix class for defining the covariance matrix of ridge regression.

    Primal form:
    $$A = (X^\top X + alpha I)$$

    Dual form:
    $$A = (X X^\top + alpha I)$$

    Attributes
    ----------
    X : sparse array or ndarray
        Data matrix
    alpha : float, default=1
        Regularization strength
        Must be greater than 0
    alpha_vec : ndarray
        Regularization strength in a 1D array
    fit_intercept : bool, default=True
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.
    _A : sparse array or ndarray, default=np.array([])
        Full matrix A.
    n_samples : int
        Number of samples, i.e. of rows of X
    n_features : int
        Number of features, i.e. of columns of X
    sparse_matrix : bool 
        Specifies if input X is sparse or not
    shape : tuple of ints
        Shape of the squared matrix A

    Methods
    -------
    get_matrix()
        Returns the full matrix A stored in attribute _A
    build_kernel_system_matrix(K)
        Build the system matrix K(K+\alpha I) of kernel ridge regression
    build_matrix()
        Build the full matrix A and store it in attribute _A
    __mul__(v)
        Returns the matrix-vector product of the input by A: Av
    get_elements(rows_idx, cols_idx)
        Returns the (rows_idx, cols_idx)-th elements of A:
        A[rows_idx, cols_idx]
    get_rows(rows_idx)
        Get ths rows of A indexed by a list of integers rows_idx
    get_diag()
        Get the diagonal of A
    """

    def __init__(
        self, X, alpha=1.0, fit_intercept=True,
    ):
        """
        Parameters
        ----------
        X : sparse array or ndarray
            Data matrix
        alpha : float, default=1.
            Regularization strength.
            Must be greater than 0.
        fit_intercept : bool, default=True
            Specifies if a constant (a.k.a. bias or intercept) should be
            added to the decision function.
        """
        self.alpha = alpha
        self.alpha_vec = np.atleast_1d(self.alpha)
        self.fit_intercept = fit_intercept
        self.X = X
        self._A = np.array([])
        self.n_samples, self.n_features = self.X.shape
        self.sparse_matrix = issparse(self.X)
        shape = np.zeros((2,))
        m = min(self.n_samples, self.n_features)
        shape[0] = m
        shape[1] = m
        self.shape = shape.astype(int)

    def get_matrix(self):  # outputs A in matrix form
        if self._A.size == 0:
            self.build_matrix()
        return self._A

    def build_kernel_system_matrix(self, K):
        r"""Build the system matrix K(K+\alpha I) of kernel ridge regression"""
        m = K.shape[0]
        self.shape[0] = m
        self.shape[1] = m
        self._A = K.copy()
        self._A.flat[:: m + 1] += self.alpha  # adding on regularization
        self._A = K.dot(self._A)

    def build_matrix(self):  # dense_output=None
        """
        Converts to dense matrix
        """
        # Return dense matrix if dense_output = True

        if self.n_features > self.n_samples:
            # dual system matrix for
            # (X X.T + alpha I) z = b, then w = X.T z
            self._A = safe_sparse_dot(self.X, self.X.T)
            # dense_output=dense_output
        else:
            # primal system matrix for
            # (X.T X + alpha I) w = X.T b
            self._A = safe_sparse_dot(self.X.T, self.X)
            # dense_output=dense_output

        if self.sparse_matrix:
            v = self._A.diagonal() + self.alpha_vec
            self._A.setdiag(v)
        else:
            m = self.shape[0]
            self._A.flat[:: m + 1] += self.alpha_vec[
                0
            ]  # doesn't work with sparse matrices

    @abstractmethod
    def __matmul__(self, v):
        """
        Takes a vector v return the matrix-vector product A @ v

        Overload the '__matmul__' operator, used through the alias @.

        Parameters
        ----------
        v : sparse array or ndarray
            Input vector

        Returns
        -------
        A_v : sparse array or ndarray
            Output vector of the matrix-vector multiplication A @ v
        """
        pass

    @abstractmethod
    def get_elements(self, rows_idx, cols_idx):
        """
        Get ths rows and columns of A indexed by rows_idx and cols_idx

        Parameters
        ----------
        rows_idx: list of ints
            Row indices
        cols_idx: list of ints
            Column indices
        """
        if isspmatrix_coo(self.X):
            raise NameError(
                "AAbstract.get_elements does not allow for X to be a sparse coo matrix. \
                This is because coo matrices do not allow indexing, e.g X[1,1]"
            )
        pass

    @abstractmethod
    def get_rows(self, rows_idx):
        """Get the rows of A indexed by a list of integers rows_idx"""
        if isspmatrix_coo(self.X):
            raise NameError(
                "AAbstract.get_rows does not allow for X to be a sparse coo matrix. \
                This is because coo matrices do not allow indexing, e.g X[1, 1]"
            )
        pass

    @abstractmethod
    def get_diag(self):
        """Get the diagonal of A"""
        # if isspmatrix_coo(self.X):
        #     raise NameError(
        #         "AAbstract.get_diag does not allow for X to be a sparse coo matrix. \
        #         This is because coo matrices do not allow indexing, e.g X[1,1]"
        #     )
        pass


class AMatrix(AAbstract):
    """
    Implements the AMatrix class where self._A is a dense or sparse matrix
    """

    def __init__(
        self, X, alpha=1.0, fit_intercept=True, K=None,
    ):
        super().__init__(
            X, alpha, fit_intercept,
        )
        if K is None:
            self.build_matrix()  # need to build matrix in this case
        else:
            self.build_kernel_system_matrix(K)

    def __matmul__(self, v):
        """
        Takes a vector v return the matrix-vector product A @ v

        Overload the '__matmul__' operator, used through the alias @.

        Parameters
        ----------
        v : sparse array or ndarray
            Input vector

        Returns
        -------
        A_v : sparse array or ndarray
            Output vector of the matrix-vector multiplication A @ v
        """
        A_v = safe_sparse_dot(self._A, v)
        return A_v

    def get_elements(self, rows_idx, cols_idx):
        return (self._A[rows_idx, :])[
            :, cols_idx
        ]  # This is so incredibly ugly, but I can't find a better solution

    def get_rows(self, rows_idx):
        return self._A[rows_idx, :]

    def get_diag(self):
        return self._A.diagonal().copy()


class AOperator(AAbstract):
    """
    Implements the AMatrix class where self._A is represented as an operator
    """

    def __init__(
        self, X, alpha=1.0, fit_intercept=True,
    ):

        super().__init__(
            X, alpha, fit_intercept,
        )

    def __matmul__(self, v):
        """
        Takes a vector v return the matrix-vector product A @ v

        Overload the '__matmul__' operator, used through the alias @.

        Parameters
        ----------
        v : sparse array or ndarray
            Input vector

        Returns
        -------
        A_v : sparse array or ndarray
            Output vector of the matrix-vector multiplication A @ v
        """
        # dual system, solves (X X.T + alpha I) z = b, then w = X.T z
        if self.n_features > self.n_samples:
            A_v = safe_sparse_dot(self.X, safe_sparse_dot(self.X.T, v)) + self.alpha * v
        # primal system, solves (X.T X + alpha I) w = X.T b
        else:
            A_v = safe_sparse_dot(self.X.T, safe_sparse_dot(self.X, v)) + self.alpha * v
        return A_v

    def get_elements(self, rows_idx, cols_idx):
        # if sparse matrix format is coo slices doesn't work!
        # assert type(self._A) is coo_matrix # TO DO LATER

        if self.n_features > self.n_samples:
            # This is good for CSR format!
            AIJ = safe_sparse_dot(self.X[rows_idx, :], self.X[cols_idx, :].T)
        else:
            # This is good for CSC format! Consider changing format depending
            AIJ = safe_sparse_dot(self.X[:, rows_idx].T, self.X[:, cols_idx])
        # Adding on alpha to the diagonal.
        interIJ = list(set(rows_idx).intersection(cols_idx))
        for i in interIJ:
            AIJ[rows_idx.index(i), cols_idx.index(i)] += self.alpha
        return AIJ

    def get_rows(self, rows_idx):
        if self.n_features > self.n_samples:
            AI = safe_sparse_dot(self.X[rows_idx, :], self.X.T)
        else:
            AI = safe_sparse_dot(self.X[:, rows_idx].T, self.X)
        cols_idx = list(range(self.shape[0]))
        # adding on alpha to the diagonal
        interIJ = list(set(rows_idx).intersection(cols_idx))
        for i in interIJ:
            AI[rows_idx.index(i), cols_idx.index(i)] += self.alpha
        return AI

    def get_diag(self):
        if self.n_features > self.n_samples:
            d = safe_sparse_dot(self.X, self.X.T).diagonal()
        else:
            d = safe_sparse_dot(self.X.T, self.X).diagonal()
        return d
