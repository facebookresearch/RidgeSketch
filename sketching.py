"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


from abc import ABC, abstractmethod
import numpy as np
import numpy.matlib as matlib
from scipy.sparse import csc_matrix, csr_matrix, issparse
from math import floor


from srht import (
    subsample_fwht,
    next_power_of_two,
)
from sparse_tools import row_wise_mult, pad_with_zeros


def generate_sample_indices(dim, sketch_size=None):
    """Generates seed for random subset of indices.

    Returns a list of `sketch_size` (1 if set to None)
    indices to subsample from a matrix with `dim` rows.
    """
    return np.random.choice(dim, size=sketch_size, replace=False).tolist()


class Sketch(ABC):
    """Abstract class defining sketch method interface"""

    def __init__(self, A, b, sketch_size):
        self.A = A
        self.b = b
        self.sketch_size = sketch_size
        self.m = A.shape[0]

    @abstractmethod
    def sketch(self, r):
        """Generates sketched system

        Args:
            A (AAbstract): abstract object representing the matrix for system
            Aw = b. A is either an 'AMatrix' or an 'AOperator'.
            sketch_size (int): sketching dimension for matrix S

        Returns:
            tuple of (SA, SAS, rs), where rs is the sketched residual
        """
        pass

    @abstractmethod
    def update_iterate(self, w, lmbda, step_size=1.0):
        """Updates the weights using the least norm solution lmbda and the step size.

        Args:
            w (np.array): weight vector at step k
            lmbda (np.array): least norm solution to S.TAS lmbda = rs
            step_size (float): step size, set to 1 for sketch-and-project
                original method

        Returns (np.array): updated weights.
        """
        pass


class SubsampleSketch(Sketch):
    def __init__(self, A, b, sketch_size):
        super().__init__(A, b, sketch_size)
        self.sample_indices = None

    def set_sketch(self):
        """Generates subsampling indices"""
        self.sample_indices = generate_sample_indices(self.m, self.sketch_size)

    def sketch(self, r):
        self.set_sketch()
        SA = self.A.get_rows(self.sample_indices)
        # SA = self.A[self.s, :]
        SAS = SA[:, self.sample_indices]
        rs = r[self.sample_indices]
        return SA, SAS, rs

    def update_iterate(self, w, lmbda, step_size=1.0):
        if self.sample_indices is None:
            raise AttributeError("sample indices before updating iterate")
        w[self.sample_indices] -= step_size * lmbda
        return w


class CoordinateDescentSketch(SubsampleSketch):
    """Implements Coordinate Descent using Subsample Sketching"""

    def __init__(self, A, b, sketch_size):
        if sketch_size != 1:
            raise ValueError("Coordinate Descent requires sketch_size = 1")

        self.prob = A.get_diag()
        self.prob = self.prob / np.sum(self.prob)
        super().__init__(A, b, sketch_size)

    def set_sketch(self):
        self.sample_indices = self.generate_sample_index()

    def generate_sample_index(self):
        return np.random.choice(self.m, size=1, replace=False, p=self.prob).tolist()


class GaussianSketch(Sketch):
    def __init__(self, A, b, sketch_size):
        super().__init__(A, b, sketch_size)
        self.S = None

    def set_sketch_matrix(self):
        """Generates a gaussian sketching matrix"""
        self.S = np.random.normal(
            size=(self.m, self.sketch_size), scale=np.sqrt(self.sketch_size)
        )

    def sketch(self, r):
        self.set_sketch_matrix()
        SA = (self.A @ self.S).T
        SAS = np.matmul(SA, self.S)
        rs = np.matmul(self.S.T, r)
        return SA, SAS, rs

    def update_iterate(self, w, lmbda, step_size=1.0):
        if self.S is None:
            raise AttributeError("sample a sketch matrix before updating iterate")
        w -= step_size * np.matmul(self.S, lmbda)
        return w


class CountSketch(Sketch):
    def __init__(self, A, b, sketch_size, build_matrix=True):
        super().__init__(A, b, sketch_size)
        self.build_matrix = build_matrix  # boolean to build sketch matrix
        self.S = None
        self.cols = None
        self.signs = None

    def set_sketch(self):
        """
        Generates a count-sketch matrix or a representation of it in two arrays

        the method returns a matrix ``S`` of size (m, sketch_size) where m is
        the size of matrix ``A``. There is only one non-zero entry per row
        which is set randomly to -1 or 1 with probability 1/2.

        The i--th value of ``cols`` denotes the index of the column
        (sampled uniformly at random in {1, sketch_size}) of the i--th row of
        the count sketch matrix ``S`` which contains a non-zero value.
        These non-zero values are set to randomly -1 or 1 with equal
        probability and stored in ``signs``.
        """
        if self.build_matrix:
            # Count Sketching with sparse matrix multiplication
            rows = np.arange(self.m)
            # cols = np.random.randint(0, self.sketch_size, self.m)

            # Sampling columns making sure that every column gets sampled
            vs = np.squeeze(
                matlib.repmat(
                    np.arange(self.sketch_size), 1, int(self.m / self.sketch_size)
                )
            )
            cols = np.concatenate(
                (vs, np.random.randint(0, self.sketch_size, self.m % self.sketch_size))
            )
            np.random.shuffle(cols)  # in-place function

            signs = np.random.choice([1, -1], self.m)
            # S in CSC format should be more efficient because CSC format is better for left-hand side multiplications A @ S
            # and S^T would be CSR leading to efficient right-hand side multiplication S^T * r
            # Note: I don't understand why the count sketch matrix (S^T for us) is a CSC matrix in the code provided here
            # https://github.com/scipy/scipy/blob/v1.4.1/scipy/linalg/_sketches.py#L57-L168
            self.S = csc_matrix((signs, (rows, cols)), shape=(self.m, self.sketch_size))
        else:
            # Implicit count Sketching with slicing
            # self.cols = np.random.choice(range(0, self.sketch_size), size=self.m)
            # Sampling columns making sure that every column gets sampled
            vs = np.squeeze(
                matlib.repmat(
                    np.arange(self.sketch_size), 1, int(self.m / self.sketch_size)
                )
            )
            self.cols = np.concatenate(
                (vs, np.random.randint(0, self.sketch_size, self.m % self.sketch_size))
            )
            np.random.shuffle(self.cols)  # in-place function

            self.signs = np.random.choice([-1, 1], size=self.m)

    def sketch(self, r):
        """
        Computes S^T A, S^T A S and S^T r for S being the count sketch matrix
        """
        self.set_sketch()
        if self.build_matrix:
            SA = (self.A @ self.S).T  # A @ S --> S should be CSC
            SAS = csc_matrix.dot(SA, self.S)  # SA @ S --> S should be CSC
            rs = csr_matrix.dot(r.T, self.S).T  # r.T @ S --> S should be CSC
            # rs = csr_matrix.dot(self.S.T, r)  # S.T @ r --> S.T should be CSC, which is the case if S is CSR
        else:
            # Computational issue: two loops over m
            SA = np.zeros((self.sketch_size, self.m))
            rs = np.zeros((self.sketch_size, 1))
            for i in range(self.m):  # TODO: remove loop or vectorize this step
                SA[self.cols[i], :] += (
                    self.signs[i] * self.A._A[i, :]
                )  # could be efficient if A were CSR sparse matrix
                # SA[self.cols[i],:] += self.signs[i] * self.A.get_rows(i) # if A is an AOperator
                rs[self.cols[i], :] += self.signs[i] * r[i]
            SAS = np.zeros(
                (self.sketch_size, self.sketch_size)
            )  # need for SA to be built before computing SAS
            for i in range(self.m):  # TODO: remove loop or vectorize this step
                SAS[:, self.cols[i]] += self.signs[i] * SA[:, i]
        return SA, SAS, rs

    def update_iterate(self, w, lmbda, step_size=1.0):
        if self.build_matrix:
            if self.S is None:
                raise AttributeError("Sample a sketch matrix before updating iterate")
            w -= step_size * csr_matrix.dot(
                self.S, lmbda
            )  # S * lmbda --> S should be CSR
        else:
            if self.cols is None or self.signs is None:
                raise AttributeError("Sample a sketch matrix before updating iterate")
            for i in range(self.m):  # TODO: remove loop or vectorize this step
                w[i] -= step_size * self.signs[i] * lmbda[self.cols[i]]
        return w


class SubcountSketch(Sketch):
    def __init__(self, A, b, sketch_size, sum_size=10):
        super().__init__(A, b, sketch_size)
        self.sum_size = sum_size
        self.subsampling_size = self.set_subsampling_size()
        self.sample_indices = None  # list
        self.signs = None  # array

    def set_subsampling_size(self):
        """Sets subsampling size based on m, sketch and sum sizes"""
        if self.sum_size * self.sketch_size > self.m:
            self.sum_size = floor(self.m / self.sketch_size)
        subsampling_size = int(self.sketch_size * self.sum_size)

        return subsampling_size

    def set_sketch(self):
        r"""
        Generates a Subcount sketch matrix.

        $S^T = \Sigma D I_{C:}$
        Where
        - $C$ is a set of s random indices in [m]
        - $D$ is a diagonal matrix of order s, which random -1 and 1
        - $\Sigma$ is a matrix which sums every k rows of the right matrix
                    [1 ... 1 0 ...................... 0]
                    [0 ... 0 1 ... 1 0 .............. 0]
                                    .
        Sigma =                     .
                                    .
                    [0 ...................... 0 1 ... 1]

        subsampling_size: s
        sum_size : k
        sketch_size : tau = s / k
        """
        # Subcount Sketching with sparse matrix multiplication
        self.sample_indices = generate_sample_indices(self.m, self.subsampling_size)
        self.signs = np.random.choice([1, -1], self.subsampling_size)

    def _sum_every_rows(self, M, sum_size):
        """Return the matrix where every `sum_size` rows are summed."""
        m = M.shape[0]
        if m % sum_size != 0:
            raise ValueError(
                "Matrix must have a number of rows which is a multiple of `sum_size`"
            )
        if issparse(M):  # subefficient for sparse matrices
            M = M.toarray()
        return M.reshape(-1, sum_size, M.shape[1]).sum(axis=1)

    def sketch(self, r):
        """Computes S^T A, S^T A S and S^T r for S being the count sketching matrix."""
        self.set_sketch()
        SA = self.A.get_rows(self.sample_indices)  # row subsampling
        SA = row_wise_mult(SA, self.signs)  # diagonal rademacher multiplication
        SA = self._sum_every_rows(SA, self.sum_size)  # summing every sum_size rows

        SAS = row_wise_mult(SA.T[self.sample_indices, :], self.signs)
        SAS = self._sum_every_rows(SAS, self.sum_size)

        rs = r.reshape(-1, 1)
        rs = row_wise_mult(r[self.sample_indices, :], self.signs)
        rs = self._sum_every_rows(rs, self.sum_size)
        return SA, SAS, rs

    def update_iterate(self, w, lmbda, step_size=1.0):
        if self.sample_indices is None or self.signs is None:
            raise AttributeError("sample a sketch matrix before updating iterate")
        w[self.sample_indices] -= step_size * row_wise_mult(
            np.repeat(lmbda, self.sum_size).reshape(-1, 1), self.signs
        )
        return w


class HadamardSketch(Sketch):
    """
    Implement the Subsampled Randomized Hadamard Transform (SRHT) with FWHT
    """

    def __init__(self, A, b, sketch_size, build_matrix=False):
        self.b = b
        self.sketch_size = sketch_size
        self.m = A.shape[0]
        self.build_matrix = build_matrix  # boolean to build sketch matrix

        self.m_prime = next_power_of_two(self.m)  # next power of two s.t. m' > m
        self.normalization = (self.sketch_size * self.m_prime) ** (
            -0.5
        )  # normalization factor of [Woodruff 2014]
        if self.build_matrix:
            self.A = A  # AOperator or AMatrix
        else:
            # Padding done only at initialization
            self.A = pad_with_zeros(
                A.get_matrix(), self.m_prime - self.m
            )  # WARNING: padded matrix

        # Refreshed at each step
        self.rademacher_diag = None
        self.sample_indices = None
        self.S = None

    def build_sketch_matrix(self):
        """Build the sketch matrix based on the sampled random arrays defining the SRHT.

        The arrays `self.rademacher_diag` and `self.sample_indices` correspond respectively to
        the randomness in matrices D and P of the full transform: S^T = PHD.
        """
        # Start with a padding matrix: identity with zeros at the bottom
        self.S = pad_with_zeros(
            np.eye(self.m), self.m_prime - self.m
        )  # S <- [I_{m} | 0]^T

        # Multiply by the random diagonal matrix
        # D = -np.eye(m_prime) # for debugging
        D = np.diag(
            self.rademacher_diag
        )  # Previously: D = np.diag(np.random.choice([1, -1], size=m_prime))
        self.S = D.dot(self.S)  # S <- D @ [I_{m} | 0]^T

        # Apply Subsampled Hadamard transform (the most efficient one seems to be fwht)
        self.S = self.normalization * subsample_fwht(
            self.S, self.sample_indices
        )  # S <- P @ H_{m'} @ D @ [I_{m} | 0]^T

        # Transpose the result
        self.S = (
            self.S.T
        )  # S <- (P @ H_{m'} @ D @ [I_{m} | 0]^T)^T = [I_{m} | 0] @ D @ H_{m'} P^T

    def set_sketch(self):
        """Generates a SRHT matrix or random elements of a SRHT"""
        self.rademacher_diag = np.random.choice(
            [1, -1], size=self.m_prime
        )  # 1st source of randomness
        self.sample_indices = generate_sample_indices(
            self.m_prime, self.sketch_size
        )  # 2nd source of randomness
        # self.sample_indices.sort() # TODO: Is it usefull to sort the sampled indices?
        # self.rademacher_diag = -np.ones(self.m_prime) # for debugging
        # self.sample_indices = np.random.choice(self.m_prime, self.sketch_size, replace=False) # for debugging
        if self.build_matrix:
            self.build_sketch_matrix()

    def sketch(self, r):
        self.set_sketch()
        if self.build_matrix:
            # Via matrix-matrix and matrix-vector multiplications
            SA = (self.A @ self.S).T
            SAS = np.matmul(SA, self.S)
            rs = np.matmul(self.S.T, r)
        else:
            # Via direct SRHT on A, SA and r, it uses padding and return objects of correct sizes
            # Building SA
            DA = row_wise_mult(
                self.A, self.rademacher_diag
            )  # A is padded in the initialization
            SA = self.normalization * subsample_fwht(
                DA, self.sample_indices
            )  # dense (tau, m) matrix

            # Building SAS
            SA_padded = pad_with_zeros(
                SA, self.m_prime - self.m, side="right"
            )  # dense (tau, m') matrix # Second padding of SA equivalent to pad A on the right
            DAS = row_wise_mult(SA_padded.T, self.rademacher_diag)
            SAS = self.normalization * subsample_fwht(
                DAS, self.sample_indices
            )  # dense (tau, tau) matrix

            # Building rs
            r_padded = pad_with_zeros(
                r, self.m_prime - self.m
            )  # r is NOT already padded
            Dr_padded = np.multiply(r_padded, self.rademacher_diag[:, np.newaxis])
            rs = self.normalization * subsample_fwht(Dr_padded, self.sample_indices)
        return SA, SAS, rs

    def update_iterate(self, w, lmbda, step_size=1.0):
        if self.S is None and (
            self.rademacher_diag is None or self.sample_indices is None
        ):
            raise AttributeError("sample a sketch matrix before updating iterate")

        if self.build_matrix:
            return w - step_size * np.matmul(self.S, lmbda)
        else:
            D = np.diag(self.rademacher_diag)
            Slmbda = (
                self.normalization
                * (lmbda.T @ subsample_fwht(D, self.sample_indices)).T
            )
            return w - step_size * Slmbda[0 : self.m]

