"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.linalg import hadamard
from itertools import combinations

from srht import fwht, subsample_fwht, next_power_of_two
from sparse_tools import csr_row_set_nz_to_val, pad_with_zeros


@pytest.fixture(scope="module")
def b():
    # 1-D numpy array of dimension (8,)
    return np.array([1, 0, 1, 0, 0, 1, 1, 0])


@pytest.fixture(scope="module")
def b_2D():
    # 2-D numpy array of dimension (8, 1)
    return np.array([1, 0, 1, 0, 0, 1, 1, 0]).reshape(-1, 1)


@pytest.fixture(scope="module")
def b_transform():
    # Ground truth: Hadamard transform of b taken from Wikipedia
    return np.array([4, 2, 0, -2, 0, 2, 0, 2])


@pytest.fixture(scope="module")
def A():
    # 2-D numpy array of dimension (4, 3)
    return np.array([[0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0]])


@pytest.fixture(scope="module")
def A_csr():
    # scipy.sparse CSR matrix of dimension (4, 3)
    return sparse.csr_matrix([[0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0]])


@pytest.fixture(scope="module")
def A_transform(A):
    # Ground truth: Hadamard transform of A
    return hadamard(4) @ A


@pytest.mark.hadamard
class TestHadamardTransform:
    def test_powers_of_two_tools(self):
        assert next_power_of_two(4) == 4
        assert next_power_of_two(14) == 16

    def test_inputs(self, b):
        b = b.reshape(1, -1)

        with pytest.raises(ValueError):
            subsample_fwht(b, ind="blabla")

        # no subsampling
        subsample_fwht(b, ind=None)

        all_indices = np.arange(len(b))
        subsample_fwht(b, ind=all_indices)

        # check power of 2 of the input
        with pytest.raises(ValueError):
            fwht(np.array([0, 1, 2]))

    def test_output_type(self, b, b_2D, A, A_csr):
        """Verifies the fwht function returns an output with same
        type as input"""
        # 1-D numpy array
        b_fwht = fwht(b)
        all_indices = np.arange(len(b))
        b_sub_fwht = subsample_fwht(b, all_indices)
        assert isinstance(b_fwht, np.ndarray)
        assert isinstance(b_sub_fwht, np.ndarray)

        # 2-D numpy array
        b_fwht = fwht(b)
        all_indices = np.arange(b_2D.shape[0])
        b_sub_fwht = subsample_fwht(b, all_indices)
        assert isinstance(b_fwht, np.ndarray)
        assert isinstance(b_sub_fwht, np.ndarray)

        # 2-D numpy matrix
        A_fwht = fwht(A)
        all_indices = np.arange(A.shape[0])
        A_sub_fwht = subsample_fwht(A, all_indices)
        assert isinstance(A_fwht, np.ndarray)
        assert isinstance(A_sub_fwht, np.ndarray)

        # Any sparse matrix outputs CSR matrix
        all_indices = np.arange(A_csr.shape[0])
        for sparse_format in ["csr", "csc", "coo", "lil", "dok"]:
            A_sparse = A_csr.asformat(sparse_format)
            A_sparse_fwht = fwht(A_sparse)
            A_sparse_sub_fwht = subsample_fwht(A_sparse, all_indices)
            assert isinstance(A_sparse_fwht, np.ndarray)
            assert isinstance(A_sparse_sub_fwht, np.ndarray)

    def test_hadamard_matrices(self):
        """Verifies the fwht function produces correct Hadamard matrices"""
        # Hard-coded Hadamard matrices through Sylvester's construction
        # https://en.wikipedia.org/wiki/Hadamard_matrix#Sylvester's_construction
        H_true = np.empty(())
        for idx in np.arange(4):
            n = 2 ** idx
            if n == 1:
                H_true = np.array([[1]])
            else:
                H_true = np.block([[H_true, H_true], [H_true, -H_true]])

            H_fwht = fwht(np.eye(n))
            assert_array_equal(H_true, H_fwht)

    def test_1D_numpy_vector(self, b, b_transform):
        """Verifies the fwht function works on 1-D numpy array"""
        assert_array_equal(b_transform, fwht(b))

    def test_2D_numpy_vector(self, b_2D, b_transform):
        """Verifies the fwht function works on 2-D numpy array"""
        # Column vector
        assert_array_equal(b_transform, fwht(b_2D).flatten())
        # Row vector
        assert_array_equal(b_transform, fwht(b_2D.reshape(1, -1)).flatten())

    def test_2D_numpy_matrix(self, A, A_transform):
        """Verifies the fwht function works on 2-D numpy matrix"""
        A_fwht = fwht(A)
        assert_array_equal(A_transform, A_fwht)

    def test_2D_sparse_matrix(self, A_csr, A_transform):
        """Verifies the fwht function works on 2-D scipy sparse array"""
        A_sparse_fwht = fwht(A_csr)
        assert_array_equal(A_transform, A_sparse_fwht)

    def test_random_transforms(self):
        """
        Verifies the 'fwht', 'subsample_fwht' function work
        with subsampling on random data
        """
        # Ten random sizes
        for m in 2 ** np.random.choice(range(4, 8), 10):
            X = np.random.rand(m, m)
            X_transform = hadamard(m) @ X  # ground truth
            X_fwht = fwht(X)
            # 3 sketch sizes
            for sketch_size in [1, 10, m]:
                sample_indices = np.random.choice(range(m), sketch_size, replace=False)
                assert_allclose(
                    X_transform[sample_indices, :], X_fwht[sample_indices, :]
                )
                assert_allclose(
                    X_transform[sample_indices, :], subsample_fwht(X, sample_indices)
                )

    def test_subsampled_transforms(self, b, b_transform, A, A_csr, A_transform):
        """Verifies the fwht, subsample_fwht function works with subsampling"""
        m = len(b)
        # Checking all possible sketching combinations
        for sketch_size in range(1, m):
            for sample_indices in combinations(range(m), sketch_size):
                sample_indices = np.array(sample_indices)
                assert_array_equal(
                    b_transform[sample_indices], subsample_fwht(b, sample_indices)
                )

        m = A.shape[0]
        # Checking all possible sketching combinations
        for sketch_size in range(1, m):
            for sample_indices in combinations(range(m), sketch_size):
                sample_indices = np.array(sample_indices)
                assert_array_equal(
                    A_transform[sample_indices, :], subsample_fwht(A, sample_indices)
                )
                assert_array_equal(
                    A_transform[sample_indices, :],
                    subsample_fwht(A_csr, sample_indices),
                )


@pytest.mark.sparse
class TestSparseMatrixTools:
    def test_row_set_nz_to_val(self, A_csr):
        # Checking that csr_row_set_nz_to_val function is correct
        B = A_csr.copy()

        # works only for sparse matrices
        with pytest.raises(ValueError):
            csr_row_set_nz_to_val(B.toarray(), 0)

        # Replace nnz values by 0
        csr_row_set_nz_to_val(B, 0)
        assert_array_equal(B[0, :].toarray().flatten(), np.zeros(B.shape[1]))
        # Replace nnz values by 7
        csr_row_set_nz_to_val(B, 1, value=7)
        assert_array_equal(B[1, :].toarray().flatten(), np.array([7, 0, 7]))
        # If the row is already full of zeros
        csr_row_set_nz_to_val(B, 2, value=9999)
        assert_array_equal(B[2, :].toarray().flatten(), np.zeros(3))

    def test_padding(self, b, A_csr, A):
        # one dimensional input
        b_padded = pad_with_zeros(b, 3, side="bottom")
        B = np.array([1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
        assert_array_equal(B, b_padded)

        # sparse
        A_padded = pad_with_zeros(A_csr, 1, side="bottom")
        B = sparse.csr_matrix([[0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0], [0, 0, 0]])
        assert_array_equal(B.toarray(), A_padded.toarray())

        A_padded = pad_with_zeros(A_csr, 1, side="top")
        B = sparse.csr_matrix([[0, 0, 0], [0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0]])
        assert_array_equal(B.toarray(), A_padded.toarray())

        A_padded = pad_with_zeros(A_csr, 1, side="left")
        B = sparse.csr_matrix([[0, 0, 1, 0], [0, 2, 0, 3], [0, 0, 0, 0], [0, 4, 0, 0]])
        assert_array_equal(B.toarray(), A_padded.toarray())

        A_padded = pad_with_zeros(A_csr, 1, side="right")
        B = sparse.csr_matrix([[0, 1, 0, 0], [2, 0, 3, 0], [0, 0, 0, 0], [4, 0, 0, 0]])
        assert_array_equal(B.toarray(), A_padded.toarray())

        # dense
        A_padded = pad_with_zeros(A, 1, side="bottom")
        B = np.array([[0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0], [0, 0, 0]])
        assert_array_equal(B, A_padded)

        A_padded = pad_with_zeros(A, 1, side="top")
        B = np.array([[0, 0, 0], [0, 1, 0], [2, 0, 3], [0, 0, 0], [4, 0, 0]])
        assert_array_equal(B, A_padded)

        A_padded = pad_with_zeros(A, 1, side="left")
        B = np.array([[0, 0, 1, 0], [0, 2, 0, 3], [0, 0, 0, 0], [0, 4, 0, 0]])
        assert_array_equal(B, A_padded)

        A_padded = pad_with_zeros(A, 1, side="right")
        B = np.array([[0, 1, 0, 0], [2, 0, 3, 0], [0, 0, 0, 0], [4, 0, 0, 0]])
        assert_array_equal(B, A_padded)
