"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from numpy.testing import assert_allclose
import pytest
import random

from sklearn.utils.extmath import safe_sparse_dot

from a_matrix import AMatrix
from a_matrix import AOperator


@pytest.mark.amatrix
class TestAMatrix:
    def test_a_matrix_dense_vector_multiplication(self, X_dense):
        alpha = 1.0
        A = AOperator(alpha=alpha, X=X_dense)
        A_d = AMatrix(alpha=alpha, X=X_dense)
        v_dim = A.shape[0]
        v = np.random.rand(v_dim, 1)
        # assert_allclose(A_d.multr(v), A.multr(v))
        # assert_allclose(
        #     np.matmul(A_d._A, v), A_d.multr(v)
        # )  # Compare against numpy matmul
        assert_allclose(A_d @ v, A @ v)
        assert_allclose(np.matmul(A_d._A, v), A_d @ v)

    def test_a_matrix_sparse_vector_multiplication(self, X_sparse):
        alpha = 1.0
        A = AOperator(alpha=alpha, X=X_sparse)
        A_d = AMatrix(alpha=alpha, X=X_sparse)
        v_dim = A.shape[0]
        v = np.random.rand(v_dim, 1)
        assert_allclose(A_d @ v, A @ v)
        assert_allclose(safe_sparse_dot(A_d._A, v), A_d @ v)

    # def test_a_matrix_left_vector_multiplication(self, X_dual):
    #     alpha = 1.0
    #     A = AOperator(alpha=alpha, X=X_dual)
    #     A_d = AMatrix(alpha=alpha, X=X_dual)
    #     v_dim = A.shape[0]
    #     v = np.random.rand(1, v_dim)
    #     # assert_allclose(A_d.multl(v), A.multl(v))
    #     # assert_allclose(np.matmul(v, A_d._A), A_d.multl(v))
    #     assert_allclose(v*A_d, v*A)
    #     assert_allclose(np.matmul(v, A_d._A), v*A_d)

    def test_a_matrix_row_sampling(self, X_dense):
        alpha = 1.0
        A = AOperator(alpha=alpha, X=X_dense)
        A_d = AMatrix(alpha=alpha, X=X_dense)
        v_dim = A.shape[0]
        row_idx = random.sample(range(0, v_dim), int(np.sqrt(v_dim)))
        assert_allclose(A_d.get_rows(row_idx), A.get_rows(row_idx))
        assert_allclose(A_d._A[row_idx, :], A.get_rows(row_idx))

    def test_a_matrix_element_sampling(self, X_dense):
        alpha = 1.0
        A = AOperator(alpha=alpha, X=X_dense)
        A_d = AMatrix(alpha=alpha, X=X_dense)
        v_dim = A.shape[0]
        row_idx = random.sample(range(0, v_dim), int(np.sqrt(v_dim)))
        col_idx = random.sample(range(0, v_dim), int(np.sqrt(v_dim)))
        assert_allclose(
            A_d.get_elements(row_idx, col_idx), A.get_elements(row_idx, col_idx)
        )
