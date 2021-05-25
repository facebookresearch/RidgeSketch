"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import pytest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import find, csc_matrix, random

from a_matrix import AMatrix

import sketching
from sketching import (
    generate_sample_indices,
    CoordinateDescentSketch,
    CountSketch,
    SubcountSketch,
    HadamardSketch,
)
from tests.conftest import SKETCH_METHODS

N_ROWS = 10
SKETCH_SIZE = 3


@pytest.fixture(scope="module")
def A():
    X = np.random.rand(N_ROWS, N_ROWS)
    A = AMatrix(alpha=0.1, X=X)
    return A


@pytest.fixture(scope="module")
def A_sparse():
    X = random(N_ROWS, N_ROWS, density=0.7, format="csr")
    A = AMatrix(alpha=0.1, X=X)
    return A


@pytest.fixture(scope="module")
def b():
    return np.random.rand(N_ROWS, 1)


@pytest.fixture
def sketch_size():
    return SKETCH_SIZE


@pytest.mark.sketch
class TestSketching:
    @pytest.mark.parametrize("method_name", SKETCH_METHODS)
    def test_sketch_instantiation(self, method_name, A, b, sketch_size):
        """
        Confirms sketch method can be instantiated
        based on Sketch Abstract Base Class
        """
        if method_name == "CoordinateDescentSketch":
            sketch_size = 1
        sketching_method = getattr(sketching, method_name)(A, b, sketch_size)

        # for HadamardSketch sketching_method.A is a np.array
        # instead of AMatrix or AOperator if build_matrix == False
        if method_name != "HadamardSketch":
            assert_allclose(sketching_method.A.get_matrix(), A.get_matrix())
        assert sketching_method.sketch_size == sketch_size

    @pytest.mark.parametrize("method_name", SKETCH_METHODS)
    def test_sketch_output_shapes(self, method_name, A, b, sketch_size):
        """Verifies shapes of sketch output: SA, SAS, rs"""
        if method_name == "CoordinateDescentSketch":
            sketch_size = 1
        sketch_method = getattr(sketching, method_name)(A, b, sketch_size)
        r = -b.copy()
        SA, SAS, rs = sketch_method.sketch(r)
        if not method_name == "HadamardSketch":
            assert SA.shape == (sketch_size, A.shape[0])
        assert SAS.shape == (sketch_size, sketch_size)
        assert rs.shape == (sketch_size, 1)

    @pytest.mark.parametrize("method_name", SKETCH_METHODS)
    def test_update_iterate(self, method_name, A, b, sketch_size):
        """Tests dimensions of weights are unchanged"""
        if method_name == "CoordinateDescentSketch":
            sketch_size = 1
        lmbda = np.random.rand(sketch_size, 1)
        sketch_method = getattr(sketching, method_name)(A, b, sketch_size)
        r = -b.copy()
        w = np.random.rand(r.shape[0], 1)
        # should raise exception if updated before sketching
        with pytest.raises(AttributeError):
            sketch_method.update_iterate(w, lmbda)
        sketch_method.sketch(r)
        assert w.shape == sketch_method.update_iterate(w, lmbda).shape

    @pytest.mark.parametrize("method_name", SKETCH_METHODS)
    def test_update_iterate_sparse(self, method_name, A_sparse, b, sketch_size):
        """
        Tests instantiation, sketch and
        dimensions of weights are unchanged
        """
        if method_name == "CoordinateDescentSketch":
            sketch_size = 1
        lmbda = np.random.rand(sketch_size, 1)
        sketch_method = getattr(sketching, method_name)(A_sparse, b, sketch_size)
        r = -b.copy()
        w = np.random.rand(r.shape[0], 1)
        # should raise exception if updated before sketching
        with pytest.raises(AttributeError):
            sketch_method.update_iterate(w, lmbda)
        sketch_method.sketch(r)
        assert w.shape == sketch_method.update_iterate(w, lmbda).shape


@pytest.mark.subsample
class TestSubsample:
    def test_subsample_indices(self, A, b, sketch_size):
        """Verifies generates indices are valid"""
        indices = generate_sample_indices(A.shape[0], sketch_size)
        indices = np.array(indices)
        assert all((indices >= 0) & (indices < A.shape[0]))


@pytest.mark.cd
class TestCoordinateDescent:
    def test_cd_sketch_size(self, A, b):
        """Verifies that the sketch size must be one"""
        with pytest.raises(ValueError):
            CoordinateDescentSketch(A, b, 3)


def build_count_sketch_matrix(cols, signs, sketch_size):
    """
    Returns the count sketch matrix S (m x sketch_size columns)
    with one non-zero value per row

    These values (-1 or 1) are given in signs.
    """
    assert len(cols) == len(signs)
    m = len(cols)
    rows = np.arange(m)
    S = csc_matrix((signs, (rows, cols)), shape=(m, sketch_size))
    return S


@pytest.mark.countsketch
class TestCountSketch:
    def test_count_sketch_nnz(self, A, b, sketch_size):
        """
        Verifies the count sketch matrix S contains only
        one non-zero value (-1 or 1) per row
        """
        m = A.shape[0]
        count_sketch = CountSketch(A, b, sketch_size, build_matrix=True)
        count_sketch.set_sketch()

        nnz_per_row = count_sketch.S.getnnz(axis=1)
        nnz_values = find(count_sketch.S)[2]

        # check that each row contains one and only one non-zero value
        assert_array_equal(np.ones(m,), nnz_per_row)

        # check that all values are either 1 or -1
        assert_array_equal(np.ones(m,), np.abs(nnz_values))

    def test_count_sketch_without_matrix(self, A, b, sketch_size):
        """Verifies that all columns are sampled"""
        m = A.shape[0]
        count_sketch = CountSketch(A, b, sketch_size, build_matrix=False)

        # check the dimension
        count_sketch.set_sketch()
        assert m == len(count_sketch.cols)

        # check the sketching
        r = -b.copy()
        SA, SAS, rs = count_sketch.sketch(r)

        assert SA.shape == (sketch_size, A.shape[0])
        assert SAS.shape == (sketch_size, sketch_size)
        assert rs.shape == (sketch_size, 1)

        w = np.random.rand(r.shape[0], 1)
        lmbda = np.random.rand(sketch_size, 1)
        assert w.shape == count_sketch.update_iterate(w, lmbda).shape


def build_subcount_sketch_matrices(
    n_rows, sample_indices, signs, sum_size, sketch_size
):
    """Returns the intermediate matrices for subcount sketch"""
    I_C = np.eye(n_rows)[sample_indices, :]
    D = np.diag(signs)

    subsampling_size = len(sample_indices)  # s should be = tau * k
    Sigma = np.zeros((sketch_size, subsampling_size))
    for i in range(sketch_size):
        Sigma[i, (i * sum_size) : ((i + 1) * sum_size)] = 1

    return I_C, D, Sigma


@pytest.mark.subcountsketch
class TestSubcountSketch:
    def test_subcount_sketch_matrix(self, A, b, sketch_size):
        """
        Verifies the count sketch matrix S contains only
        one non-zero value (-1 or 1) per row
        """
        m = A.shape[0]
        subcount_sketch = SubcountSketch(A, b, sketch_size, sum_size=2)
        assert (
            subcount_sketch.sketch_size * subcount_sketch.sum_size
            == subcount_sketch.subsampling_size
        )

        r = np.arange(0, m).reshape(-1, 1)
        SA, SAS, rs = subcount_sketch.sketch(r)

        # Build corresponding sketch matrix
        I_C, D, Sigma = build_subcount_sketch_matrices(
            m,
            subcount_sketch.sample_indices,
            subcount_sketch.signs,
            subcount_sketch.sum_size,
            subcount_sketch.sketch_size,
        )
        S = (Sigma @ D @ I_C).T
        SA_matrix = (A._A @ S).T
        SAS_matrix = SA_matrix @ S
        rs_matrix = S.T @ r

        assert_allclose(SA, SA_matrix)
        assert_allclose(SAS, SAS_matrix)
        assert_allclose(rs, rs_matrix)

    @pytest.mark.parametrize("m", [34, 657, 1000, 2371])
    def test_subcount_sketch_parameters(self, m):
        """
        Verifies the count sketch sketch, subsampling and sum sizes
        are compatible
        """
        X = np.random.rand(m, m)
        A = AMatrix(alpha=0.1, X=X)
        b = np.random.rand(m, 1)
        for sketch_size in [1, 100, 235, m]:
            for sum_size in [1, 23, m]:
                subcount_sketch = SubcountSketch(A, b, sketch_size, sum_size=sum_size)
                assert (
                    subcount_sketch.sketch_size * subcount_sketch.sum_size
                    == subcount_sketch.subsampling_size
                )

    def test_sum_every(self, A, b, sketch_size):
        """
        Sum every should not work when the number of rows is not a multiple of
        sum_size
        """
        subcount_sketch = SubcountSketch(A, b, sketch_size)

        M = np.random.rand(10, 5)
        with pytest.raises(ValueError):
            subcount_sketch._sum_every_rows(M, 7)


@pytest.mark.hadamard
class TestHadamardSketch:
    def test_hadamard_sketch_matrix(self, A, b, sketch_size):
        """Verifies the Hadamard sketch matrix S does the same transform as when
        no sketch matrix is built.
        """
        # Without sketch matrix
        hadamard_sketch = HadamardSketch(A, b, sketch_size, build_matrix=False)
        r = -b.copy()
        np.random.seed(0)  # fixing seed
        SA, SAS, rs = hadamard_sketch.sketch(r)

        # With sketch matrix S
        hadamard_sketch = HadamardSketch(A, b, sketch_size, build_matrix=True)
        r = -b.copy()
        np.random.seed(0)  # same seed
        SA_matrix, SAS_matrix, rs_matrix = hadamard_sketch.sketch(r)

        assert_allclose(SA, SA_matrix)
        assert_allclose(SAS, SAS_matrix)
        assert_allclose(rs, rs_matrix)
