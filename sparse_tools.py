"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from scipy import sparse

# Sparse matrix tools
# Set non-zero values of a given row to an input value

# For CSR matrices
def csr_row_set_nz_to_val(A, row, value=0):
    """
    Set all nonzero elements of a CSR matrix A
    (elements currently in the sparsity pattern)
    to the given value. Useful to set to 0 mostly.

    In-place function.
    https://stackoverflow.com/a/12130287/9978618
    """
    if not isinstance(A, sparse.csr_matrix):
        raise ValueError("Matrix given must be of CSR format.")
    if value == 0:
        A.data = np.delete(
            A.data, range(A.indptr[row], A.indptr[row + 1])
        )  # drop nnz values
        A.indices = np.delete(
            A.indices, range(A.indptr[row], A.indptr[row + 1])
        )  # drop nnz column indices
        A.indptr[(row + 1) :] = A.indptr[(row + 1) :] - (
            A.indptr[row + 1] - A.indptr[row]
        )
    else:
        A.data[
            A.indptr[row] : A.indptr[row + 1]
        ] = value  # replace nnz values by another nnz value


# Padding
def pad_with_zeros(A, n, side="bottom"):
    """Pad the bottom of a matrix with n rows of zeros"""
    if sparse.issparse(A):
        if side == "top":
            return sparse.vstack([sparse.csr_matrix((n, A.shape[1])), A])
        elif side == "bottom":
            return sparse.vstack([A, sparse.csr_matrix((n, A.shape[1]))])
        elif side == "left":
            return sparse.hstack([sparse.csr_matrix((A.shape[0], n)), A])
        elif side == "right":
            return sparse.hstack([A, sparse.csr_matrix((A.shape[0], n))])
    else:
        if A.ndim == 1:
            return np.pad(A, (0, n), mode="constant")
        else:
            if side == "top":
                return np.pad(A, [(n, 0), (0, 0)], mode="constant")
            elif side == "bottom":
                return np.pad(A, [(0, n), (0, 0)], mode="constant")
            elif side == "left":
                return np.pad(A, [(0, 0), (n, 0)], mode="constant")
            elif side == "right":
                return np.pad(A, [(0, 0), (0, n)], mode="constant")


# Algebraic operations
def row_wise_mult(A, d):
    """Multiply rows of A by the values contained in the 1D array d.

    If the input is sparse, this function creates a diagonal matrix D,
    which diagonal is d, and return D @ A.
    https://stackoverflow.com/a/22934388/9978618
    """
    if sparse.issparse(A):
        return A.T.multiply(d).T  # sparse array
    else:
        return np.multiply(A, d[:, np.newaxis])  # numpy array
