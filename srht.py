"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from scipy import sparse


def subsample_fwht(A, ind=None):
    """
    Use Fast Walsh-Hadamard Transform to apply
    the Subsample Hadamard transform to matrix A,
    with subsampling indices in ind.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse array
        Input matrix to transform.
    ind : list or numpy.ndarray
        Array of row indices to sample. No sampling by default (if None).

    Returns
    -------
    B : numpy.ndarray
        Hadamard transform of A. Output has the same shape as input.
    """
    # Apply the Hadamard transform
    B = fwht(A)

    if ind is None:
        return B  # no subsampling
    elif isinstance(ind, np.ndarray) or isinstance(ind, list):
        if A.ndim == 1:
            return B[ind]
        elif A.shape[0] == 1:
            return B[:, ind]
        else:
            return B[ind, :]
    else:
        raise ValueError(
            "ind must be a numpy array containing the subsampling indices."
        )


def fwht(A):
    """
    Fast Walsh–Hadamard Transform of each column of matrix A.

    Equivalent to the multiplication on the left of A
    by the Hadamard matrix of order A.shape[0].
    No normalization factor.
    The output has is dense and has the same shape as the input.

    If the input is a row vector of shape (1, m).
    The function applies the Hadamard transform to
    the input vector and not the identity.

    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse array
        Input matrix to transform.

    Returns
    -------
    B : numpy.ndarray
        Hadamard transform of A. Output has the same shape as input.
    """
    if not np.log2(A.shape[0]).is_integer():
        raise ValueError("Size of the sequence must be a power of 2.")

    if sparse.issparse(A):
        B = A.toarray()
    else:
        B = A.copy()

    # Reshape if 2D row vector: (1, n) to (n,)
    if A.shape[0] == 1:
        B = B.reshape(-1,)

    h = 1
    while h < B.shape[0]:
        for i in range(0, B.shape[0], h * 2):
            for j in range(i, i + h):
                x = B[j].copy()
                y = B[j + h].copy()

                B[j] = x + y
                B[j + h] = x - y
        h *= 2

    # Reshape back (n,) to row vector (1, n)
    if A.shape[0] == 1:
        B = B.reshape(1, -1)
    return B


# endregion

# SRHT tools
def next_power_of_two(x):
    """
    Returns the closest power of two greater or equal
    than the input integer
    """
    return 2 ** (int(x) - 1).bit_length()


if __name__ == "__main__":
    # Testing FWHT on vectors
    b = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    print("b:", b)
    b_transform = np.array(
        [4, 2, 0, -2, 0, 2, 0, 2]
    )  # ground truth taken from Wikipedia

    H = fwht(np.eye(len(b)))
    assert all(b_transform == fwht(b))
    assert all(b_transform == H @ b)
    print("fwht(b):", fwht(b))
    print("subsample_fwht(b):", subsample_fwht(b))
    print("subsample_fwht(b):", subsample_fwht(b, np.array([0, 1, 3])))

    # Row vector (1, m)
    b = b.reshape(1, -1)
    print("b:", b)
    print("fwht(b):", fwht(b))
    print("subsample_fwht(b):", subsample_fwht(b))
    print("subsample_fwht(b):", subsample_fwht(b, np.array([0, 1, 3])), "\n")

    # Column vector (m, 1)
    b = b.reshape(-1, 1)
    print("b:\n", b)
    print("fwht(b):\n", fwht(b))
    print("subsample_fwht(b):\n", subsample_fwht(b))
    print("subsample_fwht(b):\n", subsample_fwht(b, np.array([0, 1, 3])))
