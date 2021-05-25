"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
from warnings import warn


def conjugate_grad(A, b, tol, max_iter, verbose=0):
    """
    Solve a linear equation Ax = b with conjugate gradient method

    Parameters
    ----------
    A : AMatrix or AOperator
        Positive semi-definite (symmetric) matrix
    b : 1d numpy.array
        Right-hand side vector of the linear system to solve
    tol : float
        Error tolerance for solution

    Returns
    -------
    x : 1d numpy.array
        Solution of the linear system
    residuals : list of floats
        List of the normalized residuals r_k / r_0
        with $r_k = \norm{A x^k - b}_2$
    iterations : int
        Number of iterations to reach the desired tolerance
    """
    residuals = []
    m = len(b)
    x = np.zeros(b.shape)
    r = -b.copy()
    p = b.copy()
    r_k_norm = np.dot(r.T, r)
    r_sqrt_initial = np.sqrt(r_k_norm)
    r_sqrt = r_sqrt_initial
    residuals.append(1.0)

    max_iter_cg = 2 * m
    if max_iter > max_iter_cg:
        max_iter = max_iter_cg
        warn(
            f"Maximum iterations for CG must be at most 2 * m. Reset max_iter to 2 * m = {max_iter_cg}."
        )

    iterations = max_iter - 1
    for i in range(max_iter):
        if verbose and (i % int(max_iter_cg / 5) == 0):
            print(
                f"iter:{i:^8d} |  rel res norm: {(r_sqrt[0][0] / r_sqrt_initial[0][0]):.2e}"
            )

        Ap = A @ p
        alpha = r_k_norm / np.dot(p.T, Ap)
        x += alpha * p
        r += alpha * Ap
        r_k_plus_1_norm = np.dot(r.T, r)
        beta = r_k_plus_1_norm / r_k_norm
        r_k_norm = r_k_plus_1_norm
        r_sqrt = np.sqrt(r_k_plus_1_norm)
        residuals.append(r_sqrt[0][0] / r_sqrt_initial[0][0])
        if r_sqrt[0][0] / r_sqrt_initial < tol:
            print("cg solver converged on iter ", i)
            break
        p = beta * p - r
    iterations = i
    return x, residuals, iterations
