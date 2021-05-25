"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np


def generate_controlled_spectrum_data(
    n_samples, n_features, sigma_min, singular_scale,
):
    """
    Generate data with stagged singular values

    Parameters
    ----------
    n_samples : int
        Number of data samples (rows).
    n_features : int
        Number of features (columns).
    sigma_min : float
        Smallest singular value.
    singular_scale : float
        Ratio of the largest singular value over the smallest

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Training data.
        Its singular values are stagged from sigma_min to
        singular_scale*sigma_min
    y : ndarray of shape (n_samples,)
        Array of labels.
    """
    np.random.seed(0)
    if n_samples > n_features:
        sigma = np.linspace(sigma_min, singular_scale * sigma_min, n_features)
    else:
        sigma = np.linspace(sigma_min, singular_scale * sigma_min, n_samples)
    X = np.diag(sigma)

    H = np.random.rand(n_samples, n_features)
    U, _, VT = np.linalg.svd(H, full_matrices=False)
    X = U @ X @ VT

    y = np.random.rand(n_samples)

    return X, y


def generate_data_cd(n_samples, n_features, lmbda_min, eigen_scale=100.0):
    r"""
    Generate data with stagged singular values.

    Training data is such that resulting smallest eigenvalue of the covariance
    matrix for ridge regression matrix A is lmbda_min and its largest is
    lmbda_min * eigen_scale, where A is either
    Primal form:
    $$A = (X^\top X + alpha I)$$

    Dual form:
    $$A = (X X^\top + alpha I)$$

    Used for experiment validating our theory for CD with and without momentum.

    Parameters
    ----------
    n_samples : int
        Number of data samples (rows).
    n_features : int
        Number of features (columns).
    lmbda_min : float
        Smallest eigenvalue value of A (not built here).
    eigen_scale : float, default=10.
        Ratio of the largest singular value of X over the smallest.
        The trace of A increases with this parameter.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Training data.
        Corresponding covariance matrix for ridge regression A has its smallest
        eigenvalue equal to 'lmbda_min'.
    y : ndarray of shape (n_samples,)
        Array of labels.
    lmbda : float
        regularization parameter that does not affect much $\lambda_\min (A)$.
        1000 smaller than the latter.
    """
    np.random.seed(0)
    sigma_min = np.sqrt(lmbda_min)
    X, y = generate_controlled_spectrum_data(
        n_samples, n_features, sigma_min, np.sqrt(eigen_scale),
    )

    lmbda = 0
    return X, y, lmbda


if __name__ == "__main__":
    n_samples, n_features = 100, 10
    lmbda_min = 1e-8
    eigen_scale = 1e20
    print(f"lambda_min INPUT = {lmbda_min}\n")
    X, y, lmbda = generate_data_cd(n_samples, n_features, lmbda_min, eigen_scale,)
    A = X.T @ X + lmbda * np.eye(n_features)
    print(f"lambda_min (A) = {np.sort(np.linalg.eigvals(A))[0]}")
    print(f"lambda_max (A) = {np.sort(np.linalg.eigvals(A))[-1]}\n")
    print(f"kappa (A) = {np.linalg.cond(A):.2e}")
    print(f"trace (A) = {np.trace(A)}")
