"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np

from datasets.generate_controlled_data import generate_data_cd
from experiments.plot_experiments import plot_iterations_v_epsilon
from ridge_sketch import RidgeSketch


ETA_CST = 0.999


def compute_cd_theory(A, residual_norm, tol):
    """Computes theoretical complexity of Coordinate Descent without momentum.

    Args:
        A (np.array): matrix generate from data X
        residual_norm (float): residual norm of coordinate descent solution
        tol (float): tolerance used in solver.

    Returns: scalar for the theoretical complexity.
    """
    eigenvalues = np.sort(np.linalg.eigvals(A))
    max_eigenvalue, min_eigenvalue = eigenvalues[-1], eigenvalues[0]
    log_term = np.log(max_eigenvalue / tol)
    result = residual_norm * np.trace(A) / min_eigenvalue * log_term
    return result


def compute_cd_momentum_theory(A, residual_norm, tol):
    """Computes theoretical complexity of Coordinate Descent with momentum

    Args:
        A (np.array): matrix generate from data X
        residual_norm (float): residual norm of coordinate descent solution
        tol (float): tolerance used in solver.

    Returns: scalar for the theoretical complexity.
    """
    return 2.0 * residual_norm * np.trace(A) / tol


def plot_empirical_complexities(
    min_eigenvalue=1e-3,
    n_samples=1000,
    n_features=300,
    tolerance_range=[1e-4, 1e-2],
    num_points=10,
    eigen_scale=1000.0,
    fig=None,
):
    # tolerances = np.linspace(tolerance_range[0], tolerance_range[-1], num=num_points)
    tolerances = 10.0 ** np.linspace(
        np.log10(tolerance_range[0]), np.log10(tolerance_range[1]), num=num_points
    )
    # tolerances = 10. ** np.arange(-6, -2)
    # eigen_scale = 1e8
    # eigen_scale = 1000 * max(tolerances) / min_eigenvalue
    X, y, lmbda = generate_data_cd(
        n_samples, n_features, min_eigenvalue, eigen_scale=eigen_scale,
    )
    # print(f"min_eigenvalue = {min_eigenvalue:.2e}")
    # print(f"eigen_scale = {eigen_scale:.2e}")

    # print("A:")
    # A = X.T @ X + lmbda * np.eye(n_features)
    A = X.T @ X

    print(f"toleraces = {tolerances}")

    print(f"lambda_min (A) = {np.sort(np.linalg.eigvals(A))[0]}")
    print(f"lambda_max (A) = {np.sort(np.linalg.eigvals(A))[-1]}")
    print(f"kappa (A) = {np.linalg.cond(A):.2e}")
    print(f"trace (A) = {np.trace(A)}\n")

    iterations_required = []
    iterations_required_momentum = []

    for tol in tolerances:
        np.random.seed(0)
        ridge_solver = RidgeSketch(
            algo_mode="auto",
            tol=tol,
            max_iter=200000,
            solver="coordinate descent",
            random_state=0,
        )
        ridge_solver.fit(X, y)
        iterations_required.append(ridge_solver.iterations)

        np.random.seed(0)
        ridge_solver_momentum = RidgeSketch(
            algo_mode="mom",
            use_heuristic=False,
            mom_eta=ETA_CST,
            tol=tol,
            max_iter=500000,
            solver="coordinate descent",
            random_state=0,
        )
        ridge_solver_momentum.fit(X, y)
        iterations_required_momentum.append(ridge_solver_momentum.iterations)

    y_values = [iterations_required, iterations_required_momentum]
    labels = ["CD Empirical", "CD + Momentum Empirical"]
    fig = plot_iterations_v_epsilon(
        y_values, tolerances, labels, line_type="lines+markers", fig=fig
    )
    return fig


def plot_theoretical_complexities(
    min_eigenvalue=1e-3,
    n_samples=1000,
    n_features=300,
    tolerance_range=[1e-4, 1e-2],
    num_points=50,
    fig=None,
):
    """Plots theoretical complexity of cd with and without momentum.
    """
    # lmbda is the regularization parameter
    tolerances = np.linspace(tolerance_range[0], tolerance_range[-1], num=num_points)
    X, y, lmbda = generate_data_cd(
        n_samples,
        n_features,
        min_eigenvalue,
        eigen_scale=1000 * max(tolerances) / min_eigenvalue,
    )
    A = X.T @ X + lmbda * np.eye(n_features)

    cd_theory_complexities = []
    cd_theory_momentum_complexities = []

    for tol in tolerances:
        residual_norm = 1.0
        cd_theory = compute_cd_theory(A, residual_norm, tol)
        cd_theory_momentum = compute_cd_momentum_theory(A, residual_norm, tol)

        cd_theory_complexities.append(cd_theory)
        cd_theory_momentum_complexities.append(cd_theory_momentum)

    y_values = [cd_theory_complexities, cd_theory_momentum_complexities]
    labels = ["CD Theory", "CD + Momentum Theory"]
    fig = plot_iterations_v_epsilon(
        y_values, tolerances, labels, line_type="lines", fig=fig
    )
    return fig


# def main(min_eigenvalues=[1e-10, 1e-6, 1.0]):
#     """Plots theoretical and empirical complexities"""
def main():
    """BLUE ZONE : CD+MOM better than CD"""
    m = 1000
    min_eigenvalues = [1e-16]
    eigen_scale = 1e30
    tolerance_range = [1e-5, 1e-1]
    zone = "blue"
    # def main():
    #     """RED ZONE : CD better than CD+MOM"""
    #     m = 100
    #     min_eigenvalues = [1e-1]
    #     eigen_scale = 20
    #     tolerance_range = [1e-5, 1e-1]
    #     zone = "red"
    print(f"ETA_CST = {ETA_CST}")

    for min_eigenvalue in min_eigenvalues:
        # fig = plot_theoretical_complexities(
        #     min_eigenvalue=min_eigenvalue,
        #     tolerance_range=tolerance_range,
        # )
        # fig.write_image(f"experiments/results/9_1/theoretical-min-eigenval-{min_eigenvalue:.2e}.pdf")
        fig = plot_empirical_complexities(
            min_eigenvalue=min_eigenvalue,
            tolerance_range=tolerance_range,
            num_points=10,
            n_samples=m,
            n_features=m,
            eigen_scale=eigen_scale,
        )
        fig.show()
        # fig.write_image(f"experiments/results/9_1/empirical-switch_eta-min-eigenval_{min_eigenvalue:.2e}-eigen_scale_{eigen_scale:.2e}-{m}_m-{zone}_zone.pdf")
        fig.write_image(
            f"experiments/results/9_1/empirical-eta_cst_{ETA_CST}_-min-eigenval_{min_eigenvalue:.2e}-eigen_scale_{eigen_scale:.2e}-{m}_m-{zone}_zone.pdf"
        )


if __name__ == "__main__":
    main()
