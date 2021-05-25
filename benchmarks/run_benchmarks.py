"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import click
import os
import numpy as np
from scipy.sparse import random as sprandom

# Ignoring th following sparse warning
# >>> /home/nidham/phd/RidgeSketch/ridge_sketch/ridgesketch-env/lib/python3.7/site-packages/scipy/sparse/_index.py:122: SparseEfficiencyWarning: Changing the sparsity structure of a csr_matrix is expensive. lil_matrix is more efficient.
# >>> self._set_arrayXarray_sparse(i, j, x)
# import warnings
# warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

from timeit import default_timer
from collections import defaultdict
import pandas as pd
from tqdm import tqdm
import itertools
import collections
from benchmarks import plot, benchmark_configs
import shutil


# import datasets
from datasets import data_loaders
from datasets.data_loaders import DATASET_CLASSES
from ridge_sketch import RidgeSketch
from kernel_ridge_sketch import KernelRidgeSketch


class Benchmarks:
    """Class to run and store benchmarks.

    Args:
        config (benchmark_config.Config): contains configs for benchmarks
    """

    def __init__(self, config, random_state=0):
        self.config = config
        self.problems = config.problems
        self.operator_modes = config.operator_modes
        self.algo_modes = config.algo_modes
        self.accel_params = config.accel_params
        self.solvers = config.solvers
        self.sparse_formats = config.sparse_formats
        self.sketch_sizes = config.sketch_sizes
        self.kernel_parameters = config.kernel_parameters
        self.random_state = random_state
        np.random.seed(seed=random_state)

        density = config.density  # make a for loop here
        larger_dimension, smaller_dimension = (
            config.larger_dimension,
            config.smaller_dimension,
        )

        # Setting data to sparse random matrices if required
        if "primal_random" in self.problems:
            if "csc" in self.sparse_formats:
                self.X_primal_csc_random = sprandom(
                    larger_dimension, smaller_dimension, density=density, format="csc",
                )
            if "csr" in self.sparse_formats:
                self.X_primal_csr_random = sprandom(
                    larger_dimension, smaller_dimension, density=density, format="csr"
                )
            if "dense" in self.sparse_formats:
                self.X_primal_dense_random = np.random.rand(
                    larger_dimension, smaller_dimension
                )
            self.y_primal_random = np.random.rand(larger_dimension, 1)
        if "dual_random" in self.problems:
            if "csc" in self.sparse_formats:
                self.X_dual_csc_random = sprandom(
                    smaller_dimension, larger_dimension, density=density, format="csc"
                )
            if "csr" in self.sparse_formats:
                self.X_dual_csr_random = sprandom(
                    smaller_dimension, larger_dimension, density=density, format="csr"
                )
            if "dense" in self.sparse_formats:
                self.X_dual_dense_random = np.random.rand(
                    smaller_dimension, larger_dimension
                )
            self.y_dual_random = np.random.rand(smaller_dimension, 1)

        self.times_df, self.residual_norms_df = None, None

    def compute_fit_time_and_residual(
        self,
        X,
        y,
        problem,
        sparse_format,
        solver,
        sketch_size,
        operator_mode,
        algo_mode="auto",
        accel_param=(0.0, 0.0),
        n_repetitions=1,
    ):
        """Repeats model fit for n_repetitions.
        Returns quartile 1, 3 and median time taken
        """
        times = []
        residual_norms = []
        print(
            f"\n{problem} / {sparse_format} / op = {operator_mode} / algo_mode = {algo_mode} /"
            f" {solver} / sketch_size = {sketch_size} / accel_param = {accel_param}\n"
        )
        for _ in range(n_repetitions):
            if self.config.kernel is None:
                model = RidgeSketch(
                    solver=solver,
                    tol=self.config.tolerance,
                    max_iter=self.config.max_iterations,
                    operator_mode=operator_mode,
                    algo_mode=algo_mode,
                    accel_mu=accel_param[0],
                    accel_nu=accel_param[1],
                    alpha=self.config.alpha,
                    verbose=1,
                    sketch_size=sketch_size,
                    random_state=self.random_state,
                )
            else:
                model = KernelRidgeSketch(
                    solver=solver,
                    tol=self.config.tolerance,
                    max_iter=self.config.max_iterations,
                    alpha=self.config.alpha,
                    algo_mode=algo_mode,
                    accel_mu=accel_param[0],
                    accel_nu=accel_param[1],
                    verbose=1,
                    sketch_size=sketch_size,
                    random_state=self.random_state,
                    kernel=self.config.kernel,
                    kernel_sigma=self.kernel_parameters[0],
                    kernel_nu=self.kernel_parameters[1],
                )
            start = default_timer()
            model.fit(X, y)
            times.append(default_timer() - start)
            residual_norms.append(model.residual_norms)

        times_distribution = compute_distribution(times)
        residual_norms = pad_residual_norms(residual_norms)
        residual_norms_distributions = compute_distribution(residual_norms)
        return times_distribution, residual_norms_distributions

    def get_config_combinations(self):
        non_accel_algo_modes = {"auto", "mom"}
        algo_modes_set = set(self.algo_modes)
        accel_algo_mode = algo_modes_set - non_accel_algo_modes

        non_sketch_solvers = {"cg", "direct"}
        solvers_set = set(self.solvers)
        sketch_solvers = solvers_set - non_sketch_solvers

        combinations = list(
            itertools.product(
                self.sparse_formats,
                self.problems,
                sketch_solvers,
                self.sketch_sizes,
                self.operator_modes,
                accel_algo_mode,
                self.accel_params,
            )
        )

        # Adding runs for non accelerated modes
        for algo_mode in non_accel_algo_modes:
            if algo_mode in algo_modes_set:
                combinations.extend(
                    list(
                        itertools.product(
                            self.sparse_formats,
                            self.problems,
                            sketch_solvers,
                            self.sketch_sizes,
                            self.operator_modes,
                            (algo_mode,),
                            ((0.0, 0.0),),
                        )
                    )
                )

        # Only run cg and direct once per sketch size, op mode
        # and acceleration
        for solver in non_sketch_solvers:
            if solver in solvers_set:
                combinations.extend(
                    list(
                        itertools.product(
                            self.sparse_formats,
                            self.problems,
                            [solver],
                            [self.sketch_sizes[0]],
                            [self.operator_modes[0]],
                            [self.algo_modes[0]],
                            [self.accel_params[0]],
                        )
                    )
                )
        return combinations

    def run(self, n_repetitions=1, verbose=False):
        """Returns pandas dataframe with benchmarks"""
        benchmark_times = defaultdict(list)
        benchmark_residual_norms = {}

        for (
            sparse_format,
            problem,
            solver,
            sketch_size,
            operator_mode,
            algo_mode,
            accel_param,
        ) in tqdm(self.get_config_combinations()):

            is_small = False
            if problem.endswith("_small"):
                dataset_name = problem[:-6]
                is_small = True  # takes only the first 100 rows
            else:
                dataset_name = problem
            dataset_name = "".join([x.capitalize() for x in dataset_name.split("_")])
            dataset_name += "Dataset"

            if dataset_name in DATASET_CLASSES:
                # problem involves a real dataset
                dataset_class = getattr(data_loaders, dataset_name)
                dataset = dataset_class(is_small=is_small)
                # sparse_format = dataset.sparse_format
                X, y = dataset.load_X_y()
            elif problem == "primal_random" and sparse_format == "csc":
                X, y = self.X_primal_csc_random, self.y_primal_random
            elif problem == "primal_random" and sparse_format == "csr":
                X, y = self.X_primal_csr_random, self.y_primal_random
            elif problem == "primal_random" and sparse_format == "dense":
                X, y = self.X_primal_dense_random, self.y_primal_random
            elif problem == "dual_random" and sparse_format == "csc":
                X, y = self.X_dual_csc_random, self.y_dual_random
            elif problem == "dual_random" and sparse_format == "csr":
                X, y = self.X_dual_csr_random, self.y_dual_random
            elif problem == "dual_random" and sparse_format == "dense":
                X, y = self.X_dual_dense_random, self.y_dual_random
            else:
                raise ValueError(f"Problem type {problem} not supported")

            # convert to dense for direct solver and conjugate gradients
            # note conversion is excluded from timing
            (
                times_distribution,
                residual_norms_distribution,
            ) = self.compute_fit_time_and_residual(
                X,
                y,
                problem,
                sparse_format,
                solver,
                sketch_size,
                operator_mode,
                algo_mode,
                accel_param,
                n_repetitions=n_repetitions,
            )
            self.book_keeping(
                problem,
                sparse_format,
                solver,
                sketch_size,
                operator_mode,
                algo_mode,
                accel_param,
                residual_norms_distribution,
                times_distribution,
                benchmark_times,
                benchmark_residual_norms,
            )
        # End big For loop
        times_df = pd.DataFrame(benchmark_times)
        if verbose:
            print(benchmark_residual_norms)
        residual_norms_df = pd.DataFrame.from_dict(
            benchmark_residual_norms, orient="index"
        ).transpose()

        self.times_df = times_df
        self.residual_norms_df = residual_norms_df
        return times_df, residual_norms_df

    def book_keeping(
        self,
        problem,
        sparse_format,
        solver,
        sketch_size,
        operator_mode,
        algo_mode,
        accel_param,
        residual_norms_distribution,
        times_distribution,
        benchmark_times,
        benchmark_residual_norms,
    ):
        description = {
            "problem": problem,
            "sparse_format": sparse_format,
            "solver": solver,
            "sketch_size": sketch_size,
            "operator_mode": operator_mode,
            "algo_mode": algo_mode,
            "accel_param": accel_param,
        }
        description_string = f"{problem} | {sparse_format} | {solver} | sketch_size = {sketch_size} | {'op' if operator_mode else 'no-op'} | algo_mode = {algo_mode} | accel_param = {str(accel_param).replace(', ', '-')}"
        # description_string = f"{problem} | {sparse_format} | {solver} | sketch_size = {sketch_size} | {'op' if operator_mode else 'no-op'} | algo_mode = {algo_mode} | accel_param = X"

        benchmark_time = {
            **description,
            "time (median)": times_distribution.median,
            "time (1st quartile)": times_distribution.q1,
            "time (3rd quartile)": times_distribution.q3,
        }
        benchmark_times = update_times(benchmark_times, benchmark_time)
        benchmark_residual_norm = {
            f"{description_string} | residual_norms (median)": residual_norms_distribution.median,
            f"{description_string} | residual_norms (1st quartile)": residual_norms_distribution.q1,
            f"{description_string} | residual_norms (3rd quartile)": residual_norms_distribution.q3,
        }
        benchmark_residual_norms.update(benchmark_residual_norm)

    def save(self, save_path):
        """Saves benchmark results, plots and configs."""
        full_path = os.path.join(save_path, self.config.name)
        # clear old results
        shutil.rmtree(full_path, ignore_errors=True)
        os.makedirs(full_path)
        self.save_results(full_path)
        self.save_configs(full_path)
        self.save_plots(full_path)

    def save_configs(self, save_path):
        with open(os.path.join(save_path, "configs.txt"), "w") as text_file:
            print(f"Configs: {self.config.__dict__}", file=text_file)

    def save_results(self, save_path):
        self.times_df.to_csv(os.path.join(save_path, "times.csv"))
        self.residual_norms_df.to_csv(os.path.join(save_path, "residual_norms.csv"))

    def save_plots(self, save_path):
        for problem in self.problems:
            # BUG: plot only differentiates on problems not on other inputs
            # self.plot_runtimes(problem, save_path=save_path)
            self.plot_residuals(problem, save_path=save_path)

    def plot_runtimes(self, problem, save_path=None):
        plot.plot_runtimes(self.times_df, problem=problem, save_path=save_path)

    def plot_residuals(self, problem, save_path=None):
        """Plots residuals over time for the given problem.
        Creates one plot per sparse_format
        """
        print()
        run_names = []
        for col in self.residual_norms_df.columns:
            for s in [" (median)", " (1st quartile)", " (3rd quartile)"]:
                col = col.replace(s, "")
            run_names.append(col)
        run_names = list(set(run_names))  # get unique run names

        for sparse_format in self.times_df["sparse_format"].unique():
            filtered_run_names = filter_run_names(run_names, sparse_format, problem)
            formatted_run_names = format_run_names(
                filtered_run_names, sparse_format, problem
            )
            title = f"{sparse_format} {problem} Relative Residual Norms over Iterations"
            fig = plot.plot_runs_over_iterations(
                filtered_run_names,
                formatted_run_names,
                self.residual_norms_df,
                title=title,
            )
            if save_path:
                file_name = title.lower().replace(" ", "_")
                full_path = os.path.join(save_path, f"{file_name}.png")
                fig.write_image(full_path)
            else:
                fig.show()


def filter_run_names(run_names, sparse_format, problem):
    # filter runs for given problem
    problem_run_names = [run_name for run_name in run_names if problem in run_name]
    run_names = [n for n in problem_run_names if sparse_format in n]
    return run_names


def format_run_names(run_names, sparse_format, problem):
    """Selects and formats run names based on sparse format and problem"""
    formatted_run_names = ["|".join(r.split("|")[2:-1]).strip() for r in run_names]

    # remove sketch size and other properties from cg and direct
    for i, run_name in enumerate(run_names):
        if "direct" in run_name:
            formatted_run_names[i] = "direct"
        elif "cg" in run_name:
            formatted_run_names[i] = "cg"

    return formatted_run_names


Distribution = collections.namedtuple("Distribution", "q1 median q3")


def pad_residual_norms(residual_norms):
    """Pads relative residual norm arrays with different lengths with 0.
    This aligns relative residual norms of various lengths form different runs
    """
    padded_iterator = itertools.zip_longest(*residual_norms, fillvalue=0)
    return np.array(list(padded_iterator)).T


def compute_distribution(values):
    """Computes 1st, 3rd quartiles and median.

    Args:
        values (list): containing values from repeated benchmark runs.
    Returns: Distribution named tuple
    """
    values_array = np.asarray(values)
    q1, median, q3 = np.quantile(np.asarray(values_array), [0.25, 0.5, 0.75], axis=0)
    distribution = Distribution(q1, median, q3)
    return distribution


def update_times(times, benchmark):
    """Updates times dict with new benchmark values"""
    for key, val in benchmark.items():
        times[key].append(val)
    return times


def get_configs(name):
    """Returns config from benchmark_configs"""
    configs = getattr(benchmark_configs, name)
    return configs


@click.command()
@click.argument("config_name")
@click.option(
    "--folder",
    default=None,
    type=click.Path(),
    help="folder path where results are saved",
)
@click.option(
    "--n-repetitions", default=1, type=int, help="number of times to rerun benchmarks",
)
@click.option("--save/--no-save", default=True)
def run(config_name, folder, n_repetitions, save):
    if folder is None:
        folder = os.path.join(os.getcwd(), "benchmark_results")

    configs = get_configs(config_name)
    print("Configs:\n", configs.__dict__, "\n")
    benchmarks = Benchmarks(configs)

    benchmarks.run(n_repetitions=n_repetitions)
    if save:
        benchmarks.save(folder)


if __name__ == "__main__":
    run()


# Example of how to run from terminal:

# python3 benchmarks.py small
