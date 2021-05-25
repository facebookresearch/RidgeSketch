"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Experiments comparing all sketches methods for increasing momentum
(without heuristic)

Section 9.3 in the LaTex

Sketch size: m/4

Run above with different
- sketch solvers (subcount, count, subsample, gaussian, hadamard)

Output:
- residual versus time
- residual versus iterations


For example, run from the ridge_sketch folder
```
$ python experiments/different_sketches.py boston -k False -h true -r 1e-6 -t 1e-4 -m 2000 -n 3
```
"""

import click
import os
import shutil
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from timeit import default_timer

from datasets.data_loaders import (
    BostonDataset,
    CaliforniaHousingDataset,
    YearPredictionDataset,
    Rcv1Dataset,
)
from ridge_sketch import RidgeSketch
from kernel_ridge_sketch import KernelRidgeSketch
from benchmarks import compute_distribution, pad_residual_norms, update_times
from experiments.plot_experiments import plot_runs_over_iterations, plot_runs_over_time


class ExperimentSketches:
    """
    Class to run the experiments for different sketches
    """

    def __init__(
        self,
        dataset_name,
        X,
        y,
        regularizer,
        is_kernel,
        use_heuristic,
        tolerance,
        max_iter,
        solvers,
        sketch_size,
        sketch_size_formula,
        n_repetitions,
    ):
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.regularizer = regularizer
        self.is_kernel = is_kernel
        self.use_heuristic = use_heuristic
        self.tolerance = tolerance
        self.max_iter = max_iter

        # List of sketch solvers
        self.solvers = solvers

        # Only one sketch size here
        self.sketch_size = sketch_size
        self.sketch_size_formula = sketch_size_formula

        # For error areas 1st/3rd quartiles
        self.n_repetitions = n_repetitions

        # df of times of all runs of the experiment
        self.times_df = None

        # df of relative residual norms of all runs of the experiment
        self.residual_norms_df = None

    def run_full_exp(self, verbose=True):
        algo_mode = "mom"
        mom_beta = None
        step_size = None
        mom_eta = 0.995  # increasing momentum parameter

        # dict of the outputs
        times = defaultdict(list)
        residual_norms = {}
        counter = 1
        n_settings = len(self.solvers)
        for solver in self.solvers:
            if solver == "coordinate descent":
                sketch_size = 1
                sketch_size_formula = "1"
            else:
                sketch_size = self.sketch_size
                sketch_size_formula = self.sketch_size_formula

            run_name = self.dataset_name  # solver
            print(f"----> Setting {counter} over {n_settings} : {solver}")
            (
                times_distribution,
                residual_norms_distribution,
            ) = self.compute_fit_time_and_residual(
                solver, sketch_size, algo_mode, step_size, mom_beta, mom_eta,
            )

            # Storing the results
            self.book_keeping(
                run_name,
                solver,
                sketch_size_formula,
                sketch_size,
                times_distribution,
                residual_norms_distribution,
                times,
                residual_norms,
            )
            counter += 1
            print("\n")

        # converting the outputs to dataframes
        times_df = pd.DataFrame(times)
        if verbose:
            print(f"residual_norms:\n{residual_norms}")
        residual_norms_df = pd.DataFrame.from_dict(
            residual_norms, orient="index"
        ).transpose()

        self.times_df = times_df
        self.residual_norms_df = residual_norms_df

        return times_df, residual_norms_df

    def compute_fit_time_and_residual(
        self, solver, sketch_size, algo_mode, step_size, mom_beta, mom_eta
    ):
        """
        Repeats model fit for n_repetitions.

        Returns quartile 1, 3 and median time taken
        """
        times = []
        residual_norms = []
        for repetition_idx in range(self.n_repetitions):
            print(
                f"--------> Repetition {repetition_idx+1} / " f"{self.n_repetitions}",
                end="\r",
            )

            random_state = repetition_idx
            np.random.seed(seed=random_state)
            model = load_model(
                solver,
                sketch_size,
                algo_mode,
                step_size,
                mom_beta,
                mom_eta,
                self.regularizer,
                self.is_kernel,
                self.use_heuristic,
                self.tolerance,
                self.max_iter,
                random_state,
            )
            start = default_timer()
            model.fit(self.X, self.y)
            times.append(default_timer() - start)
            residual_norms.append(model.residual_norms)

        times_distribution = compute_distribution(times)
        residual_norms = pad_residual_norms(residual_norms)
        residual_norms_distribution = compute_distribution(residual_norms)

        return times_distribution, residual_norms_distribution

    def book_keeping(
        self,
        run_name,
        solver,
        sketch_size_formula,
        sketch_size,
        times_distribution,
        residual_norms_distribution,
        run_times,
        run_residual_norms,
    ):
        """Store time and relative residual norms distribution for each solver"""
        description = {
            "run_name": run_name,
            "solver": solver,
            "sketch_size_formula": sketch_size_formula,
            "sketch_size": sketch_size,
        }
        description_string = (
            f"{run_name} | {solver} | sketch_size = "
            f"{sketch_size_formula} = {sketch_size}"
        )

        run_time = {
            **description,
            "time (median)": times_distribution.median,
            "time (1st quartile)": times_distribution.q1,
            "time (3rd quartile)": times_distribution.q3,
        }
        run_times = update_times(run_times, run_time)

        run_residual_norm = {
            f"{description_string} | residual_norms (median)": residual_norms_distribution.median,
            f"{description_string} | residual_norms (1st quartile)": residual_norms_distribution.q1,
            f"{description_string} | residual_norms (3rd quartile)": residual_norms_distribution.q3,
        }
        run_residual_norms.update(run_residual_norm)

    def save(self, exp_name, save_path):
        full_path = os.path.join(save_path, exp_name)
        # clear old results
        shutil.rmtree(full_path, ignore_errors=True)
        os.makedirs(full_path)
        self.save_settings(exp_name, full_path)
        self.save_results(exp_name, full_path)
        self.save_plots(exp_name, full_path)

    def save_settings(self, exp_name, save_path):
        with open(
            os.path.join(save_path, exp_name + "_settings.txt"), "w",
        ) as text_file:
            settings_dict = {
                "dataset_name": self.dataset_name,
                "regularizer": self.regularizer,
                "tolerance": self.tolerance,
                "is_kernel": self.is_kernel,
                "n_repetitions": self.n_repetitions,
                "solvers": self.solvers,
                "sketch_size": self.sketch_size,
                "sketch_size_formula": self.sketch_size_formula,
            }
            print(f"Settings: {settings_dict}", file=text_file)

    def save_results(self, exp_name, save_path):
        self.times_df.to_csv(os.path.join(save_path, exp_name + "_times.csv"))
        self.residual_norms_df.to_csv(
            os.path.join(save_path, exp_name + "_residual_norms.csv")
        )

    def save_plots(self, exp_name, save_path):
        self.plot_residuals(
            exp_name, save_path=save_path,
        )

    def plot_residuals(self, exp_name, save_path=None):
        """
        Plots residuals over iterations and time.

        Creates one plot with all sketches.
        """
        # https://stackoverflow.com/a/6117124/9978618
        rep = {" (median)": "", " (1st quartile)": "", " (3rd quartile)": ""}
        rep = dict((re.escape(k), v) for k, v in rep.items())
        pattern = re.compile("|".join(rep.keys()))

        run_names = {
            # col for col in self.residual_norms_df.columns
            pattern.sub(lambda m: rep[re.escape(m.group(0))], col)
            for col in self.residual_norms_df.columns
        }
        run_names = list(run_names)
        run_names.sort()
        # print("\n~~~~ run_names: ")
        # for n in run_names:
        #     print(n)
        # print()

        formatted_run_names = format_run_names_exp(run_names)
        # print("~~~~ formatted_run_names: ")
        # for n in formatted_run_names:
        #     print(n)
        # print("\n")

        # Iteration plot
        fig_iter = plot_runs_over_iterations(
            run_names, formatted_run_names, self.residual_norms_df,
        )

        if save_path:
            file_name = exp_name
            file_name = re.sub("[()]", "", file_name)
            full_path = os.path.join(save_path, f"{file_name}_iterations.pdf")
            fig_iter.write_image(full_path)
        else:
            fig_iter.show()

        # Time plot
        fig_iter = plot_runs_over_time(
            run_names, formatted_run_names, self.times_df, self.residual_norms_df,
        )

        if save_path:
            file_name = exp_name
            file_name = re.sub("[()]", "", file_name)
            full_path = os.path.join(save_path, f"{file_name}_time.pdf")
            fig_iter.write_image(full_path)
        else:
            fig_iter.show()


def load_model(
    solver,
    sketch_size,
    algo_mode,
    step_size,
    mom_beta,
    mom_eta,
    regularizer,
    is_kernel,
    use_heuristic,
    tolerance,
    max_iter,
    random_state,
):
    if not is_kernel:
        model = RidgeSketch(
            alpha=regularizer,
            fit_intercept=True,
            tol=tolerance,
            solver=solver,
            sketch_size=sketch_size,
            algo_mode=algo_mode,
            step_size=step_size,
            mom_beta=mom_beta,
            mom_eta=mom_eta,
            use_heuristic=use_heuristic,
            max_iter=max_iter,
            operator_mode=False,
            random_state=random_state,
            verbose=0,
        )
    else:
        model = KernelRidgeSketch(
            alpha=regularizer,
            tol=tolerance,
            solver=solver,
            sketch_size=sketch_size,
            algo_mode=algo_mode,
            step_size=step_size,
            mom_beta=mom_beta,
            mom_eta=mom_eta,
            use_heuristic=use_heuristic,
            max_iter=max_iter,
            kernel="RBF",
            kernel_sigma=1.0,
            kernel_nu=0.5,
            random_state=random_state,
            verbose=0,
        )

    return model


def format_run_names_exp(run_names):
    """
    Selects and formats run names based on run name, solver and sketch size
    """
    formatted_run_names = []
    for r in run_names:
        r = r.split(" | ")[1]
        if r == "coordinate descent":
            r = "CD (sketch size = 1)"
        formatted_run_names.append(r)
    return formatted_run_names


@click.command()
@click.argument("dataset_name")
@click.option(
    "-k", "--is-kernel", default=False, type=bool, help="kernel version",
)
@click.option(
    "-h", "--use-heuristic", type=bool, help="use heuristic for increasing momentum",
)
@click.option(
    "-r", "--regularizer", default=1e-6, type=float, help="regularizer",
)
@click.option(
    "-t", "--tolerance", default=1e-3, type=float, help="tolerance to reach",
)
@click.option(
    "-m",
    "--max-iter",
    default=1000,
    type=int,
    help="maximum number of iteration per run",
)
@click.option(
    "-n", "--n-repetitions", default=1, type=int, help="number of runs per setting",
)
def different_sketches(
    dataset_name,
    is_kernel,
    use_heuristic,
    regularizer,
    tolerance,
    max_iter,
    n_repetitions,
):
    # Experiment settings
    save = True

    print(f"DATASET: {dataset_name}")

    # Problem settings
    if dataset_name == "boston":
        dataset = BostonDataset()
    elif dataset_name == "cali":
        dataset = CaliforniaHousingDataset()
    elif dataset_name == "year":
        dataset = YearPredictionDataset()
    elif dataset_name == "rcv1":
        dataset = Rcv1Dataset()

    X, y = dataset.load_X_y()

    # Solvers
    solvers = [
        "subsample",
        # "coordinate descent",
        "gaussian",
        "count",
        "subcount",
        "hadamard",
    ]
    # solvers = ["count"]

    # Code obviously not compatible with
    # ["direct", "cg"]
    # since momentum is not available for these solvers

    # Sketch sizes
    if is_kernel:
        m = X.shape[0]
    else:
        m = min(X.shape)
    print("m = ", m)

    # sketch_size = int(round(m))
    # sketch_size_formula = "m"
    # sketch_size_formula_reformatted = "m"

    sketch_size = int(round(m / 4))
    sketch_size_formula = "m/4"
    sketch_size_formula_reformatted = "m_over_4"

    # sketch_size = int(round(np.sqrt(m)))
    # sketch_size_formula = "sqrt(m)"
    # sketch_size_formula_reformatted = "sqrt_m"

    print(f"sketch size: {sketch_size_formula} = {sketch_size}\n")

    exp_9_3 = ExperimentSketches(
        dataset.name,
        X,
        y,
        regularizer,
        is_kernel,
        use_heuristic,
        tolerance,
        max_iter,
        solvers,
        sketch_size,
        sketch_size_formula,
        n_repetitions,
    )

    exp_9_3.run_full_exp(verbose=False)

    if save:
        exp_name = dataset.name
        if is_kernel:
            exp_name = exp_name + "_kernel"
        exp_name = f"{exp_name}_reg_{regularizer:1.0e}"
        exp_name = f"{exp_name}_tol_{tolerance:1.0e}"
        exp_name = f"{exp_name}_n_rep_{n_repetitions}"
        exp_name = f"{exp_name}_max_iter_{max_iter}"
        exp_name = f"{exp_name}_{sketch_size_formula_reformatted}"

        if use_heuristic:
            exp_name = f"{exp_name}_heuristic"

        folder_path = os.path.join(os.getcwd(), "experiments/results/9_3")
        exp_9_3.save(exp_name, folder_path)

        print("==== Results saved ====\n\n")


if __name__ == "__main__":
    different_sketches()
