"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Side experiment for constant versus heuristic increasing versus no-momentum

Section 9.2 in the LaTex

Regularizers tested are 0, 0.001*trace(A)/m^2 and 100*trace(A)/m^2

Show the effect of the regularization strength on the efficiency on momentum
versions.

For example, run from the ridge_sketch folder
```
$ python experiments/regularization_on_momentum.py boston -k False -t 1e-4 -m 2000 -n 3
```
"""
import click
import os
import shutil
from pathlib import Path
import re
import numpy as np
from scipy.sparse import issparse
from sklearn.utils.extmath import safe_sparse_dot
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
from experiments.plot_experiments import plot_runs_over_iterations


class ExperimentMomentum:
    """
    Class to run the experiments for constant vs increasing vs no-momentum
    """

    def __init__(
        self,
        dataset_name,
        X,
        y,
        regularizer,
        is_kernel,
        tolerance,
        max_iter,
        solvers,
        sketch_sizes,
        sketch_size_formulas,
        n_repetitions,
    ):
        self.dataset_name = dataset_name
        self.X = X
        self.y = y
        self.regularizer = regularizer
        self.is_kernel = is_kernel
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.solvers = solvers

        self.sketch_sizes = sketch_sizes
        self.sketch_size_formulas = sketch_size_formulas
        # check if some values are the same
        self.sketch_sizes, self.sketch_size_formulas = filter_sketch_sizes(
            self.sketch_sizes, self.sketch_size_formulas
        )

        # For error areas 1st/3rd quartiles
        self.n_repetitions = n_repetitions

        # df of times of all runs of the experiment
        self.times_df = None
        # df of relative residual norms of all runs of the experiment
        self.residual_norms_df = None

    def run_full_exp(self, verbose=True):
        # dict of the outputs
        all_times = defaultdict(list)
        all_residual_norms = {}

        # No momentum runs
        print("NO MOMENTUM")
        no_mom_times, no_mom_residual_norms = self.run_no_momentum()
        all_times.update(no_mom_times)
        all_residual_norms.update(no_mom_residual_norms)

        # Constant momentum runs
        print("CONSTANT MOMENTUM")
        cst_mom_times, cst_mom_residual_norms = self.run_constant_momentum()
        all_times.update(cst_mom_times)
        all_residual_norms.update(cst_mom_residual_norms)

        # Increasing momentum runs
        print("INCREASING MOMENTUM")
        inc_mom_times, inc_mom_residual_norms = self.run_increasing_momentum()
        all_times.update(inc_mom_times)
        all_residual_norms.update(inc_mom_residual_norms)

        # converting the outputs to dataframes
        times_df = pd.DataFrame(all_times)
        if verbose:
            print(all_residual_norms)
        residual_norms_df = pd.DataFrame.from_dict(
            all_residual_norms, orient="index"
        ).transpose()

        self.times_df = times_df
        self.residual_norms_df = residual_norms_df

        return times_df, residual_norms_df

    def run_no_momentum(self):
        run_name = "no_mom"
        algo_mode = "auto"  # no momentum
        mom_beta = None
        step_size = 1.0  # unitary step
        mom_eta = None

        no_mom_times = defaultdict(list)
        no_mom_residual_norms = {}
        counter = 1
        n_settings = 3 * len(self.solvers) * len(self.sketch_sizes)
        for solver in self.solvers:
            for i, sketch_size in enumerate(self.sketch_sizes):
                sketch_size_formula = self.sketch_size_formulas[i]
                print(
                    f"----> Setting {counter} over {n_settings} :\n"
                    f"{run_name} / {algo_mode} / {solver} / "
                    f"sketch_size = {sketch_size_formula} = {sketch_size}"
                )
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
                    no_mom_times,
                    no_mom_residual_norms,
                )
                counter += 1
                print("\n")

        return no_mom_times, no_mom_residual_norms

    def run_constant_momentum(self):
        run_name = "cst_mom"  # constant momentum setting
        algo_mode = "mom"
        # REF: N Loizou paper
        mom_beta = 0.5  # constant momentum parameter
        step_size = 1.0  # constant step size TODO: justify
        mom_eta = None

        cst_mom_times = defaultdict(list)
        cst_mom_residual_norms = {}
        counter = 1 + len(self.solvers) * len(self.sketch_sizes)
        n_settings = 3 * len(self.solvers) * len(self.sketch_sizes)
        for solver in self.solvers:
            for i, sketch_size in enumerate(self.sketch_sizes):
                sketch_size_formula = self.sketch_size_formulas[i]
                print(
                    f"----> Setting {counter} over {n_settings} :\n"
                    f"{run_name} / {algo_mode} / {solver} / "
                    f"sketch_size = {sketch_size_formula} = {sketch_size}"
                )
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
                    cst_mom_times,
                    cst_mom_residual_norms,
                )
                counter += 1
                print("\n")

        return cst_mom_times, cst_mom_residual_norms

    def run_increasing_momentum(self):
        run_name = "inc_mom"
        algo_mode = "mom"
        mom_beta = None
        step_size = None
        mom_eta = 0.995  # increasing momentum parameter

        inc_mom_times = defaultdict(list)
        inc_mom_residual_norms = {}
        counter = 1 + 2 * len(self.solvers) * len(self.sketch_sizes)
        n_settings = 3 * len(self.solvers) * len(self.sketch_sizes)
        for solver in self.solvers:
            for i, sketch_size in enumerate(self.sketch_sizes):
                sketch_size_formula = self.sketch_size_formulas[i]
                print(
                    f"----> Setting {counter} over {n_settings} :\n"
                    f"{run_name} / {algo_mode} / {solver} / "
                    f"sketch_size = {sketch_size_formula} = {sketch_size}"
                )
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
                    inc_mom_times,
                    inc_mom_residual_norms,
                )
                counter += 1
                print("\n")

        return inc_mom_times, inc_mom_residual_norms

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
        """
        Store time and relative residual norms distribution for each parametrization.

        run_name :
            - "cst_mom" for constant momentum
            - "inc_mom" for increasing momentum
            - "no_mom" for no momentum
        """
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
        """Saves benchmark results, plots and configs"""
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
                "sketch_sizes": self.sketch_sizes,
                "sketch_size_formulas": self.sketch_size_formulas,
            }
            print(f"Settings: {settings_dict}", file=text_file)

    def save_results(self, exp_name, save_path):
        self.times_df.to_csv(os.path.join(save_path, exp_name + "_times.csv"))
        self.residual_norms_df.to_csv(
            os.path.join(save_path, exp_name + "_residual_norms.csv")
        )

    def save_plots(self, exp_name, save_path):
        for i, sketch_size in enumerate(self.sketch_sizes):
            sketch_size_formula = self.sketch_size_formulas[i]
            for solver in self.solvers:
                self.plot_residuals(
                    solver,
                    sketch_size,
                    sketch_size_formula,
                    exp_name,
                    save_path=save_path,
                )

    def plot_residuals(
        self, solver, sketch_size, sketch_size_formula, exp_name, save_path=None
    ):
        """
        Plots residuals over time.

        Creates one plot per (solver, sketch size) couple,
        for all run types (no mom, constant mom, increasing mom)
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
        # print("\n~~~~ run_names: ")
        # for n in run_names:
        #     print(n)
        # print()

        filtered_run_names = filter_run_names_exp(
            run_names, solver, sketch_size_formula
        )
        # print("~~~~ filtered_run_names: ")
        # for n in filtered_run_names:
        #     print(n)

        # formatted_run_names = filtered_run_names
        # formatted_run_names = ["|".join(r.split("|")[:-1]).strip() for r in run_names]
        formatted_run_names = format_run_names_exp(
            filtered_run_names, solver, sketch_size_formula
        )
        # print("~~~~ formatted_run_names: ")
        # for n in formatted_run_names:
        #     print(n)
        # print("\n")

        # print(f"residual_norms_df.columns :\n{self.residual_norms_df.columns}")

        title = f"{self.dataset_name} Relative Residual Norms over Iterations"
        fig = plot_runs_over_iterations(
            filtered_run_names, formatted_run_names, self.residual_norms_df,
        )

        if save_path:
            split_title = title.lower().split(" ")
            file_name = exp_name + "_" + ("_").join(split_title[1:])
            sketch_size_formula = sketch_size_formula.replace("^", "_power_").replace(
                "/", "_over_"
            )
            file_name = "_".join((file_name, solver, sketch_size_formula))
            file_name = file_name.replace(" = ", "_equals_")
            file_name = file_name.replace("sqrt(m)", "sqrt_m")
            file_name = re.sub("[()]", "", file_name)
            full_path = os.path.join(save_path, f"{file_name}.pdf")
            fig.write_image(full_path)
        else:
            fig.show()


def load_model(
    solver,
    sketch_size,
    algo_mode,
    step_size,
    mom_beta,
    mom_eta,
    regularizer,
    is_kernel,
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
            use_heuristic=True,
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
            use_heuristic=True,
            max_iter=max_iter,
            kernel="RBF",
            kernel_sigma=1.0,
            kernel_nu=0.5,
            random_state=random_state,
            verbose=0,
        )

    return model


def filter_run_names_exp(run_names, solver, sketch_size_formula):
    """Filter runs for given solver and sketch_size_formula"""
    solver_run_names = [
        run_name for run_name in run_names if solver == run_name.split(" | ")[1]
    ]
    run_names = [n for n in solver_run_names if sketch_size_formula in n]

    # To alwayshave in the same order:
    # no momentum, then constant momentum, then increasing momentum
    run_names.sort(reverse=True)
    run_names = [run_names[0], run_names[2], run_names[1]]
    return run_names


def format_run_names_exp(run_names, solver, sketch_size_formula):
    """
    Selects and formats run names based on run name, solver and sketch size
    """
    # formatted_run_names = ["|".join(r.split("|")[:-1]).strip() for r in run_names]
    formatted_run_names = []
    for r in run_names:
        mom_version = r.split(" |")[0]
        if mom_version == "inc_mom":
            formatted_run_names.append("increasing momentum")
        elif mom_version == "cst_mom":
            formatted_run_names.append("constant momentum")
        elif mom_version == "no_mom":
            formatted_run_names.append("no momentum")
        else:
            raise ValueError("Unknown momentum version")
    return formatted_run_names


def filter_sketch_sizes(sketch_sizes, sketch_size_formulas):
    """Filter duplicate sketch sizes and merge the formulas"""
    filtered_sketch_sizes = sketch_sizes.copy()
    filtered_sketch_size_formulas = sketch_size_formulas.copy()
    if len(sketch_sizes) not in [1, 2, 3]:
        raise ValueError("Works only with 2 or 3 sketch sizes")
    elif len(sketch_sizes) == 2:
        if len(np.unique(filtered_sketch_sizes)) < len(filtered_sketch_sizes):
            if filtered_sketch_sizes[0] == filtered_sketch_sizes[1]:
                filtered_sketch_size_formulas[1] = (
                    filtered_sketch_size_formulas[0]
                    + " = "
                    + filtered_sketch_size_formulas[1]
                )
                del filtered_sketch_sizes[0]
                del filtered_sketch_size_formulas[0]
    elif len(sketch_sizes) == 3:
        if len(np.unique(filtered_sketch_sizes)) < len(filtered_sketch_sizes):
            idx_to_del = []
            if (
                filtered_sketch_sizes[0] == filtered_sketch_sizes[1]
                and filtered_sketch_sizes[1] == filtered_sketch_sizes[2]
            ):
                idx_to_del = [0, 1]
                filtered_sketch_size_formulas[2] = (
                    filtered_sketch_size_formulas[0]
                    + " = "
                    + filtered_sketch_size_formulas[1]
                    + " = "
                    + filtered_sketch_size_formulas[2]
                )
            elif filtered_sketch_sizes[0] == filtered_sketch_sizes[1]:
                idx_to_del = [0]
                filtered_sketch_size_formulas[1] = (
                    filtered_sketch_size_formulas[0]
                    + " = "
                    + filtered_sketch_size_formulas[1]
                )
            elif filtered_sketch_sizes[1] == filtered_sketch_sizes[2]:
                idx_to_del = [1]
                filtered_sketch_size_formulas[2] = (
                    filtered_sketch_size_formulas[1]
                    + " = "
                    + filtered_sketch_size_formulas[2]
                )
            elif filtered_sketch_sizes[0] == filtered_sketch_sizes[2]:
                idx_to_del = [0]
                filtered_sketch_size_formulas[2] = (
                    filtered_sketch_size_formulas[0]
                    + " = "
                    + filtered_sketch_size_formulas[2]
                )

            for index in sorted(idx_to_del, reverse=True):
                del filtered_sketch_sizes[index]
                del filtered_sketch_size_formulas[index]

    return filtered_sketch_sizes, filtered_sketch_size_formulas


@click.command()
@click.argument("dataset_name")
@click.option(
    "-k", "--is-kernel", default=False, type=bool, help="kernel version",
)
@click.option(
    "-t", "--tolerance", default=1e-4, type=float, help="tolerance to reach",
)
@click.option(
    "-m",
    "--max-iter",
    default=1000,
    type=int,
    help="maximum number of iteration per run",
)
@click.option(
    "-n", "--n-repetitions", default=1, type=int,
    help="number of runs per setting",
)
def regularization_on_momentum(
    dataset_name, is_kernel, tolerance, max_iter, n_repetitions
):
    # Experiment settings
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
    n_samples, n_features = X.shape

    # Save path
    save = True
    save_path = os.path.join(
        os.getcwd(),
        "experiments/results/9_2_reg_effect_on_mom",
    )
    pb_name = dataset.name
    if is_kernel:
        pb_name = pb_name + "_kernel"

    full_path = os.path.join(save_path, pb_name)
    Path(full_path).mkdir(parents=True, exist_ok=True)

    # Regularization values
    if is_kernel:
        m = n_samples
    else:
        m = min(n_samples, n_features)
    print("m = ", m)

    if is_kernel:
        A = safe_sparse_dot(X.T, X)
    else:
        if n_features > n_samples:
            # dual system matrix
            A = safe_sparse_dot(X, X.T)
        else:
            # primal system matrix
            A = safe_sparse_dot(X.T, X)

    if issparse(A):
        reg_baseline = A.diagonal().sum() / (m ** 2)
    else:
        reg_baseline = np.trace(A) / (m ** 2)
    reg_vals = [0., 1e-3*reg_baseline, 1e2*reg_baseline]
    reg_header = ["0", "1e-3*baseline", "1e2*baseline"]
    # reg_vals = [0., 1e-6, 1., 1e3]
    # reg_header = ["0", "1e-6", "1", "1e3"]

    reg_vals_df = pd.DataFrame(data=[reg_vals], columns=reg_header)
    reg_vals_df.to_csv(
        os.path.join(full_path, "regularizers.csv")
    )
    # print(f"regularizers: {reg_vals}\n")

    # Solvers
    solvers = ["count"]
    # solvers = ["subsample"]

    # Code obviously not compatible with
    # ["direct", "cg"]
    # since momentum is not available for these solvers

    # Sketch sizes
    # sketch_sizes = [m ** (2 / 3), m / 4, m / 2]
    # sketch_sizes = [int(round(x)) for x in sketch_sizes]
    # sketch_size_formulas = ["m^(2/3)", "m/4", "m/2"]

    # sketch_sizes = [np.sqrt(m), m ** (2 / 3), m / 2]
    # sketch_sizes = [int(round(x)) for x in sketch_sizes]
    # sketch_size_formulas = ["sqrt(m)", "m^(2/3)", "m/2"]

    sketch_sizes = [m / 4]
    sketch_sizes = [int(round(x)) for x in sketch_sizes]
    sketch_size_formulas = ["m/4"]

    # sketch_sizes = [np.sqrt(m)]
    # sketch_sizes = [int(round(x)) for x in sketch_sizes]
    # sketch_size_formulas = ["sqrt(m)"]

    print(f"sketch sizes: {sketch_sizes}\n")

    for regularizer in reg_vals:
        print(f"~~~~~~~~~~~~~~~~~~~~ regularizer = {regularizer:.2e}")
        exp_9_2_reg_on_mom = ExperimentMomentum(
            dataset.name,
            X,
            y,
            regularizer,
            is_kernel,
            tolerance,
            max_iter,
            solvers,
            sketch_sizes,
            sketch_size_formulas,
            n_repetitions,
        )

        exp_9_2_reg_on_mom.run_full_exp(verbose=False)

        if save:
            exp_name = dataset.name
            if is_kernel:
                exp_name = exp_name + "_kernel"
            exp_name = f"{exp_name}_{solvers[0]}"
            exp_name = f"{exp_name}_reg_{regularizer:1.0e}"
            exp_name = f"{exp_name}_tol_{tolerance:1.0e}"
            exp_name = f"{exp_name}_n_rep_{n_repetitions}"
            exp_name = f"{exp_name}_max_iter_{max_iter}_heuristic"
            exp_name = f"{exp_name}_new_legend"

            exp_9_2_reg_on_mom.save(exp_name, full_path)
        # print("\n")

    print("==== Results saved ====")


if __name__ == "__main__":
    regularization_on_momentum()
