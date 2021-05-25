"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.

Experiments for constant versus increasing versus no-momentum

Section 9.2 in the LaTex

baselines:
increasing momentum: gamma decreasing from 1 to .5, beta increasing from 0 to .5
heuristic increasing momentum: gamma = 1, beta increasing from 0 to .5
constant momentum: gamma = 1, beta = 0.5
no momentum: gamma = 1, beta = 0 ("algo_mode" = auto)

Momentum:
* set parameters from equation 31
* try eta: [0.99,0.9,0.6]


Run above with different
- sketch solvers (subcount, count, subsample)
- sketch sizes [sqrt(m), m^{3/2}, m/2].


Output:
- residual versus time
- residual versus iterations


For example, run from the ridge_sketch folder
```
$ python experiments/momentum_regimes.py boston -k False -h True -r 1e-6 -t 1e-4 -m 2000 -n 3
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
from experiments.plot_experiments import (
    plot_runs_over_iterations,
    plot_runs_over_time,
    bar_plot_times_momentum,
)


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
        use_heuristic,
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
        self.use_heuristic = use_heuristic
        self.tolerance = tolerance
        self.max_iter = max_iter

        self.solvers = solvers

        self.sketch_sizes = sketch_sizes
        self.sketch_size_formulas = sketch_size_formulas

        # For error areas 1st/3rd quartiles
        self.n_repetitions = n_repetitions

        # df of times of all runs of the experiment
        self.times_df = None
        # df of relative residual norms of all runs of the experiment
        self.residual_norms_df = None

    def run_full_exp(self, verbose=True):
        # dict of the outputs
        all_residual_norms = {}

        # No momentum runs
        print("NO MOMENTUM")
        no_mom_times, no_mom_residual_norms = self.run_no_momentum()
        all_residual_norms.update(no_mom_residual_norms)
        print()

        # Constant momentum runs
        print("CONSTANT MOMENTUM")
        cst_mom_times, cst_mom_residual_norms = self.run_constant_momentum()
        all_residual_norms.update(cst_mom_residual_norms)
        print()

        # Increasing momentum runs
        print("INCREASING MOMENTUM")
        inc_mom_times, inc_mom_residual_norms = self.run_increasing_momentum()
        all_residual_norms.update(inc_mom_residual_norms)
        print()

        # converting the outputs to dataframes
        concatenated_times = [no_mom_times, cst_mom_times, inc_mom_times]
        new_time_dict = {}
        for k in no_mom_times.keys():
            values = []
            for d in concatenated_times:
                for x in d[k]:
                    values.append(x)
            new_time_dict[k] = values
        times_df = pd.DataFrame(new_time_dict)

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

        # Plot time bar plot for each momentum version
        for solver in self.solvers:
            df_solver = self.times_df[self.times_df["solver"] == solver]
            fig_no_mom, fig_cst_mom, fig_inc_mom, fig_grouped = bar_plot_times_momentum(
                df_solver
            )

            if save_path:
                file_name = exp_name + "_time_bar"
                file_name = re.sub("[()]", "", file_name)

                full_path_no_mom = os.path.join(
                    save_path, f"{file_name}_{solver}_no_mom.pdf"
                )
                fig_no_mom.write_image(full_path_no_mom)

                full_path_cst_mom = os.path.join(
                    save_path, f"{file_name}_{solver}_cst_mom.pdf"
                )
                fig_cst_mom.write_image(full_path_cst_mom)

                full_path_inc_mom = os.path.join(
                    save_path, f"{file_name}_{solver}_inc_mom.pdf"
                )
                fig_inc_mom.write_image(full_path_inc_mom)

                full_path_grouped = os.path.join(
                    save_path, f"{file_name}_{solver}_grouped.pdf"
                )
                fig_grouped.write_image(full_path_grouped)
            else:
                fig_no_mom.show()
                fig_cst_mom.show()
                fig_inc_mom.show()
                fig_grouped.show()

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

        # Time plot
        fig_time = plot_runs_over_time(
            filtered_run_names,
            formatted_run_names,
            self.times_df,
            self.residual_norms_df,
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
            full_path = os.path.join(save_path, f"{file_name}_time.pdf")

            fig_time.write_image(full_path)
        else:
            fig_time.show()


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
    # formatted_run_names = [r.split(" |")[0] for r in run_names]
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


@click.command()
@click.argument("dataset_name")
@click.option(
    "-k", "--is-kernel", type=bool, help="kernel version",
)
@click.option(
    "-h", "--use-heuristic", type=bool, help="use heuristic for increasing momentum",
)
@click.option(
    "-r", "--regularizer", type=float, help="regularizer",
)
@click.option(
    "-t", "--tolerance", type=float, help="tolerance to reach",
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
def momentum_regimes(
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
    solvers = ["subsample", "count"]
    # solvers = ["count"]
    # solvers = ["subsample"]
    # solvers = ["subcount", "count"]  # both give almost the same updates

    # Code obviously not compatible with
    # ["direct", "cg"]
    # since momentum is not available for these solvers

    # Sketch sizes
    if is_kernel:
        m = X.shape[0]
    else:
        m = min(X.shape)
    print("m = ", m)

    sketch_sizes = [m * 0.1, m * 0.5, m * 0.9]
    sketch_sizes = [int(round(x)) for x in sketch_sizes]
    sketch_size_formulas = ["10%", "50%", "90%"]

    # sketch_sizes = [1, m * .1, m * .5, m * .9]
    # sketch_sizes = [int(round(x)) for x in sketch_sizes]
    # sketch_size_formulas = ["1", "10%", "50%", "90%"]

    # sketch_sizes = [m ** (2 / 3), m / 4, m / 2]
    # sketch_sizes = [int(round(x)) for x in sketch_sizes]
    # sketch_size_formulas = ["m^(2/3)", "m/4", "m/2"]

    print(f"sketch sizes: {sketch_sizes}\n")

    exp_9_2 = ExperimentMomentum(
        dataset.name,
        X,
        y,
        regularizer,
        is_kernel,
        use_heuristic,
        tolerance,
        max_iter,
        solvers,
        sketch_sizes,
        sketch_size_formulas,
        n_repetitions,
    )

    exp_9_2.run_full_exp(verbose=False)

    if save:
        exp_name = dataset.name
        if is_kernel:
            exp_name = exp_name + "_kernel"
        exp_name = f"{exp_name}_solvers_{('_').join(solvers)}"
        exp_name = f"{exp_name}_reg_{regularizer:1.0e}"
        exp_name = f"{exp_name}_tol_{tolerance:1.0e}"
        exp_name = f"{exp_name}_n_rep_{n_repetitions}"
        exp_name = f"{exp_name}_max_iter_{max_iter}"

        if use_heuristic:
            exp_name = f"{exp_name}_heuristic"

        try:
            folder_path = os.path.join(
                os.getcwd(), "experiments/results/9_2_sketch_size_time"
            )
        except FileNotFoundError:
            folder_path = "./experiments/results/9_2_sketch_size_time"
        exp_9_2.save(exp_name, folder_path)

        print("==== Results saved ====\n\n")


if __name__ == "__main__":
    momentum_regimes()
