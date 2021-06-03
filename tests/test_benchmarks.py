"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
import os
import pytest
from pytest import raises
from click.testing import CliRunner

from benchmarks.run_benchmarks import (
    Benchmarks,
    Distribution,
    compute_distribution,
    run,
)
from benchmarks.benchmark_configs import Configs


class TestBenchmarks:
    def test_subsample_solver(self):
        """Verifies benchmarks contain solver, timing and residuals"""
        configs = Configs()
        configs.solvers = {"subsample"}
        configs.smaller_dimension = 100
        configs.larger_dimension = 200
        configs.problems = ["primal_random"]
        configs.sparse_formats = ["csr"]

        benchmark_runner = Benchmarks(configs)
        times_df, residual_norms_df = benchmark_runner.run(n_repetitions=1)
        assert "subsample" in times_df["solver"].values
        assert "time (1st quartile)" in times_df.columns
        print(residual_norms_df.columns)
        assert (
            "primal_random | csr | subsample | sketch_size = 44 | "
            "no-op | algo_mode = auto | accel_param = (0.0-0.0) | "
            "residual_norms (1st quartile)" in residual_norms_df.columns
        )

    def test_compute_distribution(self):
        """Confirms quartiles are computed for residuals"""
        residuals = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        residuals_distribution = compute_distribution(residuals)
        assert isinstance(residuals_distribution, Distribution)
        assert np.allclose(residuals_distribution.q1, np.array([2.0, 3.0]))

    @pytest.mark.slow
    def test_configs_rcv1_dataset(self):
        self.test_configs_with_datasets("rcv1_small")

    @pytest.mark.parametrize("dataset", ["california_housing"])
    def test_configs_with_datasets(self, dataset):
        """Verifies benchmarks contain solver, timing and residuals"""
        configs = Configs(
            problems=(dataset,),
            solvers={"subsample"},
            sparse_formats=("csr",),
            smaller_dimension=10,
            larger_dimension=50,
            sketch_sizes=(7, 3),
            tolerance=1e-1,
            max_iterations=100,
        )

        benchmark_runner = Benchmarks(configs)
        times_df, residual_norms_df = benchmark_runner.run(
            n_repetitions=1, verbose=True
        )
        assert "subsample" in times_df["solver"].values
        assert "time (1st quartile)" in times_df.columns
        print("============acutal columns==================")
        print(residual_norms_df.columns)
        assert (
            "rcv1_small | csr | subsample | sketch_size = 7 | "
            "no-op | algo_mode = auto | accel_param = (0.0-0.0) | "
            "residual_norms (median)" in residual_norms_df.columns
        ) or (
            "california_housing | dense | subsample | sketch_size = 7 | "
            "no-op | algo_mode = auto | accel_param = (0.0-0.0) | "
            "residual_norms (median)" in residual_norms_df.columns
        )

    def test_cli(self):
        runner = CliRunner()
        result = runner.invoke(run, ["small", "--no-save", "--n-repetitions=1"])
        assert result.exit_code == 0

    @pytest.mark.slow
    def test_cli_saving(self):
        runner = CliRunner()
        # setups up empty directory and changes cwd to it
        with runner.isolated_filesystem():
            folder = os.getcwd()
            result = runner.invoke(
                run, ["small", "--save", "--n-repetitions=1", f"--folder={folder}"],
            )
        assert result.exit_code == 0, print(result)


class TestConfigs:
    def test_instantiation(self):
        """Confirms Configs can be instantiated"""
        _ = Configs()

    def test_attributes(self):
        """Confirms class contains expected attributes"""
        configs = Configs()
        assert configs.alpha == 1.0
        assert configs.tolerance == 1e-2
        assert configs.name == "default"
        assert configs.problems == ("primal_random",)
        assert configs.operator_modes == (False,)
        assert configs.solvers == {"direct", "cg", "subsample"}
        assert configs.smaller_dimension == 2000
        assert configs.larger_dimension == 8000

        # Check smaller_dimension must indeed be smaller
        with raises(ValueError):
            configs = Configs(smaller_dimension=100000, larger_dimension=10,)

        with raises(ValueError):
            configs = Configs(smaller_dimension=100000)

    def test_set_sketch_size(self, X):
        """Confirms sketch_size is correctly set"""
        # Check its computation at initialization
        # with configs.set_sketch_sizes()
        # int(min(self.smaller_dimension ** (1 / 2), 50000))
        configs = Configs()  # by default smaller_dimension = 2000
        assert configs.sketch_sizes == tuple([44])

        # Check explicit initialization
        configs = Configs(sketch_sizes=tuple([10]))
        assert configs.sketch_sizes == tuple([10])

        # Check default initialization for other attribute
        configs = Configs(smaller_dimension=100)  # larger_dimension = 8000
        assert configs.sketch_sizes == tuple([10])

        # Check ceiling
        configs = Configs(smaller_dimension=10000000000, larger_dimension=10000000000,)

        assert configs.sketch_sizes == tuple([50000])

        # Check its modification after initialization
        configs.sketch_sizes = tuple([3, 9, 100])
        assert configs.sketch_sizes == tuple([3, 9, 100])

        # Check that it  must be be between 1 and smaller_dimension
        with raises(ValueError):
            configs = Configs(sketch_sizes=tuple([-9]))
        with raises(ValueError):
            configs = Configs(sketch_sizes=tuple([9000000]))
