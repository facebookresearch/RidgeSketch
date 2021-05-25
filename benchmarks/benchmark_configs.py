"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


from math import floor

from datasets import data_loaders
from datasets.data_loaders import DATASET_CLASSES


class Configs:
    r"""
    Configuration class for benchmarks.

    This contains the settings for which we want to compare the runtime of
    different methods.
    Instantiated configurations at the bottom of this file can be used by
    running
    `python benchmarks.py <benchmark_name>`

    Parameters
    ----------
    name : str
        name of the configuration

    tolerance : float64
        Tolerance to reach. Default is 1e-2.
    max_iterations : int
        Maximal number of iterations. Default is 10000.
    alpha : float
        Regularization strength, must be a positive number. Default is 1.
    problems : tuple
        Name of the problems or dataset on which the ridge regression problem
        is solved.
        It can be one of the following
        - ("primal_random",),
        - ("dual_random",),
        - ("primal_random", "dual_random"),
        - ("boston",)
        - ("california_housing",)
        - ("rcv1",)
        or any dataset added in the `datasets.py` file as a `Dataset` subclass.
        If the dataset name ends with '_small', then the dataset returns only
        the first 100 rows.
    density : float
        Density of the sparse data generated if the problem contains
        `primal_random` or `dual_random`, must be between 0 and 1.
        Default is .04.
    operator_modes : boolean
        Defines the covariance matrix as an operator (true) or as numpy matrix
        (false). Default is (False,).
    algo_modes : str
        If set to 'auto', it uses classical Ridge Sketch method.
        If set to 'mom', it uses the momentum version. Default is increasing
        momentum when neither 'mom_eta','step_size' or 'mom_beta' is set.
        If set to 'accel', it uses the accelerated version.
        Default is ("auto",).
    accel_params : tuple
        First (mu) and second (nu) acceleration parameters.
        Default is ((0., 0.),).
    solvers : str
        Solvers to run for the benchmark.
        Default is {'direct', 'cg', 'subsample'}.
    sparse_formats : tuple
        Sparse format of the generated matrices when required.
        Can be either dense, csr (compressed sparse rows) or
        csc (compresses sparse column).
        Default is ("csr", ).
    kernel : str
        Name of the kernel to use.
        Default is None which does not use kernel ridge regression.
        If used, problems can only belong to
        - ("primal_random",),
        - ("boston",)
        - ("california_housing",)
        - ("rcv1",)
    kernel_parameters : tuple
        Parameters of the kernel ridge regression.
        Default is (1., .5).
    smaller_dimension : int
        Smaller dimension of the generated matrices when required.
        Default is None.
    larger_dimension : int
        Larger dimension of the generated matrices when required.
        Default is None.
    sketch_sizes : tuple
        Sketch sizes used in the benchmark for all sketch-and-project solvers.
        Default is None which corresponds to
        '$\min \left( \sqrt{\text{smaller_dimension}}, 50000\right)$'
        if the kernel mode is of, otherwise sketch_sizes must be smaller than
        the number of samples. Default value for kernel mode is
        '$\min \left( \sqrt{\text{n_samples}}, 50000\right)$'.
    """

    def __init__(
        self,
        name="default",
        tolerance=1e-2,
        max_iterations=10000,
        alpha=1.0,
        problems=("primal_random",),
        density=0.04,
        operator_modes=(False,),
        algo_modes=("auto",),
        accel_params=((0.0, 0.0),),
        solvers={"direct", "cg", "subsample", },
        sparse_formats=("csr",),
        kernel=None,
        kernel_parameters=(1.0, 0.5),
        smaller_dimension=None,
        larger_dimension=None,
        sketch_sizes=None,
    ):
        self.name = name
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.problems = problems
        self.density = density  # percentage of non-zeros in X
        self.operator_modes = operator_modes
        self.algo_modes = algo_modes
        self.accel_params = accel_params
        self.solvers = solvers
        self.sparse_formats = sparse_formats
        self.kernel = kernel
        self.kernel_parameters = kernel_parameters

        # identify which problem is treated if run on a real datasets
        if len(problems) == 1:
            problem = problems[0]
            is_small = False
            if problem.endswith("_small"):
                dataset_name = problem[:-6]
                is_small = True  # takes only the first 100 rows
            else:
                dataset_name = problem
            dataset_name = "".join([x.capitalize() for x in dataset_name.split("_")])
            dataset_name += "Dataset"

        # Set smaller and larger dimensions
        if problems in [
            ("primal_random",),
            ("dual_random",),
            ("primal_random", "dual_random"),
        ]:
            self.smaller_dimension = smaller_dimension
            self.larger_dimension = larger_dimension

            if smaller_dimension is None:
                self.smaller_dimension = 2000
            if larger_dimension is None:
                self.larger_dimension = 8000

            if problems == ("primal_random",):
                n_samples = self.larger_dimension
        elif len(problems) == 1 and dataset_name in DATASET_CLASSES:
            # problem involves a real dataset
            dataset_class = getattr(data_loaders, dataset_name)
            dataset = dataset_class(is_small=is_small)
            dimensions = dataset.get_dim()
            n_samples = dimensions[0]  # for kernel version
            self.smaller_dimension = min(dimensions)
            self.larger_dimension = max(dimensions)

            self.sparse_formats = (dataset.get_sparse_format(),)
        else:
            raise ValueError("Wrong tuple of problems provided")

        if self.smaller_dimension > self.larger_dimension:
            raise ValueError(
                "'larger_dimension' must be greater than 'smaller_dimension'"
            )

        if kernel is None:
            if sketch_sizes is None:
                self.sketch_sizes = (
                    int(min(self.smaller_dimension ** (1 / 2), 50000)),
                )
            else:
                if any(i < 1 for i in sketch_sizes) or any(
                    i > self.smaller_dimension for i in sketch_sizes
                ):
                    raise ValueError(
                        "Sketch dimension must be greater or equal than 1 and smaller than smaller_dimension"
                    )
                self.sketch_sizes = sketch_sizes
        else:
            if problems != ("primal_random",) and (
                len(problems) != 1 or dataset_name not in DATASET_CLASSES
            ):
                raise ValueError(
                    "Kernel benchmark must be run on a single real dataset or with 'primal_random'"
                )

            if sketch_sizes is None:
                self.sketch_sizes = (int(min(n_samples ** (1 / 2), 50000)),)
            else:
                if any(i < 1 for i in sketch_sizes) or any(
                    i > n_samples for i in sketch_sizes
                ):
                    raise ValueError(
                        "Sketch dimension must be greater or equal than 1 and smaller than number of samples for kernel benchmark"
                    )
                self.sketch_sizes = sketch_sizes


# Small Benchmark
small = Configs(
    name="small",
    problems=("primal_random", "dual_random"),
    operator_modes=(True, False),
    solvers={"direct", "subsample", "cg"},
    sparse_formats=("csr", "csc"),
    smaller_dimension=50,
    larger_dimension=8000,
)

# Medium Benchmark
smaller_dimension = 250
medium = Configs(
    name="medium",
    problems=("primal_random",),
    solvers={"subsample", "count", "direct", "cg", },
    sparse_formats=("csr", "csc", "dense",),
    smaller_dimension=smaller_dimension,
    larger_dimension=350,
    density=0.05,
    sketch_sizes=(int(min(smaller_dimension ** (5 / 6), 50000)),),
)

# Boston Benchmark
boston = Configs(
    name="boston",
    problems=("boston",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(5, 13,),
)

# California housing small Benchmark
cali_small = Configs(
    name="cali_small",
    problems=("california_housing_small",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(4, 8,),
)

# RCV1 small Benchmark
rcv1_small = Configs(
    name="rcv1_small",
    problems=("rcv1_small",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(10, 50,),
)

# Taxi small Benchmark
taxi_small = Configs(
    name="taxi_small",
    problems=("taxi_small",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(10, 50,),
)

# Small Kernel Benchmark
# No such thing as dual and primal for Kernel. Confusing.
kernel_small = Configs(
    name="kernel_small",
    problems=("primal_random",),
    solvers={"subsample", "count", "direct", "cg", },  # , "hadamard"
    sparse_formats=("csc",),
    smaller_dimension=250,
    larger_dimension=4000,
    density=0.25,
    sketch_sizes=(100, 400,),  # sketch_size are greater than n_samples
    kernel="Matern",
    alpha=1,
)

# Small accel Kernel Benchmark
# No such thing as dual and primal for Kernel. Confusing.
# No acceleration available for direct solver.
kernel_small_ac = Configs(
    name="kernel_small_ac",
    problems=("primal_random",),
    algo_modes=("auto", "accel"),
    accel_params=((.9, 1.1), (.8, 1.), (.8, 1.2)),
    solvers={"subsample", },  # "direct",
    sparse_formats=("csc",),
    smaller_dimension=2000,
    larger_dimension=4000,
    density=0.25,
    # sketch_sizes=(100,),  # sketch_size are greater than n_samples
    kernel="Matern",
    alpha=1./4000,
    tolerance=1e-4,
)


# Kernel Benchmark
# slow
kernel_cali = Configs(
    name="kernel_cali",
    problems=("california_housing",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(4000,),
    kernel="Matern",
)

# Kernel Benchmark
kernel_rcv1_small = Configs(
    name="kernel_rcv1_small",
    problems=("rcv1_small",),  # rcv1 gives out of memory!
    solvers={"subsample", "count", "cg", },
    # sketch_sizes=(10000, 20000, 40000,),
    sketch_sizes=(10, 25, 50,),
    kernel="Matern",
)

# Sparse Medium Subcount and Subsample Benchmark
smaller_dimension = 4000
sparse_medium_half_dense = Configs(
    name="sparse_medium_half_dense",
    problems=("primal_random",),
    solvers={"subsample", "subcount", "cg", },
    sparse_formats=("csr",),
    smaller_dimension=smaller_dimension,
    larger_dimension=10000,
    density=0.5,
    sketch_sizes=(
        floor(smaller_dimension / 2),
        int(min(smaller_dimension ** (5 / 6), 50000)),
    ),
)


# RCV1 Benchmark
rcv1 = Configs(
    name="rcv1",
    problems=("rcv1",),
    solvers={"subsample", "count", "direct", "cg", },
    sketch_sizes=(
        # floor(smaller_dimension / 2),
        int(min(smaller_dimension ** (5 / 6), 50000)),
    ),
)


# Sparse Small Subcount and Subsample Benchmark
smaller_dimension = 1000
sparse_small = Configs(
    name="sparse_small",
    problems=("primal_random",),
    solvers={"subsample", "subcount", "cg", },
    sparse_formats=("csr",),
    smaller_dimension=smaller_dimension,
    larger_dimension=4000,
    density=0.5,
    # sketch_sizes=(int(min(smaller_dimension ** (5 / 6), 50000)),)
    sketch_sizes=(
        floor(smaller_dimension / 10),
        floor(smaller_dimension / 4),
        floor(smaller_dimension / 2),
    ),
)

# Sparse Subcount Benchmark
smaller_dimension = 100
sparse_subcount = Configs(
    name="sparse_subcount",
    problems=("primal_random",),
    solvers={"subcount", },
    sparse_formats=("csr",),
    smaller_dimension=smaller_dimension,
    larger_dimension=400,
    density=0.1,
    sketch_sizes=(
        floor(smaller_dimension / 10),
        floor(smaller_dimension / 4),
        floor(smaller_dimension / 2),
    ),
)

# Medium Momentum Benchmark
mom_accel_auto = Configs(
    name="mom_accel_auto",
    problems=("primal_random",),
    algo_modes=("auto", "mom", "accel"),
    solvers={"subsample", },
    sparse_formats=("csr",),
    smaller_dimension=5000,
    larger_dimension=10000,
)

# Kernel with Momentum Benchmark
kernel_mom = Configs(
    name="kernel_mom",
    problems=("primal_random",),
    algo_modes=("mom",),
    solvers={"subsample", },
    sparse_formats=("dense",),
    smaller_dimension=500,
    larger_dimension=7000,
    kernel="Matern",
)
