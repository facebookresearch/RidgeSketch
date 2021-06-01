# RidgeSketch
An open source package in Python for solving large scale ridge regression using the sketch-and-project technique. 

For details, see the [RidgeSketch paper](https://arxiv.org/abs/2105.05565). 

RidgeSketch aims to match the Scikit-learn API:
```
n_samples, n_features = 1000, 500
X = np.random.rand(n_samples, n_features)
y = np.random.rand(n_samples, 1)

model = RidgeSketch(
            alpha=1e-1,
            solver="subsample",
            sketch_size=10,
            verbose=1,
        )

model.fit(X, y)
```

## Installation

First ensure you're in a Python 3 virtual environment (see instructions below). 

To install the package and requirements:
```pip install -e .```

### Setup a virtual environment
Create a [Python 3 virtual environment](https://docs.python.org/3/tutorial/venv.html).

For Unix or MacOS this can be done by executing: `source activate [env name]`. Then,
1. `python3 -m venv ridgesketch-env`
2. `source activate ridgesketch-env/bin/activate`


## Tutorials

Tutorial notebooks for running and adding new sketches are in the `tutorials` subdirectory.

## Citation

```
@misc{gazagnadou2021textttridgesketch,
      title={$\texttt{RidgeSketch}$: A Fast sketching based solver for large scale ridge regression}, 
      author={Nidham Gazagnadou and Mark Ibrahim and Robert M. Gower},
      year={2021},
      eprint={2105.05565},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```

## Advanced Usage

Please visit our [documentation](https://ridge-sketch.readthedocs.io/en/latest/) for API details.

### Run Benchmarks

To run benchmarks:
1. Specify the desired configurations in `benchmark_configs.py` (see `small` for an example)
2. Run benchmark: `python benchmarks.py [options] [name of config]`

For example to run benchmarks with the small configs: `python benchmarks.py small`

For a full list of options see:
```
Usage: benchmarks.py [OPTIONS] CONFIG_NAME

Options:
  --folder PATH            folder path where results are saved
  --n-repetitions INTEGER  number of times to rerun benchmarks
  --save / --no-save
  --help                   Show this message and exit.
```

### Add a Dataset

To add a dataset:
1. Enter `datasets/data_loaders.py` and create a new Dataset subclass
2. Specify the private arguments :
   - '_n_samples' : number of samples
   - '_n_features' : number of features
   - '_sparse_format' : format ("dense", "csr" or "csc") of the design matrix X
3. Try to run a benchmark with this dataset by following previous section


### Add a Custom Sketching Method

To create your own sketching method, inherit from `Sketch` and implement `sketch()` and `update_iterate()`:

```python

from sketching import Sketch

class MySketch(Sketch):

  def __init__(self, A, b, sketch_size):
    super().__init__(A, b, sketch_size)

  def sketch(self, r):
    """Returns a tuple of (SA, SAS, rs)"""
    pass

  def update_iterate(self, w, lmbda, step_size=1.0):
    """Returns updated weights"""
    pass
```

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) file for how to help out.


# License

See LICENSE file.





