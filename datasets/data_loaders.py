"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""


import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import (
    fetch_rcv1,
    fetch_california_housing,
    load_boston,
)
from sklearn.preprocessing import LabelEncoder

try:
    import pre_process_taxi_data
    from year_prediction_data import load_year_prediction_data
except ImportError:
    from datasets import pre_process_taxi_data
    from datasets.year_prediction_data import load_year_prediction_data


class Dataset(ABC):
    """
    Abstract class defining dataset object

    Parameters
    ----------
    - is_small : boolean
        Loads only the first 100 rows of the dataset if True,
        else loads the entire dataset.
        Default is False.


    Attributes
    ----------
    - _n_samples : int
        Number of samples. Set to 100 if "is_small" is True.
    - _n_features : int
        Number of features.
    - _sparse_format : str
        Format of the design matrix X of the dataset.
        Can be either "dense" (numpy array),
        "csr" or "csc" (scipy sparse array).
    """

    def __init__(self, is_small=False):
        self.is_small = is_small

    def get_dim(self):
        """Get dimensions of the dataset"""
        return self._n_samples, self._n_features

    def get_sparse_format(self):
        """Get the format of the dataset (dense, csr or csc)"""
        return self._sparse_format

    @abstractmethod
    def load_X_y(self):
        """Load the dataset in X and y"""
        pass


class BostonDataset(Dataset):
    def __init__(self, is_small=False):
        super().__init__(is_small=is_small)
        self.name = "boston"
        if self.is_small:
            self._n_samples = 100
        else:
            self._n_samples = 506
        self._n_features = 13
        self._sparse_format = "dense"

    def load_X_y(self):
        """
        Load and return the boston house-prices dataset (regression)from
        sklearn

        Dataset is stored in 2D numpy.ndarray's with
        506 samples and 13 features.

        Features
        ----------
        - CRIM per capita crime rate by town
        - ZN proportion of residential land zoned for lots over 25,000 sq.ft.
        - INDUS proportion of non-retail business acres per town
        - CHAS Charles River dummy variable
        (= 1 if tract bounds river; 0 otherwise)
        - NOX nitric oxides concentration (parts per 10 million)
        - RM average number of rooms per dwelling
        - AGE proportion of owner-occupied units built prior to 1940
        - DIS weighted distances to five Boston employment centres
        - RAD index of accessibility to radial highways
        - TAX full-value property-tax rate per $10,000
        - PTRATIO pupil-teacher ratio by town
        - B 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
        - LSTAT % lower status of the population
        - MEDV Median value of owner-occupied homes in $1000’s
        """
        X, y = load_boston(return_X_y=True)
        y = y.reshape(-1, 1)
        return X[: self._n_samples, :], y[: self._n_samples, :]


class CaliforniaHousingDataset(Dataset):
    def __init__(self, is_small=False):
        super().__init__(is_small=is_small)
        self.name = "california_housing"
        if self.is_small:
            self._n_samples = 100
        else:
            self._n_samples = 20640
        self._n_features = 8
        self._sparse_format = "dense"

    def load_X_y(self):
        """
        Load and return the California housing dataset

        Matrix X is stored in a dense numpy array with
        20,640 samples and 8 features.

        Features
        ----------
            - MedInc: median income in block
            - HouseAge: median house age in block
            - AveRooms: average number of rooms
            - AveBedrms: average number of bedrooms
            - Population: block population
            - AveOccup: average house occupancy
            - Latitude: house block latitude
            - Longitude: house block longitude
        """
        california_housing = fetch_california_housing()
        X, y = california_housing.data, california_housing.target
        y = y.reshape(-1, 1)
        return X[: self._n_samples, :], y[: self._n_samples, :]


class YearPredictionDataset(Dataset):
    def __init__(
        self,
        path="./datasets/year_prediction_data/YearPredictionMSD.csv",
        is_small=False,
    ):
        super().__init__(is_small=is_small)
        self.name = "year_prediction"
        self.path = path
        if self.is_small:
            self._n_samples = 100
        else:
            self._n_samples = 515345
        self._n_features = 90
        self._sparse_format = "dense"

    def load_X_y(self):
        """
        Load and return the Year Prediction MSD dataset

        Matrix X is stored in a dense numpy array with
        515,345 samples and 90 features.
        """
        X, y = load_year_prediction_data(self.path, n_samples=self._n_samples)
        y = y.reshape(-1, 1)
        return X, y


class Rcv1Dataset(Dataset):
    def __init__(self, is_small=False):
        super().__init__(is_small=is_small)
        self.name = "rcv1"
        if self.is_small:
            self._n_samples = 100
        else:
            self._n_samples = 804414
        self._n_features = 47236
        self._sparse_format = "csr"

    def load_X_y(self):
        """
        Load and return the rcv1 dataset

        Matrix X is stored in a sparse CSR array with
        80,4414 samples and 47,236 features.

        Compressed size is 658 MB.
        """
        rcv1 = fetch_rcv1()
        X, y = rcv1.data, rcv1.target
        names = np.array(rcv1.target_names.tolist())

        # integer encode multi-labels
        labels = self._y_to_unique_labels(y.toarray(), names)
        y = self._integer_encode(labels)
        y = y.reshape(-1, 1)
        return X[: self._n_samples, :], y[: self._n_samples, :]

    def _y_to_unique_labels(self, y, names):
        """Concatenates multi-labels into a unique string for
        each combination of labels.
        """
        names = np.array(names)
        y_names = []
        for row in y:
            indices = np.flatnonzero(row)
            y_names.append("".join(names[indices]))
        return y_names

    def _integer_encode(self, labels):
        """Encodes unique labels into integers for use in regression."""
        le = LabelEncoder()
        integers = le.fit_transform(labels)
        return integers.astype(np.float64)


class TaxiDataset(Dataset):
    # path is to change by user
    def __init__(self, path="./datasets/taxi-data/train.csv", is_small=False):
        super().__init__(is_small=is_small)
        self.name = "taxi"
        self.path = path
        if self.is_small:
            self._n_samples = 100
        else:
            self._n_samples = 10000
        self._n_features = 20320
        self._sparse_format = "dense"

    def load_X_y(self):
        """
        Load and return the subsampled discretized taxi dataset

        Matrix X is stored in a dense numpy array with
        n_samples and 20,320 features.
        """
        X, y = self._taxi_data(path=self.path)
        y = y.reshape(-1, 1)
        return X[: self._n_samples, :], y[: self._n_samples, :]

    def _taxi_data(self, path="../../../taxi-data/train.csv", n_samples=10000):
        """
        Loads `n_samples` rows of the taxi dataset.

        Discretization of the features is based on the first 10,000 rows
        out of 1.4 million for computational efficiency.
        """
        features_df, target = pre_process_taxi_data.main(path, n_samples)
        return (
            features_df.values,
            target.to_numpy(dtype=np.float64),
        )


# get all Dataset subclasses
DATASET_CLASSES = [cls.__name__ for cls in Dataset.__subclasses__()]


if __name__ == "__main__":
    print(f"List of available datasets: {DATASET_CLASSES}\n")

    print("--- RCV1 dataset")
    rcv1 = Rcv1Dataset(is_small=False)
    X, y = rcv1.load_X_y()
    print(f"X shape: {X.shape}")
    print(f"Y example label: {y[3]}")
    print(f"Format: {rcv1._sparse_format}\n")

    print("--- Year Prediction MSD dataset")
    year_prediction = YearPredictionDataset(is_small=False)
    X, y = year_prediction.load_X_y()
    print(f"X shape: {X.shape}")
    print(f"Y example label: {y[3]}")
    print(f"Format: {year_prediction._sparse_format}\n")

    print("--- Taxi dataset")
    taxi = TaxiDataset(is_small=True)
    X, y = taxi.load_X_y()
    print(f"X shape: {X.shape}")
    print(f"Y example label: {y[3]}")
    print(f"Format: {taxi._sparse_format}")
