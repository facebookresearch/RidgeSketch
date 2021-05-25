"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.


Data from:
https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip
Please use the function below to download this dataset.

Size: 428 MiB.
"""
import os
from pathlib import Path
import requests
import zipfile
import numpy as np
import pandas as pd


def load_year_prediction_data(path, n_samples=515345):
    """Loads Year Prediction MSD data

    Dimensions: 515,345 samples and 90 features.
    Target is in the first column containing the year to predict.
    """
    df = pd.read_csv(path, header=None, nrows=n_samples)
    y = df.iloc[:, 0].values
    y = y.astype(np.float64)
    X = df.iloc[:, 1:].values
    return X, y


def download_year_prediction_data():
    filename_txt = "YearPredictionMSD.txt"
    filename_csv = "YearPredictionMSD.csv"
    data_path = "datasets/year_prediction_data/"
    full_path = os.path.join(os.getcwd(), data_path)

    if not os.path.isfile(os.path.join(full_path, filename_csv)):
        print(f"Data not found in:\n'{os.path.join(full_path, filename_csv)}'")
        print("Downloading YearPredictionMSD dataset")

        # Create folder if needed
        Path(full_path).mkdir(parents=True, exist_ok=True)

        # Download zip file
        zip_path = os.path.join(full_path, filename_txt + ".zip")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
        download_url(url, zip_path)

        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(full_path)
        os.remove(zip_path)

        # Renaming .txt in .csv
        file_path = os.path.join(full_path, filename_txt)
        os.rename(file_path, file_path[:-4] + ".csv")
    else:
        print(f"Data already found in:\n'{os.path.join(full_path, filename_csv)}'")


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, "wb") as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)


if __name__ == "__main__":
    download_year_prediction_data()

    PATH = "./datasets/year_prediction_data/YearPredictionMSD.csv"
    features, target = load_year_prediction_data(PATH, n_samples=100)
    print(features.shape)
    print(target.shape)
