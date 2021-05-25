"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
---
Data URL: https://www.kaggle.com/c/nyc-taxi-trip-duration/data
Please download and update the path below

Formulates this into a regression problem with trip duration as the Y.
Time features are discretized.

After feature extraction: 61.5 GiB.
"""


import pandas as pd


def load_data(path, n_samples):
    """
    Loads taxi data with parsed dates

    Args:
        path (str): path specifying location of taxi datasets from Kaggle
        n_samples (int): number of rows to load from the data
    """
    df = pd.read_csv(
        path, parse_dates=["pickup_datetime", "dropoff_datetime"], nrows=n_samples,
    )
    return df


def build_features(df):
    """Adds features for month, day, hour, minute of pickup and dropoff times.
    Returns features_df, target"""
    # drop bool columns
    df = df.drop(columns=["store_and_fwd_flag"])
    df["pickup_day_of_week"] = df["pickup_datetime"].apply(lambda x: x.dayofweek)
    df["pickup_month"] = df["pickup_datetime"].apply(lambda x: x.month)

    df["dropoff_day_of_week"] = df["dropoff_datetime"].apply(lambda x: x.dayofweek)
    df["dropoff_month"] = df["dropoff_datetime"].apply(lambda x: x.month)

    df["pickup_hour_of_day"] = df["pickup_datetime"].apply(lambda x: x.hour)
    df["dropoff_hour_of_day"] = df["dropoff_datetime"].apply(lambda x: x.hour)

    df["pickup_minute_of_day"] = df["pickup_datetime"].apply(lambda x: x.minute)
    df["dropoff_minute_of_day"] = df["dropoff_datetime"].apply(lambda x: x.minute)

    df["pickup_second_of_day"] = df["pickup_datetime"].apply(lambda x: x.second)
    df["dropoff_second_of_day"] = df["dropoff_datetime"].apply(lambda x: x.second)

    features_df = df
    target = features_df["trip_duration"]
    features_df = features_df.drop(
        ["id", "pickup_datetime", "dropoff_datetime", "trip_duration"], axis=1
    )
    return features_df, target


def discretize_times(features_df):
    """Transforms month, week, hour, minutes into discrete columns"""
    categorical_columns = [
        "pickup_month",
        "dropoff_month",
        "pickup_day_of_week",
        "dropoff_day_of_week",
        "pickup_hour_of_day",
        "dropoff_hour_of_day",
        "pickup_minute_of_day",
        "dropoff_minute_of_day",
        "pickup_second_of_day",
        "dropoff_second_of_day",
    ]

    for column in categorical_columns:
        one_hot_columns = pd.get_dummies(features_df[column], prefix=column)
        features_df = pd.concat([features_df, one_hot_columns], join="inner", axis=1)

    # drop old columns
    features_df = features_df.drop(categorical_columns, axis=1)
    return features_df


def discretize_locations(features_df, bins=5000):  # bins reduced to fit in RAM
    columns = [
        "pickup_longitude",
        "pickup_latitude",
        "dropoff_longitude",
        "dropoff_latitude",
    ]
    for column in columns:
        features_df.insert(0, column + "_bins", pd.cut(features_df[column], bins=bins))

    columns_bins = [x + "_bins" for x in columns]
    features_df = pd.get_dummies(features_df, prefix_sep=" ", columns=columns_bins)
    return features_df


def main(path, n_samples=10000):
    df = load_data(path, n_samples)
    features_df, target = build_features(df)
    features_df = discretize_times(features_df)
    features_df = discretize_locations(features_df)
    return features_df, target


if __name__ == "__main__":
    PATH = "./datasets/taxi-data/train.csv"
    features_df, target = main(PATH, n_samples=100)
    print(features_df.head())
    print(features_df.shape)
