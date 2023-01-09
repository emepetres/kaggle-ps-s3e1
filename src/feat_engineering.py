from typing import Tuple
import numpy as np
import pandas as pd

import config
from common.kaggle import download_competition_data

from common.feature_engineering.geo import (
    compute_geo_features,
    add_distance_to_locations,
    add_distance_to_line,
    add_distance_to_cluster_centroids,
    add_rotation_features,
)


def _merge_with_original_data(
    train: pd.DataFrame, test: pd.DataFrame, postcode: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.datasets import fetch_california_housing

    original_data = fetch_california_housing()
    synthetic_data = train.drop(columns="id")
    test_data = test
    original_data = pd.DataFrame(
        data=np.hstack([original_data["data"], original_data["target"].reshape(-1, 1)]),
        columns=synthetic_data.columns,
    )

    original_data["synthetic_data"] = 0
    synthetic_data["synthetic_data"] = 1
    test_data["synthetic_data"] = 1

    if postcode:
        df_orig_geo = compute_geo_features(df_train, cache=config.ORIGINAL_GEO_CACHE)
        df_synth_geo = compute_geo_features(df_train, cache=config.TRAIN_GEO_CACHE)
        df_test_geo = compute_geo_features(df_train, cache=config.TEST_GEO_CACHE)

        original_data["postcode"] = df_orig_geo["postcode"]
        synthetic_data["postcode"] = df_synth_geo["postcode"]
        test_data["postcode"] = df_test_geo["postcode"]

    merged_train = pd.concat([synthetic_data, original_data]).reset_index(drop=True)

    return merged_train, test_data


if __name__ == "__main__":
    # Download competition data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df_train = pd.read_csv(config.TRAIN_DATA)
    df_test = pd.read_csv(config.TEST_DATA)

    # Merge training data with original dataset
    df_train, df_test = _merge_with_original_data(df_train, df_test, postcode=True)
    df_train["postcode"] = (
        df_train["postcode"]
        .fillna("0")
        .apply(lambda x: str(x).replace("-", "."))
        .astype(float)
    )
    df_test["postcode"] = (
        df_test["postcode"]
        .fillna("0")
        .apply(lambda x: str(x).replace("-", "."))
        .astype(float)
    )

    # Add distance to main cities
    cities = {
        "Sacramento": (38.576931, -121.494949),
        "SanFrancisco": (37.780080, -122.420160),
        "SanJose": (37.334789, -121.888138),
        "LosAngeles": (34.052235, -118.243683),
        "SanDiego": (32.715759, -117.163818),
    }
    add_distance_to_locations(cities, df_train)
    add_distance_to_locations(cities, df_test)

    # Add distance to coast line
    coast = [
        [32.664472968971786, -117.16139777220666],
        [33.20647603453836, -117.38308931734736],
        [33.77719697387153, -118.20238415808473],
        [34.46343131623148, -120.01447157053916],
        [35.42731619324845, -120.8819602254066],
        [35.9284107340049, -121.48920228383551],
        [36.982737132545495, -122.028973002425],
        [37.61147966825591, -122.49163361836126],
        [38.3559871217218, -123.06032062543764],
        [39.79260770260524, -123.82178288918176],
        [40.799744611668416, -124.18805587680554],
        [41.75588735544064, -124.19769463963775],
    ]
    add_distance_to_line("coast", coast, df_train)
    add_distance_to_line("coast", coast, df_test)

    # add distance to cluster centroids
    add_distance_to_cluster_centroids(df_train, df_test)

    # add rotation features
    add_rotation_features(df_train, [15, 30, 45])
    add_rotation_features(df_test, [15, 30, 45])

    df_train.to_csv(config.PREPROCESSED_TRAIN_DATA, index=False)
    df_test.to_csv(config.PREPROCESSED_TEST_DATA, index=False)
