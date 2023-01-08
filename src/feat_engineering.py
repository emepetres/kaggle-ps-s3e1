from typing import Tuple
import numpy as np
import pandas as pd

import config
from common.kaggle import download_competition_data

# # from common.feature_engineering.geo import compute_geo_features


def _merge_with_original_data(
    train: pd.DataFrame, test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.datasets import fetch_california_housing

    original_data = fetch_california_housing()
    synthetic_data = train.drop(columns="id")
    original_data = pd.DataFrame(
        data=np.hstack([original_data["data"], original_data["target"].reshape(-1, 1)]),
        columns=synthetic_data.columns,
    )

    synthetic_data["synthetic_data"] = 1
    original_data["synthetic_data"] = 0

    merged_train = pd.concat([synthetic_data, original_data]).reset_index(drop=True)

    merged_test = test
    merged_test["synthetic_data"] = 1

    return merged_train, merged_test


if __name__ == "__main__":
    # Download competition data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df_train = pd.read_csv(config.TRAIN_DATA)
    df_test = pd.read_csv(config.TEST_DATA)

    # Merge training data with original dataset
    df_train, df_test = _merge_with_original_data(df_train, df_test)

    # Add postcode data
    pass

    df_train.to_csv(config.PREPROCESSED_TRAIN_DATA, index=False)
    df_test.to_csv(config.PREPROCESSED_TEST_DATA, index=False)
