from pathlib import Path
import pandas as pd
import numpy as np
from sklearn import model_selection

from common.kaggle import download_competition_data
import config


def merge_with_original_data(data: pd.DataFrame) -> pd.DataFrame:
    from sklearn.datasets import fetch_california_housing

    original_data = fetch_california_housing()
    synthetic_data = data.drop(columns="id")
    original_data = pd.DataFrame(
        data=np.hstack([original_data["data"], original_data["target"].reshape(-1, 1)]),
        columns=synthetic_data.columns,
    )

    synthetic_data["synthetic_data"] = 1
    original_data["synthetic_data"] = 0

    return pd.concat([synthetic_data, original_data]).reset_index(drop=True)


if __name__ == "__main__":
    # Download competition data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df = pd.read_csv(config.TRAINING_DATA)

    # Merge training data with original dataset
    df = merge_with_original_data(df)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # calculate the number of bins by Struge's rule
    # Taking the floor value, you can also just round it
    num_bins = int(np.floor(1 + np.log2(len(df))))

    # bin targets
    df.loc[:, "bins"] = pd.cut(df[config.TARGET], bins=num_bins, labels=False)

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=df.bins.values)):
        df.loc[v_, "kfold"] = f

    # drop de bins column
    df = df.drop("bins", axis=1)

    # save the new csv with kfold column
    Path(config.TRAINING_FOLDS).parent.mkdir(exist_ok=True)
    df.to_csv(config.TRAINING_FOLDS, index=False)
