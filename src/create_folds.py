from pathlib import Path
import pandas as pd
from sklearn import model_selection

from common.kaggle import download_competition_data
import config

if __name__ == "__main__":
    # Download competition data if necessary
    download_competition_data(config.COMPETITION, config.INPUTS)

    # Read training data
    df = pd.read_csv(config.TRAINING_DATA)

    # we create a new column called kfold and fill it with -1
    df["kfold"] = -1

    # the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    # fetch labels
    y = df["MedHouseVal"].values

    # initiate the kfold class from model_selection module
    kf = model_selection.KFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f

    # save the new csv with kfold column
    Path(config.TRAINING_FOLDS).parent.mkdir(exist_ok=True)
    df.to_csv(config.TRAINING_FOLDS, index=False)
