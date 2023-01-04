import argparse
import pandas as pd
import numpy as np

import config
from model_dispatcher import (
    CustomModel,
    LogisticRegressionModel,
    DecisionTreeModel,
    DecisionTreeModelSVD,
    XGBoost,
)


def run(fold: int, model: CustomModel) -> float:
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FOLDS)

    # all columns are features except target, id and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold", "id")]
    ord_features = features  # all features are ordinal
    cat_features = []

    # initialize model
    custom_model = model(df, fold, config.TARGET, cat_features, ord_features)

    # encode all features
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data and get rmse score
    rmse = custom_model.predict_and_score()

    # print rmse
    print(f"Fold = {fold}, RMSE = {rmse}")

    return rmse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="xgb")

    args = parser.parse_args()

    model = None
    if args.model == "lr":
        model = LogisticRegressionModel
    elif args.model == "rf":
        model = DecisionTreeModel
    elif args.model == "svd":
        model = DecisionTreeModelSVD
    elif args.model == "xgb":
        model = XGBoost
    else:
        raise argparse.ArgumentError(
            "Only 'lr' (logistic regression)"
            ", 'rf' (random forest)"
            ", 'svd' (random forest with truncate svd)"
            ", 'xgb' (XGBoost)"
            " models are supported"
        )

    scores = [run(fold_, model) for fold_ in range(5)]
    print(f"Mean RMSE = {np.mean(scores)}")
