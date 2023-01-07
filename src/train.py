import argparse
from datetime import datetime
from pathlib import Path
from typing import Tuple
import pandas as pd
import numpy as np

import config
from model_dispatcher import (
    CustomModel,
    DecisionTreeModel,
    XGBoost,
)


def run(fold: int, model: CustomModel) -> Tuple[float, np.ndarray]:
    # load the full training data with folds
    df = pd.read_csv(config.TRAINING_FOLDS)
    df_test = pd.read_csv(config.TEST_DATA)

    # all columns are features except target, id and kfold columns
    features = [f for f in df.columns if f not in (config.TARGET, "kfold", "id")]
    ord_features = features  # all features are ordinal
    cat_features = []

    # initialize model
    custom_model = model(
        df, fold, config.TARGET, cat_features, ord_features, test=df_test
    )

    # encode all features
    custom_model.encode()

    # fit model on training data
    custom_model.fit()

    # predict on validation data and get rmse score
    rmse = custom_model.predict_and_score()

    # print rmse
    print(f"Fold = {fold}, RMSE = {rmse}")

    # predict on test
    preds = custom_model.predict_test()

    return (rmse, preds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="xgb")

    args = parser.parse_args()

    model = None
    if args.model == "rf":
        model = DecisionTreeModel
    elif args.model == "xgb":
        model = XGBoost
    else:
        raise argparse.ArgumentError(
            "Only 'rf' (random forest) and 'xgb' (XGBoost) models are supported"
        )

    validation_scores = []
    preds = []
    for fold_ in range(5):
        score, predictions = run(fold_, model)
        validation_scores.append(score)
        preds.append(predictions)

    valid_rmse = np.mean(validation_scores)
    print(f"Validation RMSE = {valid_rmse}")

    # create submission
    pred = np.mean(np.array(preds), axis=0)

    df_sub = pd.read_csv(config.SUBMISSION_SAMPLE)
    df_sub[config.TARGET] = pred

    dt = datetime.now().strftime("%y%m%d.%H%M")
    submission_file = (
        Path(config.OUTPUTS) / f"{dt}-submission-{args.model}-{valid_rmse}.csv"
    )
    submission_file.parent.mkdir(exist_ok=True)
    df_sub.to_csv(submission_file, index=False)
