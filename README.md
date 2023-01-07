# kaggle-ps-s3e1

Kaggle Playground Series 3, Episode 1 competition

NOTES

* The best model performing so far is CatBoost
* Adding the original dataset, with a new column indicating its origin, improves the performance by 0.045!

Ideas to obtain the best model:

* [x] K-fold
* [x] Train & validation over decision tree and XGBoost
* [x] Test & submission
* [x] Stratified K-fold with bins
* [x] Test using LightGBM
* [x] Test using CatBoost
* [x] Add original dataset + column indicating origin
* [ ] Localization feature engenieering
* [ ] Combination of features
* [ ] PCA / y-sne /feature importance / trim of less important features
* [ ] NN Tabular model
* [ ] Hyperparameters tunning
* [ ]  Try ensembles of different algoritms

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [rf|xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
