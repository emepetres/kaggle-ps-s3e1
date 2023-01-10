# kaggle-ps-s3e1

Kaggle Playground Series 3, Episode 1 competition

NOTES

* The best model without tuning is catboost, but LightGBP performs better after tuning
* Adding the original dataset, with a new column indicating its origin, improves the performance by 0.045!
* Localization feat engineering is key, but not all features give improvements

Ideas to obtain the best model:

* [x] K-fold
* [x] Train & validation over decision tree and XGBoost
* [x] Test & submission
* [x] Stratified K-fold with bins -> Improvement
* [x] Test using LightGBM -> Improvement
* [x] Test using CatBoost -> Improvement
* [x] Add original dataset + column indicating origin -> Huge improvement
* [ ] Localization feature engenieering
  * [x] postcode as numerical -> Tiny improvement
  * [ ] postcode & other geo data as categorical
  * [ ] use reverse geocoder
  * [x] add distance to main cities and coast line -> improvement
  * [x] add distance to cluster centroids -> improvement
  * [x] add rotation features -> improvement
* [x] Split folds in 10 bins -> improvement
* [ ] Combination of features
* [ ] PCA / y-sne /feature importance / trim of less important features
* [ ] NN Tabular model
* [x] Hyperparameters tunning -> improvement (copied from other notebooks)
* [ ] Try ensembles of different algoritms

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=lgbm]  # [xgb|lgbm|cb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
