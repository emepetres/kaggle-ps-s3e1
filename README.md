# kaggle-ps-s3e1

Kaggle Playground Series 3, Episode 1 competition

NOTES

* XGBoost works much better than random forests.

Tasks:

* ~~K-fold~~
* ~~Train & validation over decision tree and XGBoost~~
* ~~Test & submission~~
* Stratified K-fold with bins
* Combination of features
* Add original dataset + column indicating origin
* Localization feature engenieering
* PCA / y-sne /feature importance / trim of less important features
* NN Tabular model
* Hyperparameters tunning

## Train, validation & submission

```bash
cd src
conda activate ml
python create_folds.py
python -W ignore train.py [--model=xgb]  # [rf|xgb]
```

Submission is stored in outputs folder (see `config.py` for complete path)
