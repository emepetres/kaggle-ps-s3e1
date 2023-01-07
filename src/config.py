DATA_STORAGE_PATH = "/run/media/jcarnero/linux-data"
COMPETITION = "playground-series-s3e1"

DATA_PATH = DATA_STORAGE_PATH + "/kaggle/" + COMPETITION
INPUTS = DATA_PATH + "/input"
PREPROCESSED = DATA_PATH + "/preprocess"
OUTPUTS = DATA_PATH + "/output"

TARGET = "MedHouseVal"
TRAINING_DATA = INPUTS + "/train.csv"
TEST_DATA = INPUTS + "/test.csv"
SUBMISSION_SAMPLE = INPUTS + "/sample_submission.csv"
TRAINING_FOLDS = PREPROCESSED + "/merged_train_folds.csv"
