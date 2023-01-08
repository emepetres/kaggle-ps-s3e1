DATA_STORAGE_PATH = "/run/media/jcarnero/linux-data"
COMPETITION = "playground-series-s3e1"

DATA_PATH = DATA_STORAGE_PATH + "/kaggle/" + COMPETITION
INPUTS = DATA_PATH + "/input"
PREPROCESSED = DATA_PATH + "/preprocess"
OUTPUTS = DATA_PATH + "/output"

TARGET = "MedHouseVal"
TRAIN_DATA = INPUTS + "/train.csv"
TEST_DATA = INPUTS + "/test.csv"
SUBMISSION_SAMPLE = INPUTS + "/sample_submission.csv"

PREPROCESSED_TRAIN_DATA = PREPROCESSED + "/train.csv"
PREPROCESSED_TEST_DATA = PREPROCESSED + "/test.csv"
TRAIN_GEO_CACHE = PREPROCESSED + "/train_geo.csv"
TEST_GEO_CACHE = PREPROCESSED + "/test_geo.csv"
ORIGINAL_GEO_CACHE = PREPROCESSED + "/original_geo.csv"
TRAIN_FOLDS = PREPROCESSED + "/train_folds.csv"
