# -*- coding: utf-8 -*-

import re
import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
from utils import preprocess_data, rmse
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

WRITE = False
SUBMIT = True

PATH_DATA = "../data/"
PATH_PREPRO = PATH_DATA + "preprocessing/"

train = pd.read_csv(PATH_DATA + "train.csv")
submission = pd.read_csv(PATH_DATA + "sample_submission.csv")

if WRITE:
    train_set = preprocess_data()
    train_set.to_csv(PATH_PREPRO + "train_set.csv", index=False)
    test_set = preprocess_data("test")
    test_set.to_csv(PATH_PREPRO + "test_set.csv", index=False)
else:
    train_set = pd.read_csv(PATH_PREPRO + "train_set.csv")
    test_set = pd.read_csv(PATH_PREPRO + "test_set.csv")


train_set = pd.merge(train_set, train, on='segment_id')
train_set = train_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
test_set = test_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))


# Training
params = {'num_leaves': 600, 'max_bin': 2713, 'num_iterations': 2250,
          'n_estimators': 2800, 'max_depth': 35, 'min_child_samples': 543,
          'learning_rate': 0.0065, 'min_data_in_leaf': 40,
          'bagging_fraction': 0.78359, 'feature_fraction': 0.08613,
          'random_state': 42}
model = LGBMRegressor(**params)

train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
y = train_set['time_to_eruption']

if not SUBMIT:
    train, val, y, y_val = train_test_split(train, y,
                                            random_state=42,
                                            test_size=0.2)
    model.fit(train, y)
    preds = model.predict(val)
    print('Simple LGB model rmse: ', rmse(y_val, preds))
    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_,
                                          train.columns)),
                               columns=['Value', 'Feature'])
    plt.figure(figsize=(10, 20))
    sns.barplot(x="Value", y="Feature",
                data=feature_imp.sort_values(by="Value", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('../data/lgbm_importances-01.png')

else:
    model.fit(train, y)


# Submission
if SUBMIT:
    segments = test_set.segment_id
    test_set = test_set[train.columns]
    preds = model.predict(test_set)
    test_set["time_to_eruption"] = preds
    test_set = pd.concat([segments, test_set], axis=1)
    submission = pd.merge(submission,
                          test_set[['segment_id', 'time_to_eruption']],
                          on='segment_id')
    submission.drop(['time_to_eruption_x'], axis=1, inplace=True)
    submission.columns = ['segment_id', 'time_to_eruption']
    submission.to_csv('../data/submission.csv', index=False)
