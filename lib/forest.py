# -*- coding: utf-8 -*-
import re
import pandas as pd
from utils import lr_decay
import lightgbm as lgb
from sklearn.model_selection import KFold
from numpy import argmax, zeros, array
import warnings
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error as mse
warnings.filterwarnings("ignore")

PATH_DATA = "../data/"
PATH_PREPRO = PATH_DATA + "preprocessing/"

train = pd.read_csv(PATH_DATA + "train.csv")
submission = pd.read_csv(PATH_DATA + "sample_submission.csv")

train_set = pd.read_csv(PATH_PREPRO + "train_set.csv")
test_set = pd.read_csv(PATH_PREPRO + "test_set.csv")

train_set = pd.merge(train_set, train, on='segment_id')
train_set = train_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
test_set = test_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

NB_FOLDS = 10
WRITE = True
SUBMIT = True


params = {'learning_rate': 0.2894526698568187, 'boosting_type': 'dart',
          'objective': 'regression', 'metric': 'mae',
          'sub_feature': 0.5279155828037814, 'num_leaves': 316,
          'min_data': 48, 'max_depth': 85, 'max_bin': 1057,
          'min_data_in_leaf': 63, 'n_estimators': 1903}

kf = KFold(n_splits=NB_FOLDS, shuffle=True, random_state=42)
kf.get_n_splits(train_set)


dataset_target = train_set['time_to_eruption']
dataset_value = train_set.drop('time_to_eruption', axis=1)

if WRITE:
    i = 0
    for train_index, test_index in kf.split(dataset_value, dataset_target):
        i += 1
        train_target = dataset_target.iloc[train_index]
        train_val = dataset_value.iloc[train_index]

        clf = lgb.LGBMRegressor(**params)
        clf.fit(train_val.values, train_target)
        clf.booster_.save_model(f'{PATH_DATA}model/tree_{i}.txt')
        print(f"Tree {i} done")


# Submission
if SUBMIT:

    for col in dataset_value.columns:
        if col not in test_set.columns:
            test_set[col] = 0
    test_set = test_set[dataset_value.columns]

    preds = zeros((len(test_set), ), dtype=float)
    for fold in range(1, NB_FOLDS+1):
        clf = lgb.Booster(model_file=PATH_DATA+f'model/tree_{fold}.txt')
        preds += array(clf.predict(test_set))

    preds /= NB_FOLDS
    submission['time_to_eruption'] = preds
    submission.to_csv("../data/submission.csv", index=False)
    print("File generated")
