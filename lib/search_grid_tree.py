# -*- coding: utf-8 -*-

import numpy as np
import re
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")

PATH_DATA = "../data/"
PATH_PREPRO = PATH_DATA + "preprocessing/"

train_file = pd.read_csv(PATH_DATA + "train.csv")
train_set = pd.read_csv(PATH_PREPRO + "train_set.csv")
train_set = pd.merge(train_set, train_file, on='segment_id')
train_set = train_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

train = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
y = train_set['time_to_eruption']
x_train, x_test, y_train, y_test = train_test_split(train, y,
                                                    random_state=42,
                                                    test_size=0.2)

pp = {}
min = 2800000

for i in range(20):
    print('iteration number', i+1)
    params = {}  # initialize parameters
    params['learning_rate'] = np.random.uniform(0.2, 0.3)
    params['boosting_type'] = np.random.choice(['gbdt', 'dart', 'goss'])
    params['objective'] = 'regression'
    params['metric'] = 'mae'
    params['sub_feature'] = np.random.uniform(0, 1)
    params['num_leaves'] = np.random.randint(400, 600)
    params['min_data'] = np.random.randint(10, 100)
    params['max_depth'] = np.random.randint(105, 200)
    params['max_bin'] = np.random.randint(1005, 3200)
    params['min_data_in_leaf'] = np.random.randint(40, 100)
    params['n_estimators'] = np.random.randint(800, 4000)

    model = LGBMRegressor(**params)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    mae = mse(y_test, preds, squared=False)
    if mae < min:
        min = mae
        pp = params

print('Minimum is: ', min)
print('Used params', pp)
