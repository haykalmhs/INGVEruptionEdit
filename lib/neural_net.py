# -*- coding: utf-8 -*-

import numpy as np
import re
import warnings
import pandas as pd
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold
from sklearn import preprocessing
from keras.layers import (BatchNormalization, Flatten, Convolution1D,
                          Input, Dense, LSTM)
from keras.callbacks import ModelCheckpoint
from keras import models, optimizers
from keras import backend as K

import tensorflow
import tensorflow as tf
warnings.filterwarnings("ignore")
sess = tensorflow.compat.v1.Session(
    config= tensorflow.compat.v1.ConfigProto(log_device_placement=True))
from numpy.random import seed
from keras.models import load_model
import keras.backend as K
from utils import preprocess_data

seed(2020)
tf.random.set_random_seed(2021)

NB_MODELS = 2
K_FOLD = 5
BATCH_SIZE = 4096*2
EPOCHS = 1500
GENERATE_PREPRO = True


PATH_DATA = "../data/"
PATH_PREPRO = PATH_DATA + "preprocessing/"

if GENERATE_PREPRO:
    train_set = preprocess_data()
    train_set.to_csv(PATH_PREPRO + "train_set.csv", index=False)
    test_set = preprocess_data("test")
    test_set.to_csv(PATH_PREPRO + "test_set.csv", index=False)
else:
    train_set = pd.read_csv(PATH_PREPRO + "train_set.csv")
  
train = pd.read_csv(PATH_DATA + "train.csv")
train_set = pd.merge(train_set, train, on='segment_id')
train_set = train_set.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

train_sample = train_set.drop(['segment_id', 'time_to_eruption'], axis=1)
targets = train_set['time_to_eruption']
submission = pd.read_csv(PATH_DATA + 'sample_submission.csv')
test = pd.read_csv(PATH_PREPRO + "test_set.csv")
test.drop(['segment_id'], axis=1, inplace=True)
test = test[train_sample.columns]


def get_model():
    inp = Input(shape=(1, train_sample.shape[1]))
    x = BatchNormalization()(inp)
    x = LSTM(128, return_sequences=True)(x)
    x = Convolution1D(128, (2), activation='relu', padding="same")(x)
    x = Convolution1D(84, (2), activation='relu', padding="same")(x)
    x = Convolution1D(64, (2), activation='relu', padding="same")(x)
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dense(32, activation="relu")(x)
    ttf = Dense(1, activation='relu', name='regressor')(x)

    model = models.Model(inputs=inp, outputs=ttf)
    opt = optimizers.Nadam(lr=0.005)
    model.compile(optimizer=opt, loss='mae', metrics=['mae'])
    return model


def normalize(X_train, X_valid, X_test, normalize_opt, excluded_feat):
    feats = [f for f in X_train.columns if f not in excluded_feat]
    if normalize_opt is not None:
        if normalize_opt == 'min_max':
            scaler = preprocessing.MinMaxScaler()
        scaler = scaler.fit(X_train[feats])
        X_train[feats] = scaler.transform(X_train[feats])
        X_valid[feats] = scaler.transform(X_valid[feats])
        X_test[feats] = scaler.transform(X_test[feats])
    return X_train, X_valid, X_test


kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=1337)
kf = list(kf.split(np.arange(len(train_sample))))

oof_final = np.zeros(len(train_sample))
sub_final = np.zeros(len(submission))
i = 0
while i < NB_MODELS:
    print('Running Model ', i+1)

    oof = np.zeros(len(train_sample))
    prediction = np.zeros(len(submission))

    for _, (train_index, valid_index) in enumerate(kf):

        train_x = train_sample.iloc[train_index]
        train_y = targets.iloc[train_index]

        valid_x = train_sample.iloc[valid_index]
        valid_y = targets.iloc[valid_index]

        # #apply min max scaler on training, validation data
        train_x, valid_x, test_scaled = normalize(train_x.copy(),
                                                  valid_x.copy(),
                                                  test.copy(), 'min_max', [])

        train_x = train_x.values.reshape(train_x.shape[0], 1, train_x.shape[1])
        valid_x = valid_x.values.reshape(valid_x.shape[0], 1, valid_x.shape[1])
        test_scaled = test_scaled.values.reshape(test_scaled.shape[0],
                                                 1, test_scaled.shape[1])

        model = get_model()
        # cb_checkpoint = ModelCheckpoint("model.hdf5",
        #                                 monitor='val_mae',
        #                                 save_weights_only=True,
        #                                 save_best_only=True)

        model.fit(train_x, train_y,
                  epochs=EPOCHS,
                  batch_size=BATCH_SIZE, verbose=0,
                  validation_data=(valid_x, [valid_y]))

        # model.load_model("model.hdf5")
        oof[valid_index] += model.predict(valid_x).ravel()
        prediction += model.predict(test_scaled).ravel()/K_FOLD

    # Obtain the MAE for this run.
    model_score = mse(targets, oof, squared=False)/1e6
    i += 1

    if model_score < 2.77:
        print(f"MAE: {model_score} averaged")
        print(model_score)
        oof_final += oof/NB_MODELS
        sub_final += prediction/NB_MODELS
    else:
        print(f"MAE: {model_score} not averaged")
        print(model_score)

print(f"\nMAE for NN: {mse(targets, oof_final, squared=False):.0f}")
submission['time_to_eruption'] = sub_final
submission.to_csv(PATH_DATA + 'submission.csv', index=False)
