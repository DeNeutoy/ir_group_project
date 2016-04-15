'''
This model is n-layer non-linear NN.
It has only basic linear features: T, avgLoad_prev, weekday, hour = 33 feeatures
If it finds all the relevant non-linearities it should be like linear regression with non-linear features
This model produces 12/13% accuracy
run with train1
'''

import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from config import *

def get_model(nT_in, nT_out,nFeatures, nHidden,nOutput):

    model = Sequential()
    model.add(Activation(activation='linear', input_shape=(nFeatures,)))
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())

    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())
    model.add(BatchNormalization())
    model.add(Dense(nFeatures, W_regularizer=l2(1e-10)))
    model.add(PReLU())


    model.add(BatchNormalization())
    model.add(Dense(nOutput, W_regularizer=l2(1e-10)))


    adam = Adam(lr=0.01)
    model.compile(optimizer=adam, loss='mse')

    return model


#     - prev_week = 0,1...
#     - next_week = 1,2...
#     - prev_data
#         - weekday = 0...6
#             - date
#             - load
#                 - [zone,hour] = 20x24 np/array
#             - temp
#                 - [zone,hour] = 11x24 np.array
#     - next_data
#         - the same

def model_data(data):

    X = []
    Y = []

    for case in data:
        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_HOURS):

                features = np.ones((1,))

                avg_prev_load = np.zeros((N_ZONES,))
                for j_weekday in range(7):
                    avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7.0
                features = np.concatenate((features,avg_prev_load),axis=0)

                for iT in range(N_TEMPS):
                    t = np.ones((1,))*day_data['temp'][iT,hour]
                    features = np.concatenate((features,t),axis=0)

                h = np.ones((1,))*hour
                features = np.concatenate((features,h),axis=0)
                w = np.ones((1,))*weekday
                features = np.concatenate((features,w),axis=0)

                # date = day_data["date"]
                # t = ((date[0]*12+date[1])*30 + date[2])/1000
                # t = np.ones((N_ZONES,))*t
                # features = np.concatenate((features,t),axis=0)

                #features = np.log(features)
                X.append(features)

                y = day_data['load'][:,hour]
                #y = np.log(y)
                Y.append(y)

    X = np.array(X)
    Y = np.array(Y)  / 1e3

    return (X,Y)