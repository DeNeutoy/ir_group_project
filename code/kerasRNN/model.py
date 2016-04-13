import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from config import *

#
#
# def center_normalize(x):
#     return (x - K.mean(x)) / K.std(x)

def get_model(nT_in, nT_out,nFeatures, nHidden,nOutput):

    # model = Sequential()
    # model.add(SimpleRNN(nHidden, input_shape=(nT_in,nFeatures), activation='tanh', return_sequences=False))
    # # model.add(LSTM(nT_out*nFeatures+ nHidden, input_shape=(nT_in,nFeatures), activation='tanh', return_sequences=False))
    # # model.add(Dropout(.2))
    # model.add(Dense(nT_out*nFeatures))
    # model.add(Dense(nT_out*nFeatures))
    # model.add(Dense(nT_out*nFeatures))
    # model.add(Dense(nT_out*nFeatures))
    # model.add(Reshape((nT_out,nFeatures)))
    # # model.add(Activation('linear'))
    # # model.add(SimpleRNN(16))

    model = Sequential()
    model.add(Activation(activation='linear', input_shape=(nFeatures,)))
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


#
#
# def model_data(data):
#
#     X = []
#     Y = []
#
#     for case in data:
#         week_data = case['next_data']
#         for weekday,day_data in enumerate(week_data):
#             for hour in range(N_TEMPS+1):
#
#                 features = np.ones((N_ZONES,))
#
#                 avg_prev_load = np.zeros((N_ZONES,))
#                 for j_weekday in range(7):
#                     avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7.0
#                 features = np.concatenate((features,avg_prev_load),axis=0)
#
#                 #
#                 # T = day_data['temp'][:,hour]
#                 # features = np.concatenate((features,T),axis=0)
#                 # features = np.concatenate((features,T*T),axis=0)
#
#                 for iT in range(N_TEMPS):
#                     t = np.ones((N_ZONES,))*day_data['temp'][iT,hour]
#                     features = np.concatenate((features,t),axis=0)
#                     features = np.concatenate((features,t*t),axis=0)
#
#
#
#                 h = np.ones((N_ZONES,))*hour
#                 features = np.concatenate((features,h),axis=0)
#                 features = np.concatenate((features,h*h),axis=0)
#                 features = np.concatenate((features,h*h*h),axis=0)
#                 w = np.ones((N_ZONES,))*weekday
#                 features = np.concatenate((features,w),axis=0)
#                 features = np.concatenate((features,w*w),axis=0)
#
#                 # date = day_data["date"]
#                 # t = ((date[0]*12+date[1])*30 + date[2])/1000
#                 # t = np.ones((N_ZONES,))*t
#                 # features = np.concatenate((features,t),axis=0)
#
#                 #features = np.log(features)
#                 X.append(features)
#
#                 y = day_data['load'][:,hour]
#                 #y = np.log(y)
#                 Y.append(y)
#
#     X = np.array(X)
#     Y = np.array(Y)
#     return (X,Y)



def model_data(data):

    X = []
    Y = []

    for case in data:
        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_TEMPS+1):

                features = np.ones((1,))

                avg_prev_load = np.zeros((N_ZONES,))
                for j_weekday in range(7):
                    avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7.0
                features = np.concatenate((features,avg_prev_load),axis=0)

                for iT in range(N_TEMPS):
                    t = np.ones((1,))*day_data['temp'][iT,hour]
                    features = np.concatenate((features,t),axis=0)
                    features = np.concatenate((features,t*t),axis=0)



                h = np.ones((1,))*hour
                features = np.concatenate((features,h),axis=0)
                features = np.concatenate((features,h*h),axis=0)
                features = np.concatenate((features,h*h*h),axis=0)
                w = np.ones((1,))*weekday
                features = np.concatenate((features,w),axis=0)
                features = np.concatenate((features,w*w),axis=0)

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