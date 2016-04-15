'''
This model combines RNN with dense layers
Input to RNN is a sequence of 7 average daily loads and temperatures
Input to dense leyers is the output from RNN and a vector of (temperatures, day of the week, hour)
The output of the whole network predicts loads for 20 zones on a particular day of the week and hour.
This model produces 12/13% accuracy
run with train1
'''



import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Activation,Reshape
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras import backend as K

from config import *

def get_model(nTin, nRNNFeatures, nRNNHidden,nFeatures,nOutput):
    '''
    Define RNN+NN model
    '''

    model = Graph()

    model.add_input(input_shape=(nTin,nRNNFeatures),                            name="RNN_input")
    model.add_input(input_shape=(nFeatures,),                                   name="X_input")


    model.add_node(LSTM(nRNNHidden, input_shape=(nTin,nRNNFeatures), activation='tanh', return_sequences=True), \
                   name="rnn1", input="RNN_input")
    model.add_node(LSTM(nRNNHidden,                                  activation='tanh', return_sequences=False), \
                   name="rnn", input="rnn1")


    model.add_node(Reshape((nRNNHidden+nFeatures,)), merge_mode='concat',       name='reshape',inputs=["rnn","X_input"])

    nDense = nRNNHidden+nFeatures

    model.add_node(BatchNormalization(),                                        name="norm1",       input="reshape")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense1",      input="norm1")
    model.add_node(PReLU(),                                                     name="relu1",       input="dense1")

    model.add_node(BatchNormalization(),                                        name="norm2",       input="relu1")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense2",      input="norm2")
    model.add_node(PReLU(),                                                     name="relu2",       input="dense2")

    model.add_node(BatchNormalization(),                                        name="norm3",       input="relu2")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense3",      input="norm3")
    model.add_node(PReLU(),                                                     name="relu3",       input="dense3")

    model.add_node(BatchNormalization(),                                        name="norm4",       input="relu3")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense4",      input="norm4")
    model.add_node(PReLU(),                                                     name="relu4",       input="dense4")

    model.add_node(BatchNormalization(),                                        name="norm_out",       input="relu4")
    model.add_node(Dense(nOutput, W_regularizer=l2(1e-10)),                     name="dense_out",      input="norm_out")
    model.add_output('out',                                                     input='dense_out')


    adam = Adam(lr=0.1)
    model.compile(optimizer=adam, loss={"out":'mse'})

    return model

# data format
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
    '''
    Prepare data
    '''

    X = []
    Y = []
    RNN = []

    for case in data:
        week_data = case['prev_data']

        # calculate daily averages for previous week
        RNN_prev_week = []
        for day_data in week_data:
            avg =  np.zeros((N_ZONES,))
            for hour in range(N_HOURS):
                load = day_data["load"][:,hour]
                avg += load/float(N_HOURS)

            avgT = np.zeros((N_TEMPS,))
            for hour in range(N_HOURS):
                temp = day_data["temp"][:,hour]
                avgT += temp/float(N_HOURS)
            RNN_day = np.concatenate((avg,avgT),axis=0)
            RNN_prev_week.append(RNN_day)

        RNN_prev_week = np.array(RNN_prev_week, dtype=np.float32)

        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_HOURS):

                features = np.ones((1,))

                # temperatures
                for iT in range(N_TEMPS):
                    t = np.ones((1,))*day_data['temp'][iT,hour]
                    features = np.concatenate((features,t),axis=0)

                # hour of the day
                h = np.ones((1,))*hour
                features = np.concatenate((features,h),axis=0)

                # day of the week
                w = np.ones((1,))*weekday
                features = np.concatenate((features,w),axis=0)

                X.append(features)
                y = day_data['load'][:,hour]
                Y.append(y)

                RNN.append(RNN_prev_week)

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)  / 1e3
    RNN = np.array(RNN, dtype=np.float32)  / 1e3
    return {"RNN_input":RNN, "X_input":X, "out":Y}