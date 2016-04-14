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

#
#
# def center_normalize(x):
#     return (x - K.mean(x)) / K.std(x)

def get_model(nTin, nRNNFeatures, nRNNHidden, \
              nTin2, nRNNFeatures2, nRNNHidden2,
              nFeatures,nOutput):

    model = Graph()
    #model.add_node(Activation(activation='linear', input_shape=(nFeatures,)),   name="X_input")
    model.add_input(input_shape=(nTin,nRNNFeatures),                            name="RNN_input")
    model.add_input(input_shape=(nTin2,nRNNFeatures2),                          name="RNN2_input")
    model.add_input(input_shape=(nFeatures,),                                   name="X_input")

    # model.add_node(BatchNormalization(),                                        name="rnn_norm",       input="RNN_input")
    # model.add_node(LSTM(nRNNHidden, input_shape=(nTin,nRNNFeatures), activation='relu', return_sequences=False), \
    #                name="rnn", input="rnn_norm")

    # model.add_node(SimpleRNN(nRNNHidden, input_shape=(nTin,nRNNFeatures), activation='tanh', return_sequences=False),\
    model.add_node(LSTM(nRNNHidden, input_shape=(nTin,nRNNFeatures), activation='tanh', return_sequences=False), \
                   name="rnn", input="RNN_input")

    # model.add_node(LSTM(nRNNHidden, input_shape=(nTin,nRNNFeatures), activation='tanh', return_sequences=True), \
    #                name="rnn1", input="RNN_input")
    # model.add_node(LSTM(nRNNHidden,                                  activation='tanh', return_sequences=True), \
    #                name="rnn2", input="rnn1")
    # # model.add_node(LSTM(nRNNHidden,                                  activation='tanh', return_sequences=True), \
    # #                name="rnn3", input="rnn2")
    # # model.add_node(LSTM(nRNNHidden,                                  activation='tanh', return_sequences=True), \
    # #                name="rnn4", input="rnn3")
    # model.add_node(LSTM(nRNNHidden,                                  activation='tanh', return_sequences=False), \
    #                name="rnn", input="rnn2")


    # model.add_node(SimpleRNN(nRNNHidden2, input_shape=(nTin2,nRNNFeatures2), activation='tanh', return_sequences=False), \
    model.add_node(LSTM(nRNNHidden2, input_shape=(nTin2,nRNNFeatures2), activation='tanh', return_sequences=False), \
                   name="rnn_2", input="RNN2_input")


    model.add_node(Reshape((nRNNHidden+nRNNHidden2+nFeatures,)), merge_mode='concat',\
                                                                    name='reshape',inputs=["rnn","rnn_2","X_input"])

    nDense = nRNNHidden+nRNNHidden2+nFeatures

    model.add_node(BatchNormalization(),                                        name="norm1",       input="reshape")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense1",      input="norm1")
    model.add_node(PReLU(),                                                     name="relu1",       input="dense1")

    model.add_node(BatchNormalization(),                                        name="norm2",       input="relu1")
    model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense2",      input="norm2")
    model.add_node(PReLU(),                                                     name="relu2",       input="dense2")
    #
    # model.add_node(BatchNormalization(),                                        name="norm3",       input="relu2")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense3",      input="norm3")
    # model.add_node(PReLU(),                                                     name="relu3",       input="dense3")
    #
    # model.add_node(BatchNormalization(),                                        name="norm4",       input="relu3")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense4",      input="norm4")
    # model.add_node(PReLU(),                                                     name="relu4",       input="dense4")
    #
    # model.add_node(BatchNormalization(),                                        name="norm5",       input="relu4")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense5",      input="norm5")
    # model.add_node(PReLU(),                                                     name="relu5",       input="dense5")
    #
    # model.add_node(BatchNormalization(),                                        name="norm6",       input="relu5")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense6",      input="norm6")
    # model.add_node(PReLU(),                                                     name="relu6",       input="dense6")

    # model.add_node(BatchNormalization(),                                        name="norm7",       input="relu6")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense7",      input="norm7")
    # model.add_node(PReLU(),                                                     name="relu7",       input="dense7")
    #
    # model.add_node(BatchNormalization(),                                        name="norm8",       input="relu7")
    # model.add_node(Dense(nDense, W_regularizer=l2(1e-10)),                      name="dense8",      input="norm8")
    # model.add_node(PReLU(),                                                     name="relu8",       input="dense8")


    model.add_node(BatchNormalization(),                                        name="norm_out",       input="relu2")
    model.add_node(Dense(nOutput, W_regularizer=l2(1e-10)),                     name="dense_out",      input="norm_out")
    model.add_output('out',                                                     input='dense_out')


    adam = Adam(lr=0.01)
    model.compile(optimizer=adam, loss={"out":'mse'})

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
    RNN = []
    RNN2 = []

    for case in data:
        week_data = case['prev_data']

        # RNN_prev_week = []
        # for day_data in week_data:
        #     for hour in range(N_HOURS):
        #         load = day_data["load"][:,hour].T
        #         RNN_prev_week.append(load)

        # RNN_prev_week = np.zeros((1,N_ZONES))
        # for day_data in week_data:
        #     for hour in range(N_HOURS):
        #         load = day_data["load"][:,hour].T
        #         RNN_prev_week += load/7.0/24.0

        RNN_prev_week = []
        for day_data in week_data:
            avg =  np.zeros((N_ZONES,))
            for hour in range(N_HOURS):
                load = day_data["load"][:,hour]
                avg += load/7.0

            avgT = np.zeros((N_TEMPS,))
            for hour in range(N_HOURS):
                temp = day_data["temp"][:,hour]
                avgT += temp/7.0
            RNN_day = np.concatenate((avg,avgT),axis=0)
            RNN_prev_week.append(RNN_day)

        RNN_prev_week = np.array(RNN_prev_week, dtype=np.float32)

        RNN2_prev_week = []
        for hour in range(N_HOURS):
            avg =  np.zeros((N_ZONES,))
            for day_data in week_data:
                load = day_data["load"][:,hour]
                avg += load/float(N_HOURS)

            avgT = np.zeros((N_TEMPS,))
            for day_data in week_data:
                temp = day_data["temp"][:,hour]
                avgT += temp/float(N_HOURS)
            RNN2_day = np.concatenate((avg,avgT),axis=0)
            RNN2_prev_week.append(RNN2_day)

        RNN2_prev_week = np.array(RNN2_prev_week, dtype=np.float32)


        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_HOURS):

                features = np.ones((1,))

                # avg_prev_load = np.zeros((N_ZONES,))
                # for j_weekday in range(7):
                #     avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7.0
                # features = np.concatenate((features,avg_prev_load),axis=0)

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

                RNN.append(RNN_prev_week)
                RNN2.append(RNN2_prev_week)



    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)  / 1e3
    RNN = np.array(RNN, dtype=np.float32)  / 1e3
    RNN2 = np.array(RNN2, dtype=np.float32)  / 1e3
    return {"RNN_input":RNN,"RNN2_input":RNN2, "X_input":X, "out":Y}