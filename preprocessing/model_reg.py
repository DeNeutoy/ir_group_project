'''
This file creates simple non-linear fieatures on the energy datset
and runs regression model on it.
Produces (15,16)% on training and validation sets
'''


import time
import csv
import os
import numpy as np
import scipy as sp
from dateutil import rrule
from datetime import datetime, timedelta
import calendar
import matplotlib.pyplot as plt

from config_energy import *


train_file = "data/energy/preprocess/train.npy"
test_file = "data/energy/preprocess/test.npy"


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

def prepare_data(data):
    '''
    Calculate feature vectors on the energy dataset
    '''

    X = []
    Y = []

    for case in data:
        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_HOURS):

                # bias
                features = np.ones((1,))

                # average load over previous week
                avg_prev_load = np.zeros((N_ZONES,))
                for j_weekday in range(7):
                    avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7
                features = np.concatenate((features,avg_prev_load),axis=0)

                # temperature features for each of the weather stations
                for iT in range(N_TEMPS):
                    t = np.ones((1,))*day_data['temp'][iT,hour]
                    features = np.concatenate((features,t),axis=0)
                    features = np.concatenate((features,t*t),axis=0)

                # hour of the day features
                h = np.ones((1,))*hour
                features = np.concatenate((features,h),axis=0)
                features = np.concatenate((features,h*h),axis=0)
                features = np.concatenate((features,h*h*h),axis=0)

                # day of the week features
                w = np.ones((1,))*weekday
                features = np.concatenate((features,w),axis=0)
                features = np.concatenate((features,w*w),axis=0)

                X.append(features)
                y = day_data['load'][:,hour].T
                Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)

def plot(Y,Y_pred,i=0,iZone=0):
    '''
    This functionplots one week of observed and predicted loads
    '''
    n = N_HOURS*7
    plt.plot(range(n),Y[i*n:(i+1)*n,iZone])
    plt.plot(range(n),Y_pred[i*n:(i+1)*n,iZone])

    for j in range(n):
        print Y[i*n+j,0], Y_pred[i*n+j,0]

    plt.show()

def run():

    # read data
    data_train = np.load(train_file)
    X,Y = prepare_data(data_train)
    data_test = np.load(test_file)
    X_test,Y_test = prepare_data(data_test)
    print("Training on %d examples with %d features" % X.shape)

    # train
    lambda_reg = 1e1
    xy = np.dot(X.T,Y)
    xx = np.dot(X.T,X)
    n,_ = xx.shape
    xx += lambda_reg * np.eye(n)
    xxinv = np.linalg.inv(xx)
    W = np.dot(xxinv,xy)

    # predict
    Y_pred_train = np.dot(W.T,X.T).T
    Y_pred_test  = np.dot(W.T,X_test.T).T

    error_train = np.mean((Y - Y_pred_train) * (Y - Y_pred_train))
    error_test  = np.mean((Y_test - Y_pred_test) * (Y_test - Y_pred_test))
    avg_train = np.mean(Y)
    avg_test = np.mean(Y_test)
    print("Error train/test = %f / %f" % (np.sqrt(error_train)/avg_train, np.sqrt(error_test)/avg_test))

    plot(Y,Y_pred_train,i=0,iZone=0)
    print "End."



if __name__ == "__main__":
    os.chdir("../")
    run()