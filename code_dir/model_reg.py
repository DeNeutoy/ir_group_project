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

    X = []
    Y = []

    for case in data:
        week_data = case['next_data']
        for weekday,day_data in enumerate(week_data):
            for hour in range(N_TEMPS+1):

                features = np.ones((N_ZONES,))

                avg_prev_load = np.zeros((N_ZONES,))
                for j_weekday in range(7):
                    avg_prev_load += np.mean(case["prev_data"][j_weekday]['load'],1)/7
                features = np.concatenate((features,avg_prev_load),axis=0)

                #
                # T = day_data['temp'][:,hour]
                # features = np.concatenate((features,T),axis=0)
                # features = np.concatenate((features,T*T),axis=0)

                for iT in range(N_TEMPS):
                    t = np.ones((N_ZONES,))*day_data['temp'][iT,hour]
                    features = np.concatenate((features,t),axis=0)
                    features = np.concatenate((features,t*t),axis=0)


                h = np.ones((N_ZONES,))*hour
                features = np.concatenate((features,h),axis=0)
                features = np.concatenate((features,h*h),axis=0)
                features = np.concatenate((features,h*h*h),axis=0)
                w = np.ones((N_ZONES,))*weekday
                features = np.concatenate((features,w),axis=0)
                features = np.concatenate((features,w*w),axis=0)

                # date = day_data["date"]
                # t = ((date[0]*12+date[1])*30 + date[2])/1000
                # t = np.ones((N_ZONES,))*t
                # features = np.concatenate((features,t),axis=0)

                #features = np.log(features)
                X.append(features)

                y = day_data['load'][:,hour].T
                #y = np.log(y)
                Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    return (X,Y)

def plot(Y,Y_pred,i=0,iZone=0):
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

    # Y_pred_train = np.exp(Y_pred_train)
    # Y_pred_test = np.exp(Y_pred_test)
    # Y = np.exp(Y)
    # Y_test = np.exp(Y_test)
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