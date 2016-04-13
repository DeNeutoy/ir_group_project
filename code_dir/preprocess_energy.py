#
# this files reads data from data/energy and writes it to data/energy/preprocess
# train and test data
#
# format:
# train[i]
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


import time
import csv
import os
import numpy as np
from dateutil import rrule
from datetime import datetime, timedelta
import calendar
from config_energy import *

load_file = "data/energy/Load_history.csv"
temp_file = "data/energy/temperature_history.csv"
preprocess_dir = "data/energy/preprocess"
train_file = preprocess_dir + "/train.npy"
test_file = preprocess_dir + "/test.npy"


def read_data():
    #
    # this reads the data and groups by date
    #
    data_time = {}

    f_data = open(load_file)
    csv_reader = csv.reader(f_data)
    csv_reader.next()
    for line in csv_reader:
        zone = int(line[0])
        year = int(line[1])
        month = int(line[2])
        day = int(line[3])
        date = (year,month,day)

        try:
            load = []
            for x in line[4:]:
                value = float(x.replace(",",""))
                if np.isnan(value):
                    raise RuntimeError("Cannot cast input to float")
                load.append(value)
        except Exception as e:
            continue
        if not date in data_time:
            data_time[date] = {}
        if not 'load' in data_time[date]:
            data_time[date]['load'] = np.empty((N_ZONES,N_HOURS))
        if not len(load) == 24:
            continue
        data_time[(year,month,day)]['load'][zone-1] = load
    f_data.close()

    f_data = open(temp_file)
    csv_reader = csv.reader(f_data)
    csv_reader.next()
    for line in csv_reader:
        zone = int(line[0])
        year = int(line[1])
        month = int(line[2])
        day = int(line[3])
        try:
            temp = []
            for x in line[4:]:
                value = float(x.replace(",",""))
                if np.isnan(value):
                    raise RuntimeError("Cannot cast input to float")
                temp.append(value)
        except Exception as e:
            continue
        date = (year,month,day)
        if not date in data_time:
            data_time[date] = {}
        if not 'temp' in data_time[date]:
            data_time[date]['temp'] = np.empty((N_TEMPS,N_HOURS))
        if not len(temp) == 24:
            continue
        data_time[(year,month,day)]['temp'][zone-1] = temp
    f_data.close()

    return data_time




def group_by_week(date_to_data):
    #
    # this checks the data and groups by week,weekday
    # if data is inconsistent entry for that week is "missing"
    #

    print "="*20, " group by week ", "="*20

    cal= calendar.Calendar(firstweekday=0)
    week_to_data = []
    missing_weeks = []
    start = datetime(2004,1,5)
    end   = datetime(2008,6,30)
    week = 0
    for dt_week_start in rrule.rrule(rrule.WEEKLY, dtstart=start, until=end):

        try:
            this_week_data = []
            for dt in rrule.rrule(rrule.DAILY, dtstart=dt_week_start, until=dt_week_start+timedelta(days=6)):
                date = (dt.year, dt.month, dt.day)

                weekday = calendar.weekday(dt.year, dt.month, dt.day)
                print "Processing : %s, week = %d. weekday = %d" % (dt, week, weekday)
                if not len(this_week_data) == weekday:
                    raise RuntimeError("Week Day mismatch in processing data")
                load = date_to_data[date]['load']
                if not load.shape == (N_ZONES,N_HOURS):
                    raise RuntimeError("Bad input shape")
                temp = date_to_data[date]['temp']
                if not temp.shape == (N_TEMPS,N_HOURS):
                    raise RuntimeError("Bad input shape")
                day_data = {
                    'date':date,
                    'load':load,
                    'temp':temp
                }
                this_week_data.append(day_data)
            if not len(this_week_data) == 7:
                raise RuntimeError("Number of weekdays is wrong in data")
        except Exception as e:
            print e
            missing_weeks.append(week)
            this_week_data = "missing"

        week_to_data.append(this_week_data)
        week +=1

    print "Missing weeks"
    for w in missing_weeks:
        print w

    return week_to_data



def split_data(data, split_ratio = 0.2):
    np.random.seed(12345)
    N = len(data)
    idx = np.arange(N)
    idx_test = np.random.choice(idx, size=int(np.floor(N * split_ratio)),replace=False)
    idx_train = np.setdiff1d(idx, idx_test, assume_unique=True)

    data_train = [data[i] for i in idx_train]
    data_test = [data[i] for i in idx_test]

    return data_train, data_test



def preprocess():

    # read data by date
    date_to_data = read_data()

    # group data by week
    week_to_data = group_by_week(date_to_data)

    # split into week/week examples
    data = []
    for i in range(len(week_to_data)-1):
        if (not week_to_data[i] == "missing") and (not week_to_data[i+1] == "missing"):
            data.append({
                "prev_week":i,
                "next_week":i+1,
                "prev_data":week_to_data[i],
                "next_data":week_to_data[i+1]
            })

    # split into train/dev set
    data_train, data_test = split_data(data, 0.2)

    # save data
    if not os.path.exists(preprocess_dir):
        os.mkdir(preprocess_dir)

    np.save(train_file, data_train)
    np.save(test_file, data_test)
    print "End."


if __name__ == "__main__":
    os.chdir("../")
    preprocess()
