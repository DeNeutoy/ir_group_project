import time
import csv
import os
import numpy as np
from dateutil import rrule
from datetime import datetime, timedelta
import calendar
from config_energy import *
import pickle

load_file = "data/energy/Load_history.csv"
temp_file = "data/energy/temperature_history.csv"
preprocess_dir = "data/energy/preprocessNN"
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

    return week_to_data, missing_weeks


def create_chunks(week_to_data, missing_weeks):

    continuous_chunks = []
    this_chunk = None
    week = 0
    while week < len(week_to_data)-1:

        if (not week in missing_weeks):

            this_week = week_to_data[week]
            # extract daily loads from this week
            daily_loads = [this_week[day]["load"] for day in range(len(this_week))]
            # concatenate daily loads together to form a (20,24*7) array
            all_week_data = reduce(lambda x,y: np.concatenate([x,y],axis=1),daily_loads[1:], daily_loads[0])

            # append to what is already in this_chunk, or begin a new chunk
            if this_chunk is None:
                this_chunk = all_week_data

            else:

                    this_chunk = np.concatenate([this_chunk, all_week_data], axis=1)
        # if this week is a missing week, do nothing except append the current chunk to output and refresh
        else:
            if this_chunk is not None:
                continuous_chunks.append(this_chunk)
            this_chunk = None

        week += 1

    return continuous_chunks


def generate_dataset(chunks, train_size, predict_size):
    """train_size: size in hours of training data for NN
       predict_size: size in hours of data to predict for NN
    """
    X = []
    Y = []
    for chunk in chunks:
        chunk_length = chunk.shape[1]
        for i in range(chunk_length - (train_size + predict_size) -1):

            thisX = chunk[:,i:i+train_size]
            thisY = chunk[:,i+train_size:i+train_size+predict_size]
            X.append(thisX)
            Y.append(thisY)


    X = np.array(X)
    Y = np.array(Y)
    return [X, Y]


def data_iterator(data_path):

    data = pickle.load(open(data_path, "rb"))
    X,Y = data

    for i in range(X.shape[1]):
        yield (X[i], Y[i])


# TODO: write a splitter for train/val/test

def preprocess(input_size, output_size, save_data_path):
    # read data by date
    date_to_data = read_data()
    # group data by week
    week_to_data, missing_weeks = group_by_week(date_to_data)
    # create continuous chunks which split at missing weeks
    continuous_chunks = create_chunks(week_to_data, missing_weeks)
    # create individual data instances for training NN
    data = generate_dataset(continuous_chunks, input_size, output_size)
    # save instances
    pickle.dump(data, open(save_data_path, "wb"))

if __name__ == "__main__":

    # test to check preprocessing works
    os.chdir("../")
    preprocess(24*7,24,"data/energy/preprocess/NN_data.pkl")