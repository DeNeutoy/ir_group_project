# ir_group_project

Energy Load models
--------------------------------------------
The data is available on a Kaggle competition page:
https://www.kaggle.com/c/global-energy-forecasting-competition-2012-load-forecasting/data

To run Energy Load models:
--  Copy data to data/energy directory
--  run code_dir/preprocess_energy.py
    This preprocesses energy loads data and split the data into training and test sets
--  To run regression model run
    code_dir/model_reg.py
--  To run fully connected NN run
    code/kerasRNN/train1.py
    This uses model defined in code/kerasRNN/model1_2.py
    and saves output files in output/kerasRNN/dense
--  To run RNN+NN model run
    code/kerasRNN/train2.py
    This uses model defined in code/kerasRNN/model2_1.py
    and saves output files in output/kerasRNN/hidden2X
--  To run 2RNN+NN model run (this is experimental as the model converges poorly)
    code/kerasRNN/train2.py
    This uses model defined in code/kerasRNN/model.py
    and saves output files in output/kerasRNN/hidden4X
~

Household reading models
--------------------------------------------
The data is available UCI ML page:
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

To run Household LSTM/GRU/PEEPHOLE model:
-- copy downloaded data(named as "household_power_consumption.txt") to code/LSTM_household directory
-- run LSTM_household/LSTM.py
-- run LSTM_household/GRU.py
-- run LSTM_household/Peephole.py

readers can change the "ratio" parameters to decide the how many percentages of full data set will be used on training and prediction. 
