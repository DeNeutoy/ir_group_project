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


Household Load Forecasting
--------------------------------------------------
Data is available here:
https://archive.ics.uci.edu/ml/datasets/Individual+household+electric+power+consumption

To run the various models, simply run:

-- GRU.py
-- LSTM.py
-- Peephole.py

The custom layer we implemented using the Keras RNN interface can be found in code_keras/kerasRNN/PeepholeLayer.py