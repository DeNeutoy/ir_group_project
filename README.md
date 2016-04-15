# ir_group_project

Energy Load models
--------------------------------------------
To run Energy Load models:
--  first run code_dir/preprocess_energy.py
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
