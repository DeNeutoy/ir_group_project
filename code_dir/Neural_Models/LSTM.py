
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow.python.platform

import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import seq2seq
from tensorflow.models.rnn.ptb import reader
from tensorflow.models.rnn import rnn_cell, rnn
import code.Neural_Models.config as cf
from tensorflow.python.ops import array_ops


class LSTMModel(object):

    def __init__(self, is_training, config):

        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.input_dim = input_dim = config.input_dim
        self.output_dim = output_dim = config.output_dim
        size = config.hidden_size
        vocab_size = config.vocab_size

        # TODO Change number of steps here to be different for the input/output
        self.input_data = tf.placeholder(tf.int32, [batch_size, input_dim])
        self.targets = tf.placeholder(tf.int32, [batch_size, output_dim])

        self.initial_state = array_ops.zeros(
                    array_ops.pack([self.batch_size, self.input_dim]),
                     dtype=tf.float32).set_shape([None, self.input_dim])

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0) # set to 1!

        if is_training and config.keep_prob < 1:
          lstm_cell = rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=config.keep_prob)

        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)
        inputs = self.input_data

        if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, input_dim, inputs)]
        output, states = rnn.rnn(cell, inputs, initial_state=self.initial_state, dtype=tf.float32)

        output = tf.reshape(tf.concat(1, output), [-1, size])      # pointless???
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = seq2seq.sequence_loss_by_example(  # TODO change number of steps here, what to do about softmax?
                [logits],[tf.reshape(self.targets, [-1])],[tf.ones([batch_size * num_steps])],vocab_size)

        self.cost = cost = tf.reduce_sum(loss) / batch_size
        self.final_state = states[-1]

        if is_training:

            self.lr = tf.Variable(0.0, trainable=False)
            tvars = tf.trainable_variables()                # all params inside model which are trainable
            grads, _ = tf.clip_by_global_norm(
                    tf.gradients(cost, tvars), config.max_grad_norm)

            optimizer = tf.train.GradientDescentOptimizer(self.lr)
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))



def main(unused_args):
    pass


if __name__ == '__main__':
    flags = tf.flags
    logging = tf.logging
    flags.DEFINE_string("data_path", None, "data_path")

    FLAGS = flags.FLAGS

    from tensorflow.python.platform import flags
    from sys import argv

    flags.FLAGS._parse_flags()

    main(argv)