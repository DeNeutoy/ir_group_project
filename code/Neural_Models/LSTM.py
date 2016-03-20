
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
        size = config.hidden_size
        vocab_size = config.vocab_size

        self.input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self.targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        self.initial_state = self.initial_state = array_ops.zeros(
                    array_ops.pack([self.batch_size, self.num_steps]),
                     dtype=tf.float32).set_shape([None, self.num_steps])

        lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=1.0) # set to 1!

        if is_training and config.keep_prob < 1:
          lstm_cell = rnn_cell.DropoutWrapper(
              lstm_cell, output_keep_prob=config.keep_prob)

        cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

        # TODO: Remove embeddings
        embedding = tf.get_variable("embedding", [vocab_size, size])
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        if is_training and config.keep_prob < 1:
          inputs = tf.nn.dropout(inputs, config.keep_prob)

        inputs = [tf.squeeze(single_input, [1]) for single_input in tf.split(1, num_steps, inputs)]
        output, states = rnn.rnn(cell, inputs, initial_state=self.initial_state)

        output = tf.reshape(tf.concat(1, output), [-1, size])      # pointless???
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = seq2seq.sequence_loss_by_example(
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



def run_epoch(session, m, data, eval_op, verbose=False):     # m = model
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size) - 1) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()
  for step, (x, y) in enumerate(reader.ptb_iterator(data, m.batch_size,
                                                    m.num_steps)):
    cost, state, _ = session.run([m.cost, m.final_state, eval_op],
                                 {m.input_data: x,
                                  m.targets: y,
                                  m.initial_state: state})
    costs += cost
    iters += m.num_steps

    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)



def main(unused_args):

    # TODO: Change data source
    raw_data = reader.ptb_raw_data(FLAGS.data_path)
    train_data, valid_data, test_data, _ = raw_data

    # TODO: change config so batch size is fixed to 20(number of stations) or change model to accept 3D Tensors
    config = cf.SmallConfig()
    eval_config = cf.SmallConfig()
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                config.init_scale)
    with tf.variable_scope("model", reuse=None, initializer=initializer):
        m = LSTMModel(is_training=True, config=config)

    with tf.variable_scope("model", reuse=True, initializer=initializer):
        mvalid = LSTMModel(is_training=False, config=config)
        mtest = LSTMModel(is_training=False, config=eval_config)

    tf.initialize_all_variables().run()

    for i in range(config.max_max_epoch):

          lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
          session.run(tf.assign(m.lr, config.learning_rate * lr_decay))

          print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
          train_perplexity = run_epoch(session, m, train_data, m.train_op,
                                       verbose=True)
          print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
          valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
          print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

    test_perplexity = run_epoch(session, mtest, test_data, tf.no_op())
    print("Test Perplexity: %.3f" % test_perplexity)

if __name__ == '__main__':
    flags = tf.flags
    logging = tf.logging
    flags.DEFINE_string("data_path", None, "data_path")

    FLAGS = flags.FLAGS

    from tensorflow.python.platform import flags
    from sys import argv

    flags.FLAGS._parse_flags()

    main(argv)