# https://github.com/ilivans/tf-rnn-attention

import tensorflow as tf
from tensorflow.contrib.rnn import BasicRNNCell

from tensorflow.python.ops.rnn import dynamic_rnn as rnn
import tensorflow.contrib.slim as slim
import numpy as np


class VanillaRNN:
    def __init__(self, sess, params):
        self.params = params
        self.sess = sess

        # Different placeholders
        self.batch_ph = tf.placeholder(tf.int32, [None, None])
        rnn_inputs = tf.one_hot(self.batch_ph, depth=self.params['vocab'])
        self.init_state = tf.placeholder(shape=[None, self.params['state']], dtype=tf.float32,
                                         name='initial_state')

        cell = BasicRNNCell(num_units=self.params['state'])
        _, self.final_state = rnn(cell, inputs=rnn_inputs, initial_state=self.init_state, dtype=tf.float32)

        # Fully connected layer
        self.y_hat = slim.fully_connected(self.final_state, 1, activation_fn=None,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                     biases_initializer=tf.truncated_normal_initializer())

        self.target_ph = tf.placeholder(tf.float32)
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_hat, labels=self.target_ph))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(self.loss)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(self.y_hat)), self.target_ph), tf.float32))

    def predict(self, sentence):
        sentence = np.array(sentence)
        y_hat = self.sess.run(self.y_hat, feed_dict={self.init_state: self.params['init_state'], self.batch_ph: sentence})
        return y_hat > 0

    def get_next_state(self, state, input, stack=None, use_stack=False):
        curr_input = np.reshape(input,[1,1])
        state = np.reshape(state, [1,self.params['state']])
        next_state = self.sess.run(self.final_state, feed_dict={self.batch_ph: curr_input, self.init_state: state})
        return next_state[0]

    def is_accept(self, state):
        state = np.reshape(state, [1, self.params['state']])
        y_hat = self.sess.run(self.y_hat, feed_dict={self.init_state: state, self.batch_ph: [[]]})
        return y_hat > 0

    def train(self, sentence, label):
        sentence = np.array([sentence])
        acc, loss, _ = self.sess.run([self.accuracy, self.loss, self.optimizer],
                                     feed_dict={self.batch_ph: sentence, self.target_ph: label,
                                                self.init_state: self.params['init_state']})
        return acc, loss

    def test(self, X_test, y_test):
        accuracy_test = 0
        loss_test = 0

        for raw_sentence, label in zip(X_test, y_test):
            sentence = np.array([raw_sentence])
            loss, acc = self.sess.run([self.loss, self.accuracy],
                                      feed_dict={self.batch_ph: sentence, self.target_ph: label,
                                                 self.init_state: self.params['init_state']})
            accuracy_test += acc
            loss_test += loss

        accuracy_test /= len(y_test)
        loss_test /= len(y_test)
        return accuracy_test, loss_test
