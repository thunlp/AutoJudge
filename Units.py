"""
This model contains what we need to build the r-net graph (which are splited into several parts).
"""

import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, MultiRNNCell, BasicLSTMCell
from tensorflow.contrib.layers import variance_scaling_initializer as VSI
from tensorflow.contrib.rnn import LSTMStateTuple


# Gated recurrent units
def gru(state_size):  # num_units-->"dimension"
    gru_cell = GRUCell(num_units=state_size)
    return gru_cell


def lstm(state_size):
    lstm_cell = BasicLSTMCell(num_units=state_size)
    return lstm_cell


# Create a stacked gated recurrent units, with num_layers-layer
def gru_n(state_size, num_layers):
    if num_layers == 1:
        return gru(state_size)
    stacked_gru_cells = MultiRNNCell(
        [gru(state_size) for _ in range(num_layers)])
    return stacked_gru_cells

    # by default state_is_tuple=True
    # return in tuple
    # num_layers * state_size


def lstm_n(state_size, num_layers):
    if num_layers == 1:
        return lstm(state_size)
    stacked_lstm_cells = MultiRNNCell(
        [lstm(state_size) for _ in range(num_layers)])
    return stacked_lstm_cells

    # by default state_is_tuple=True
    # return in tuple
    # num_layers * state_size


class Gated_Attention_Cell(GRUCell):
    def __init__(self, num_units,
                 core,
                 batch_size,
                 attention_mode=False,
                 scope="GA_",
                 input_size=None,
                 activation=tf.tanh):
        super().__init__(num_units=num_units,
                         # input_size=input_size,
                         activation=activation)
        self.core = core  # batch*time*dim
        with tf.variable_scope(scope, reuse=False):
            # define attention parameters
            W1 = tf.get_variable('WuQ', shape=[num_units, num_units], dtype=tf.float32,
                                 initializer=VSI(mode='FAN_AVG'))  # W_u^Q
            W2 = tf.get_variable('WuP', shape=[num_units, num_units], dtype=tf.float32,
                                 initializer=VSI(mode='FAN_AVG'))  # W_u^P
            W3 = tf.get_variable('WvP', shape=[num_units, num_units], dtype=tf.float32,
                                 initializer=VSI(mode='FAN_AVG'))  # W_v^P
            V = tf.get_variable('V', shape=[num_units, 1], dtype=tf.float32, initializer=VSI(mode='FAN_AVG'))  # v
            att_params = {
                'W1': W1, 'W2': W2, 'W3': W3, 'V': V
            }
            # cell=gru(state_size=num_units)
            Wg = tf.get_variable('Wg', shape=[2 * num_units, 2 * num_units], dtype=tf.float32,
                                 initializer=VSI(mode='FAN_AVG'))
        self.param = att_params
        self.Wg = Wg
        self.batch_size = batch_size
        self.attention_mode = attention_mode

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs,  # up{t}  #batch*dim
                 init_state,  # vp{t-1}
                 level=3,
                 scope=None):
        ct, at = attention_pooling(states_a=self.core, states_b_i=inputs, state_c=init_state,
                                   params=self.param, dim=self._num_units, batch_size=self.batch_size)
        if level == 1:  # pass ct
            outputs = super().__call__(ct, init_state)
        else:
            input_ = tf.concat([inputs, ct], axis=1)
            if level == 2:  # pass the concatenation
                outputs = super().__call__(input_, init_state)
            else:
                gt = tf.nn.sigmoid(tf.matmul(input_, self.Wg))
                input_ = gt * input_
                outputs = super().__call__(input_, init_state)  # pass [u,c]*
        # 先计算出ct，然后调用原来本身的__call__
        if not self.attention_mode:
            return (outputs[0], outputs[1])
        else:
            return (at, outputs[1])


def attention_pooling(states_a,
                      states_b_i,
                      state_c,
                      params,
                      dim,
                      batch_size):
    W1, W2, W3 = params['W1'], params['W2'], params['W3']
    # s_ij -> [B,L,d]
    s = tf.tanh(tf.expand_dims(tf.matmul(states_b_i, W2), axis=1) +
                tf.reshape(tf.matmul(tf.reshape(states_a, [-1, dim]), W1), [batch_size, -1, dim]) +
                tf.expand_dims(tf.matmul(state_c, W3), axis=1))  # {s_j^t}_j
    V = params['V']  # [d, 1]
    # e_ij -> softmax(aV_a) : [B, L]
    scores = tf.nn.softmax(
        tf.reshape(tf.matmul(tf.reshape(s, [-1, dim]), V), [batch_size, -1]))  # a_i^t  batch*timesteps
    # c_i -> weighted sum of encoder states
    return (tf.reduce_sum(states_a * tf.expand_dims(scores, axis=-1), axis=1),
            scores)  # [B*t*d]*[B*t*1]-->[B*t*d]对每个进行了加权-->求和[B, d] #c_t


def conv(inputs, embedding_dimension, filter_sizes, num_filters, dropout_keep_probability, batch_size, padding="VALID",
         reduce=tf.reduce_max):
    """
    filter_sizes: a string of format"a1,a2,a3"
    inputs: batch*time*dimension, final representation of all the input
    """
    pooled_outputs = []
    inputs_extended = tf.expand_dims(inputs, -1)
    for i, filter_size in enumerate(filter_sizes.split(",")):
        with tf.name_scope("conv-maxpool-layer-%s" % filter_size):
            size = int(filter_size)
            filter_shape = [size, embedding_dimension, 1, num_filters]
            filters_W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            filters_B = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="B")
            con = tf.nn.conv2d(
                inputs_extended,
                filters_W,
                strides=[1, 1, 1, 1],
                padding=padding,
                name="conv")  # return [batch,height_step,width_step,num_filter] --> [batch,height_step,1,num_filter]
            h = tf.nn.relu(tf.nn.bias_add(con, filters_B), name="relu")
            pooled = reduce(tf.squeeze(h, axis=[2]), axis=1)  # pooling --> [batch,num_filter]
            pooled_outputs.append(pooled)
    pooled_ = tf.concat(pooled_outputs, axis=1)  # batch * all_filter

    with tf.name_scope("dropout"):
        dropped_ = tf.nn.dropout(pooled_, dropout_keep_probability)
    return dropped_  # ,pooled_*(dropout_keep_probability)
