import tensorflow as tf


def length(sequence):
    populated = tf.sign(tf.abs(sequence))
    length = tf.cast(tf.reduce_sum(populated, axis=1), tf.int32)
    mask = tf.cast(tf.expand_dims(populated, -1), tf.float32)
    return length, mask


def biLSTM(inputs, dim, seq_len, name):
    with tf.name_scope(name):
        with tf.variable_scope('forward' + name):
            lstm_fwd = tf.contrib.rnn.LSTMCell(num_units=dim)
        with tf.variable_scope('backward' + name):
            lstm_bwd = tf.contrib.rnn.LSTMCell(num_units=dim)

        hidden_states, cell_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=lstm_fwd,
                                                                     cell_bw=lstm_bwd,
                                                                     inputs=inputs,
                                                                     sequence_length=seq_len,
                                                                     dtype=tf.float32,
                                                                     scope=name)

    return hidden_states, cell_states


def last_output(output, true_length):
    max_length = int(output.get_shape()[1])
    length_mask = tf.expand_dims(tf.one_hot(true_length-1, max_length,
                                            on_value=1., off_value=0.), -1)
    last_output = tf.reduce_sum(tf.multiply(output, length_mask), 1)
    return last_output


def masked_softmax(scores, mask):
    numerator = tf.exp(tf.subtract(scores, tf.reduce_max(scores, 1,
                                                         keepdims=True))) * mask
    denominator = tf.reduce_sum(numerator, 1, keepdims=True)
    weights = tf.div(numerator, denominator)
    return weights