from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, DropoutWrapper
from esim_model.utils import print_shape


class ESIM(object):
    def __init__(self, seq_length, n_vocab, embedding_size,
                 hidden_size, attention_size, n_classes,
                 batch_size, learning_rate, optimizer,
                 l2, clip_value):
        self._parameter_init(seq_length, n_vocab, embedding_size,
                             hidden_size, attention_size, n_classes,
                             batch_size, learning_rate, optimizer,
                             l2, clip_value)
        self._placeholder_init()

        self.logits = self._logits_op()
        self.loss = self._loss_op()
        self.acc = self._acc_op()
        self.train = self._training_op()

        tf.add_to_collection('train_mini', self.train)

    def _parameter_init(self, seq_length, n_vocab, embedding_size,
                        hidden_size, attention_size, n_classes,
                        batch_size, learning_rate, optimizer,
                        l2, clip_value):
        self.seq_length = seq_length
        self.n_vocab = n_vocab
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.attention_size = attention_size
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.l2 = l2
        self.clip_value = clip_value

    def _placeholder_init(self):
        self.premise = tf.placeholder(tf.int32, [None, self.seq_length], 'premise')
        self.hypothesis = tf.placeholder(tf.int32, [None, self.seq_length], 'hypothesis')
        self.y = tf.placeholder(tf.float32, [None, self.n_classes], 'y_true')
        self.premise_mask = tf.placeholder(tf.int32, [None,], 'premise_actual_length')
        self.hypothesis_mask = tf.placeholder(tf.int32, [None,], 'hypothesis_actual_length')
        self.embed_matrix = tf.Variable(tf.random_uniform([self.n_vocab, self.embedding_size], -1.0, 1.0), name="embed_matrix")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _logits_op(self):
        a_bar, b_bar = self._inputEncodingBlock('input_encoding')
        m_a, m_b = self._localInferenceBlock(a_bar, b_bar, 'local_inference')
        logits = self._compositionBlock(m_a, m_b, self.hidden_size, 'composition')
        return logits

    def _feedForwardBlock(self, inputs, hidden_dims, num_units,
                          scope, isReuse=False, initializer=None):
        with tf.variable_scope(scope, reuse=isReuse):
            if initializer is None:
                initializer = tf.random_normal_initializer(0.0, 0.1)

            with tf.variable_scope('feed_forward_layer1'):
                inputs = tf.nn.dropout(inputs, self.dropout_keep_prob)
                outputs = tf.layers.dense(inputs, hidden_dims, tf.nn.relu, kernel_initializer=initializer)
            with tf.variable_scope('feed_forward_layer2'):
                outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
                results = tf.layers.dense(outputs, num_units, tf.nn.tanh, kernel_initializer=initializer)
                return results

    def _biLSTMBlock(self, inputs, num_units, scope,
                     seq_len=None, isReuse=False):
        with tf.variable_scope(scope, reuse=isReuse):
            lstmCell = LSTMCell(num_units=num_units)
            dropLSTMCell = lambda: DropoutWrapper(lstmCell, output_keep_prob=self.dropout_keep_prob)
            fwLSTMCell, bwLSTMCell = dropLSTMCell(), dropLSTMCell()
            output = tf.nn.bidirectional_dynamic_rnn(cell_fw=fwLSTMCell,
                                                     cell_bw=bwLSTMCell,
                                                     inputs=inputs,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)
            return output

    def _inputEncodingBlock(self, scope):
        with tf.device('/cpu:0'):
            self.Embedding = tf.get_variable('Embedding', [self.n_vocab, self.embedding_size], tf.float32)
            self.embeded_left = tf.nn.embedding_lookup(self.Embedding, self.premise)
            self.embeded_right = tf.nn.embedding_lookup(self.Embedding, self.hypothesis)
            print_shape('embeded_left', self.embeded_left)
            print_shape('embeded_right', self.embeded_right)

        with tf.variable_scope(scope):
            outputsPremise, finalStatePremise = self._biLSTMBlock(self.embeded_left,
                                                                  self.hidden_size,
                                                                  'biLSTM',
                                                                  self.premise_mask)
            outputsHypothesis, finalStateHypothesis = self._biLSTMBlock(self.embeded_right,
                                                                        self.hidden_size,
                                                                        'biLSTM',
                                                                        self.hypothesis_mask,
                                                                        isReuse=True)
            a_bar = tf.concat(outputsPremise, axis=2)
            b_bar = tf.concat(outputsHypothesis, axis=2)
            print_shape('a_bar', a_bar)
            print_shape('b_bar', b_bar)
            return a_bar, b_bar

    def _localInferenceBlock(self, a_bar, b_bar, scope):
        with tf.variable_scope(scope):
            attentionWeights = tf.matmul(a_bar, tf.transpose(b_bar, [0, 2, 1]))
            print_shape('att_wei', attentionWeights)

            attentionSoft_a = tf.nn.softmax(attentionWeights)
            attentionSoft_b = tf.nn.softmax(tf.transpose(attentionWeights))
            attentionSoft_b = tf.transpose(attentionSoft_b)
            print_shape('att_soft_a', attentionSoft_a)
            print_shape('att_soft_b', attentionSoft_b)

            a_hat = tf.matmul(attentionSoft_a, b_bar)
            b_hat = tf.matmul(attentionSoft_b, a_bar)
            print_shape('a_hat', a_hat)
            print_shape('b_hat', b_hat)

            a_diff = tf.subtract(a_bar, a_hat)
            a_mul = tf.multiply(a_bar, a_hat)
            print_shape('a_diff', a_diff)
            print_shape('a_mul', a_mul)

            b_diff = tf.subtract(b_bar, b_hat)
            b_mul = tf.multiply(b_bar, b_hat)
            print_shape('b_diff', b_diff)
            print_shape('a_mul', b_mul)

            m_a = tf.concat([a_bar, a_hat, a_diff, a_mul], axis=2)
            m_b = tf.concat([b_bar, b_hat, b_diff, b_mul], axis=2)
            print_shape('m_a', m_a)
            print_shape('m_b', m_b)
            return m_a, m_b

    def _compositionBlock(self, m_a, m_b, hiddenSize, scope):
        outputV_a, finalStateV_a = self._biLSTMBlock(m_a, hiddenSize, 'biLSTM')
        outputV_b, finalStateV_b = self._biLSTMBlock(m_b, hiddenSize, 'biLSTM', isReuse=True)
        v_a = tf.concat(outputV_a, axis=2)
        v_b = tf.concat(outputV_b, axis=2)

        print_shape('v_a', v_a)
        print_shape('v_b', v_b)

        v_a_avg = tf.reduce_mean(v_a, axis=1)
        v_b_avg = tf.reduce_mean(v_b, axis=1)
        v_a_max = tf.reduce_max(v_a, axis=1)
        v_b_max = tf.reduce_max(v_b, axis=1)
        print_shape('v_a_avg', v_a_avg)
        print_shape('v_a_max', v_a_max)
        print_shape('v_b_avg', v_b_avg)
        print_shape('v_b_max', v_b_max)

        v = tf.concat([v_a_avg, v_a_max, v_b_avg, v_b_max], axis=1)
        print_shape('v', v)
        y_hat = self._feedForwardBlock(v, self.hidden_size, self.n_classes, 'feed_forward')
        return y_hat

    def _loss_op(self, l2_lambda=0.0001):
        with tf.name_scope('cost'):
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.logits)
            loss = tf.reduce_mean(losses, name='loss_val')
            weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
            l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * l2_lambda
            loss += l2_loss
        return loss

    def _acc_op(self):
        with tf.name_scope('acc'):
            label_pred = tf.argmax(self.logits, 1, name='label_pred')
            label_true = tf.argmax(self.y, 1, name='label_true')
            correct_pred = tf.equal(tf.cast(label_pred, tf.int32), tf.cast(label_true, tf.int32))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='Accuracy')
        return accuracy

    def _training_op(self):
        with tf.name_scope('training'):
            if self.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            elif self.optimizer == 'rmsprop':
                optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
            elif self.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
            elif self.optimizer == 'sgd':
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
            elif self.optimizer == 'adadelta':
                optimizer = tf.train.AdadeltaOptimizer(self.learning_rate)
            elif self.optimizer == 'adagrad':
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                ValueError('Unknown optimizer : {0}'.format(self.optimizer))
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v))
        return train_op







