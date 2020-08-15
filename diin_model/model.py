import tensorflow as tf
from diin_model import utils
from diin_model.nn import *


class DIINModel(object):
    def __init__(self, config):
        self.config = config
        self.embedding_dim = config.emb_dim
        self.dim = config.hidden_dim
        self.sequence_length = config.seq_length
        self.pred_size = config.n_classes
        self.optimizer = config.optimizer
        self.clip_value = config.clip_value
        self.learning_rate = config.learning_rate
        self.n_vocab = config.n_vocab
        self.embedding_size = config.embedding_size
        self.dropout_keep_rate = config.dropout_keep_prob

    def build(self, premise_x, hypothesis_x, is_train):
        # self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.premise_x = premise_x
        self.hypothesis_x = hypothesis_x
        self.is_train = is_train

        if self.is_train:
            # self.dropout_keep_rate = tf.train.exponential_decay(self.config.keep_rate, self.global_step,
            #                                                     self.config.dropout_decay_step,
            #                                                     self.config.dropout_decay_rate,
            #                                                     staircase=False,
            #                                                     name='dropout_keep_rate')
            self.config.keep_rate = self.dropout_keep_rate
        else:
            self.config.keep_rate = 1.0
        self.is_train = tf.cast(self.is_train, tf.bool)

        ## Fucntion for embedding lookup and dropout at embedding layer
        def emb_drop(E, x):
            emb = tf.nn.embedding_lookup(E, x)
            emb_drop = tf.cond(self.is_train, lambda: tf.nn.dropout(emb, self.dropout_keep_rate), lambda: emb)
            return emb_drop

        # Get lengths of unpadded sentences
        prem_seq_lengths, prem_mask = utils.length(self.premise_x)  # mask [N, L , 1]
        hyp_seq_lengths, hyp_mask = utils.length(self.hypothesis_x)
        self.prem_mask = prem_mask
        self.hyp_mask = hyp_mask

        self.embed_matrix = tf.Variable(tf.random_uniform([self.n_vocab, self.embedding_size], -1.0, 1.0), name="embed_matrix")

        ### Embedding layer ###
        with tf.variable_scope("emb"):
            with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                self.E = tf.Variable(self.embed_matrix, trainable=True)
                premise_in = emb_drop(self.E, self.premise_x)  # P
                hypothesis_in = emb_drop(self.E, self.hypothesis_x)  # H

        with tf.variable_scope("highway") as scope:
            premise_in = highway_network(premise_in, self.config.highway_num_layers, True, wd=self.config.wd,
                                         is_train=self.is_train)
            scope.reuse_variables()
            hypothesis_in = highway_network(hypothesis_in, self.config.highway_num_layers, True, wd=self.config.wd,
                                            is_train=self.is_train)

        with tf.variable_scope("prepro") as scope:
            pre = premise_in
            hyp = hypothesis_in
            for i in range(self.config.self_att_enc_layers):
                with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                    p = self_attention_layer(self.config, self.is_train, pre,
                                             p_mask=prem_mask, scope="{}_layer_self_att_enc".format(i))  # [N, len, dim]
                    h = self_attention_layer(self.config, self.is_train, hyp,
                                             p_mask=hyp_mask, scope="{}_layer_self_att_enc_h".format(i))
                    pre = p
                    hyp = h
                    variable_summaries(p, "p_self_enc_summary_layer_{}".format(i))
                    variable_summaries(h, "h_self_enc_summary_layer_{}".format(i))

        with tf.variable_scope("main") as scope:

            def model_one_side(config, main, support, main_length, support_length, main_mask, support_mask, scope):
                bi_att_mx = bi_attention_mx(config, self.is_train, main, support, p_mask=main_mask,
                                            h_mask=support_mask)  # [N, PL, HL]

                bi_att_mx = tf.cond(self.is_train, lambda: tf.nn.dropout(bi_att_mx, config.keep_rate),
                                    lambda: bi_att_mx)
                out_final = dense_net(config, bi_att_mx, self.is_train)

                return out_final

            premise_final = model_one_side(self.config, p, h, prem_seq_lengths, hyp_seq_lengths, prem_mask, hyp_mask,
                                           scope="premise_as_main")
            f0 = premise_final

        self.logits = linear(f0, self.pred_size, True, bias_start=0.0, scope="logit",
                             squeeze=False, wd=self.config.wd, input_keep_prob=self.config.keep_rate,
                             is_train=self.is_train)

    def build_loss(self, y):
        self.y = y

        self.total_cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits))
        self.acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.arg_max(self.logits, dimension=1), tf.cast(self.y, tf.int64)), tf.float32))

        return self.total_cost

    def train_op(self):
        return self._training_op()

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
        gradients, v = zip(*optimizer.compute_gradients(self.total_cost))
        if self.clip_value is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
        train_op = optimizer.apply_gradients(zip(gradients, v), global_step=tf.train.get_global_step())
        return train_op


def bi_attention_mx(config, is_train, p, h, p_mask=None, h_mask=None, scope=None):  # [N, L, 2d]
    with tf.variable_scope(scope or "dense_logit_bi_attention"):
        PL = p.get_shape().as_list()[1]
        HL = h.get_shape().as_list()[1]
        p_aug = tf.tile(tf.expand_dims(p, 2), [1, 1, HL, 1])
        h_aug = tf.tile(tf.expand_dims(h, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, HL, 1]), tf.bool), axis=3)
            h_mask_aug = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(h_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            ph_mask = p_mask_aug & h_mask_aug
        ph_mask = None

        h_logits = p_aug * h_aug

        return h_logits


def self_attention(config, is_train, p, p_mask=None, scope=None):  # [N, L, 2d]
    with tf.variable_scope(scope or "self_attention"):
        PL = p.get_shape().as_list()[1]
        dim = p.get_shape().as_list()[-1]
        # HL = tf.shape(h)[1]
        p_aug_1 = tf.tile(tf.expand_dims(p, 2), [1, 1, PL, 1])
        p_aug_2 = tf.tile(tf.expand_dims(p, 1), [1, PL, 1, 1])  # [N, PL, HL, 2d]
        self_mask = None

        if p_mask is None:
            ph_mask = None
        else:
            p_mask_aug_1 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 2), [1, 1, PL, 1]), tf.bool), axis=3)
            p_mask_aug_2 = tf.reduce_any(tf.cast(tf.tile(tf.expand_dims(p_mask, 1), [1, PL, 1, 1]), tf.bool), axis=3)
            self_mask = p_mask_aug_1 & p_mask_aug_2

        h_logits = get_logits([p_aug_1, p_aug_2], None, True, wd=config.wd, mask=self_mask,
                              is_train=is_train, func=config.self_att_logit_func, scope='h_logits')  # [N, PL, HL]
        self_att = softsel(p_aug_2, h_logits)

        return self_att


def self_attention_layer(config, is_train, p, p_mask=None, scope=None):
    with tf.variable_scope(scope or "self_attention_layer"):
        PL = tf.shape(p)[1]
        # HL = tf.shape(h)[1]
        # if config.q2c_att or config.c2q_att:
        self_att = self_attention(config, is_train, p, p_mask=p_mask)

        print("self_att shape")
        print(self_att.get_shape())

        p0 = fuse_gate(config, is_train, p, self_att, scope="self_att_fuse_gate")

        return p0


def dense_net(config, denseAttention, is_train):
    with tf.variable_scope("dense_net"):
        dim = denseAttention.get_shape().as_list()[-1]
        act = tf.nn.relu if config.first_scale_down_layer_relu else None
        fm = tf.contrib.layers.convolution2d(denseAttention,
                                             int(dim * config.dense_net_first_scale_down_ratio),
                                             config.first_scale_down_kernel,
                                             padding="SAME", activation_fn=act)

        fm = dense_net_block(config, fm, config.dense_net_growth_rate,
                             config.dense_net_layers, config.dense_net_kernel_size,
                             is_train, scope="first_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate,
                                        scope='second_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train, scope="second_dense_net_block")
        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate,
                                        scope='third_transition_layer')
        fm = dense_net_block(config, fm, config.dense_net_growth_rate, config.dense_net_layers,
                             config.dense_net_kernel_size, is_train,
                             scope="third_dense_net_block")

        fm = dense_net_transition_layer(config, fm, config.dense_net_transition_rate,
                                        scope='fourth_transition_layer')

        shape_list = fm.get_shape().as_list()
        out_final = tf.reshape(fm, [-1, shape_list[1] * shape_list[2] * shape_list[3]])
        return out_final


def dense_net_block(config, feature_map, growth_rate, layers, kernel_size,
                    is_train, padding="SAME", act=tf.nn.relu, scope=None):
    with tf.variable_scope(scope or "dense_net_block"):
        conv2d = tf.contrib.layers.convolution2d
        dim = feature_map.get_shape().as_list()[-1]

        list_of_features = [feature_map]
        features = feature_map
        for i in range(layers):
            ft = conv2d(features, growth_rate, (kernel_size, kernel_size),
                        padding=padding, activation_fn=act)
            list_of_features.append(ft)
            features = tf.concat(list_of_features, axis=3)

        return features


def dense_net_transition_layer(config, feature_map, transition_rate, scope=None):
    with tf.variable_scope(scope or "transition_layer"):
        out_dim = int(feature_map.get_shape().as_list()[-1] * transition_rate)
        feature_map = tf.contrib.layers.convolution2d(feature_map, out_dim, 1,
                                                      padding="SAME", activation_fn=None)

        feature_map = tf.nn.max_pool(feature_map, [1, 2, 2, 1], [1, 2, 2, 1], "VALID")

        return feature_map




