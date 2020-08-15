from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.util import nest
import tensorflow as tf
from functools import reduce
from operator import mul
import numpy as np

VERY_BIG_NUMBER = 1e30
VERY_SMALL_NUMBER = 1e-30
VERY_POSITIVE_NUMBER = VERY_BIG_NUMBER
VERY_NEGATIVE_NUMBER = -VERY_BIG_NUMBER


def get_initializer(matrix):
    def _initializer(shape, dtype=None, partition_info=None, **kwargs): return matrix
    return _initializer


def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var


def variable_with_weight_decay(name, shape, stddev, wd):
    """Helper to create an initialized Variable with weight decay.

    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.

    Args:
      name: name of the variable
      shape: list of ints
      stddev: standard deviation of a truncated Gaussian
      wd: add L2Loss weight decay multiplied by this float. If None, weight
          decay is not added for this Variable.

    Returns:
      Variable Tensor
    """
    var = variable_on_cpu(name, shape,
                           tf.truncated_normal_initializer(stddev=stddev))
    if wd:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            print(var.name)
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(0, grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def mask(val, mask, name=None):
    if name is None:
        name = 'mask'
    return tf.multiply(val, tf.cast(mask, 'float'), name=name)


def exp_mask(val, mask, name=None):
    """Give very negative number to unmasked elements in val.
    For example, [-3, -2, 10], [True, True, False] -> [-3, -2, -1e9].
    Typically, this effectively masks in exponential space (e.g. softmax)
    Args:
        val: values to be masked
        mask: masking boolean tensor, same shape as tensor
        name: name for output tensor

    Returns:
        Same shape as val, where some elements are very small (exponentially zero)
    """
    if name is None:
        name = "exp_mask"
    return tf.add(val, (1 - tf.cast(mask, 'float')) * VERY_NEGATIVE_NUMBER, name=name)


def flatten(tensor, keep):
    fixed_shape = tensor.get_shape().as_list()
    start = len(fixed_shape) - keep
    left = reduce(mul, [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start)])
    out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i] for i in range(start, len(fixed_shape))]
    flat = tf.reshape(tensor, out_shape)
    return flat


def reconstruct(tensor, ref, keep):
    ref_shape = ref.get_shape().as_list()
    tensor_shape = tensor.get_shape().as_list()
    ref_stop = len(ref_shape) - keep
    tensor_start = len(tensor_shape) - keep
    pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
    keep_shape = [tensor_shape[i] or tf.shape(tensor)[i] for i in range(tensor_start, len(tensor_shape))]
    # pre_shape = [tf.shape(ref)[i] for i in range(len(ref.get_shape().as_list()[:-keep]))]
    # keep_shape = tensor.get_shape().as_list()[-keep:]
    target_shape = pre_shape + keep_shape
    out = tf.reshape(tensor, target_shape)
    # print("out shape")
    # print(out.get_shape())
    return out


def add_wd(wd, scope=None):
    scope = scope or tf.get_variable_scope().name
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    with tf.name_scope("weight_decay"):
        for var in variables:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name="{}/wd".format(var.op.name))
            tf.add_to_collection('losses', weight_decay)


# def grouper(iterable, n, fillvalue=None, shorten=False, num_groups=None):
#     args = [iter(iterable)] * n
#     out = izip_longest(*args, fillvalue=fillvalue)
#     out = list(out)
#     if num_groups is not None:
#         default = (fillvalue, ) * n
#         assert isinstance(num_groups, int)
#         out = list(each for each, _ in zip_longest(out, range(num_groups), fillvalue=default))
#     if shorten:
#         assert fillvalue is None
#         out = (tuple(e for e in each if e is not None) for each in out)
#     return out

def padded_reshape(tensor, shape, mode='CONSTANT', name=None):
    paddings = [[0, shape[i] - tf.shape(tensor)[i]] for i in range(len(shape))]
    return tf.pad(tensor, paddings, mode=mode, name=name)



def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out


def softsel(target, logits, mask=None, scope=None):
    """

    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out



def linear(args, output_size, bias, bias_start=0.0, scope=None, squeeze=False, wd=0.0, input_keep_prob=1.0,
           is_train=None):
    with tf.variable_scope(scope or "linear"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]

        flat_args = [flatten(arg, 1) for arg in args]
        # if input_keep_prob < 1.0:
        assert is_train is not None
        flat_args = [tf.cond(is_train, lambda: tf.nn.dropout(arg, input_keep_prob), lambda: arg)
                     for arg in flat_args]
        flat_out = _linear(flat_args, output_size, bias)
        out = reconstruct(flat_out, args[0], 1)
        if squeeze:
            out = tf.squeeze(out, [len(args[0].get_shape().as_list()) - 1])
        if wd:
            add_wd(wd)

    return out


def dropout(x, keep_prob, is_train, noise_shape=None, seed=None, name=None):
    with tf.name_scope(name or "dropout"):
        # if keep_prob < 1.0:
        d = tf.nn.dropout(x, keep_prob, noise_shape=noise_shape, seed=seed)
        out = tf.cond(is_train, lambda: d, lambda: x)
        return out
        # return x


def softmax(logits, mask=None, scope=None):
    with tf.name_scope(scope or "Softmax"):
        if mask is not None:
            logits = exp_mask(logits, mask)
        flat_logits = flatten(logits, 1)
        flat_out = tf.nn.softmax(flat_logits)
        out = reconstruct(flat_out, logits, 1)

        return out


def softsel(target, logits, mask=None, scope=None):
    """

    :param target: [ ..., J, d] dtype=float
    :param logits: [ ..., J], dtype=float
    :param mask: [ ..., J], dtype=bool
    :param scope:
    :return: [..., d], dtype=float
    """
    with tf.name_scope(scope or "Softsel"):
        a = softmax(logits, mask=mask)
        target_rank = len(target.get_shape().as_list())
        out = tf.reduce_sum(tf.expand_dims(a, -1) * target, target_rank - 2)
        return out


def double_linear_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0,
                         is_train=None):
    with tf.variable_scope(scope or "Double_Linear_Logits"):
        first = tf.tanh(linear(args, size, bias, bias_start=bias_start, scope='first',
                               wd=wd, input_keep_prob=input_keep_prob, is_train=is_train))
        second = linear(first, 1, bias, bias_start=bias_start, squeeze=True, scope='second',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            second = exp_mask(second, mask)
        return second


def linear_logits(args, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None):
    with tf.variable_scope(scope or "Linear_Logits"):
        logits = linear(args, 1, bias, bias_start=bias_start, squeeze=True, scope='first',
                        wd=wd, input_keep_prob=input_keep_prob, is_train=is_train)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def sum_logits(args, mask=None, name=None):
    with tf.name_scope(name or "sum_logits"):
        if args is None or (nest.is_sequence(args) and not args):
            raise ValueError("`args` must be specified")
        if not nest.is_sequence(args):
            args = [args]
        rank = len(args[0].get_shape())
        logits = sum(tf.reduce_sum(arg, rank - 1) for arg in args)
        if mask is not None:
            logits = exp_mask(logits, mask)
        return logits


def get_logits(args, size, bias, bias_start=0.0, scope=None, mask=None, wd=0.0, input_keep_prob=1.0, is_train=None,
               func=None):
    if func is None:
        func = "sum"
    if func == 'sum':
        return sum_logits(args, mask=mask, name=scope)
    elif func == 'linear':
        return linear_logits(args, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'double':
        return double_linear_logits(args, size, bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                                    input_keep_prob=input_keep_prob,
                                    is_train=is_train)
    elif func == 'dot':
        assert len(args) == 2
        arg = args[0] * args[1]
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'scaled_dot':
        assert len(args) == 2
        dim = args[0].get_shape().as_list()[-1]
        arg = args[0] * args[1]
        arg = arg / tf.sqrt(tf.constant(dim, dtype=tf.float32))
        return sum_logits([arg], mask=mask, name=scope)
    elif func == 'mul_linear':
        assert len(args) == 2
        arg = args[0] * args[1]
        return linear_logits([arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    elif func == 'proj':
        assert len(args) == 2
        d = args[1].get_shape()[-1]
        proj = linear([args[0]], d, False, bias_start=bias_start, scope=scope, wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        return sum_logits([proj * args[1]], mask=mask)
    elif func == 'tri_linear':
        assert len(args) == 2
        new_arg = args[0] * args[1]
        return linear_logits([args[0], args[1], new_arg], bias, bias_start=bias_start, scope=scope, mask=mask, wd=wd,
                             input_keep_prob=input_keep_prob,
                             is_train=is_train)
    else:
        raise Exception()


def highway_layer(arg, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None, output_size=None):
    with tf.variable_scope(scope or "highway_layer"):
        if output_size is not None:
            d = output_size
        else:
            d = arg.get_shape()[-1]
        trans = linear([arg], d, bias, bias_start=bias_start, scope='trans', wd=wd, input_keep_prob=input_keep_prob,
                       is_train=is_train)

        trans = tf.nn.relu(trans)
        gate = linear([arg], d, bias, bias_start=bias_start, scope='gate', wd=wd, input_keep_prob=input_keep_prob,
                      is_train=is_train)
        gate = tf.nn.sigmoid(gate)
        if d != arg.get_shape()[-1]:
            arg = linear([arg], d, bias, bias_start=bias_start, scope='arg_resize', wd=wd,
                         input_keep_prob=input_keep_prob, is_train=is_train)
        out = gate * trans + (1 - gate) * arg
        return out


def highway_network(arg, num_layers, bias, bias_start=0.0, scope=None, wd=0.0, input_keep_prob=1.0, is_train=None,
                    output_size=None):
    with tf.variable_scope(scope or "highway_network"):
        prev = arg
        cur = None
        for layer_idx in range(num_layers):
            cur = highway_layer(prev, bias, bias_start=bias_start, scope="layer_{}".format(layer_idx), wd=wd,
                                input_keep_prob=input_keep_prob, is_train=is_train, output_size=output_size)
            prev = cur
        return cur


def conv1d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv1d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        # if is_train is not None and keep_prob < 1.0:
        in_ = dropout(in_, keep_prob, is_train)
        xxc = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        out = tf.reduce_max(tf.nn.relu(xxc), 2)  # [-1, JX, d]
        return out


def multi_conv1d(in_, filter_sizes, heights, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "multi_conv1d"):
        assert len(filter_sizes) == len(heights)
        outs = []
        for filter_size, height in zip(filter_sizes, heights):
            if filter_size == 0:
                continue
            out = conv1d(in_, filter_size, height, padding, is_train=is_train, keep_prob=keep_prob,
                         scope="conv1d_{}".format(height))
            outs.append(out)
        # concat_out = tf.concat(2, outs)
        concat_out = tf.concat(outs, axis=2)
        return concat_out


def conv2d(in_, filter_size, height, padding, is_train=None, keep_prob=1.0, scope=None):
    with tf.variable_scope(scope or "conv2d"):
        num_channels = in_.get_shape()[-1]
        filter_ = tf.get_variable("filter", shape=[1, height, num_channels, filter_size], dtype='float')
        bias = tf.get_variable("bias", shape=[filter_size], dtype='float')
        strides = [1, 1, 1, 1]
        if is_train is not None and keep_prob < 1.0:
            in_ = dropout(in_, keep_prob, is_train)
        out = tf.nn.conv2d(in_, filter_, strides, padding) + bias  # [N*M, JX, W/filter_stride, d]
        return out


def cosine_similarity(lfs, rhs):  # [N, d]
    dot = tf.reduce_sum(lfs * rhs, axis=1)
    base = tf.sqrt(tf.reduce_sum(tf.square(lfs), axis=1)) * tf.sqrt(tf.reduce_sum(tf.square(rhs), axis=1))
    return dot / base


def variable_summaries(var, scope):
    """summaries for tensors"""
    with tf.name_scope(scope or 'summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def dense_logits(config, args, out_size, bias, bias_start=0.0, scope=None,
                 mask=None, wd=0.0, input_keep_prob=1.0, is_train=None, func=None):
    with tf.variable_scope(scope or "dense_logits"):

        # Tri_linear
        if func == "tri_linear":
            new_arg = args[0] * args[1]
            cat_dim = len(new_arg.get_shape().as_list()) - 1
            cat_args = tf.concat([args[0], args[1], new_arg], axis=cat_dim)
            print("cat args shape")
            print(cat_args.get_shape())

            # if config.dense_logits_with_mask:
            #     out = linear(cat_args, out_size ,True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd, input_keep_prob=config.keep_rate,
            #                             is_train=is_train)
            #     if mask is not None:
            #         mask_exp = tf.cast(tf.tile(tf.expand_dims(mask, 3), [1,1,1, out_size]),tf.float32)
            #         out = mask_exp * out
            #     return out

            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)
        elif func == "mul":
            cat_args = args[0] * args[1]

            out = linear([cat_args], out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "cat_linear":
            cat_dim = len(args[0].get_shape().as_list()) - 1
            cat_args = tf.concat([args[0], args[1]], axis=cat_dim)

            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "diff_mul":
            diff = args[0] - args[1]
            mul = args[0] * args[1]
            cat_dim = len(mul.get_shape().as_list()) - 1
            cat_args = tf.concat([diff, mul], axis=cat_dim)
            out = linear(cat_args, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)

        elif func == "diff":
            diff = args[0] - args[1]
            out = linear(diff, out_size, True, bias_start=0.0, scope="dense_logit_linear", squeeze=False, wd=wd,
                         input_keep_prob=config.keep_rate, is_train=is_train)


        else:
            raise Exception()

        # if config.dense_logits_with_mask:
        #     if mask is not None:
        #         mask_exp = tf.cast(tf.tile(tf.expand_dims(mask, 3), [1,1,1, out_size]),tf.float32)
        #         out = mask_exp * out

        variable_summaries(out, "dense_logits_out_summaries")

        if config.visualize_dense_attention_logits:
            list_of_logits = tf.unstack(out, axis=3)
            for i in range(len(list_of_logits)):
                tf.summary.image("dense_logit_layer_{}".format(i), tf.expand_dims(list_of_logits[i], 3), max_outputs=2)

        return out


def fuse_gate(config, is_train, lhs, rhs, scope=None):
    with tf.variable_scope(scope or "fuse_gate"):
        dim = lhs.get_shape().as_list()[-1]
        # z
        # if config.fuse_gate_KR_1_0:
        #     lhs_1 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        #     rhs_1 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0, input_keep_prob=1.0, is_train=is_train)
        # else:
        lhs_1 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_1", squeeze=False, wd=config.wd,
                       input_keep_prob=config.keep_rate, is_train=is_train)
        rhs_1 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_1", squeeze=False, wd=0.0,
                       input_keep_prob=config.keep_rate, is_train=is_train)
        if config.self_att_fuse_gate_residual_conn and config.self_att_fuse_gate_relu_z:
            z = tf.nn.relu(lhs_1 + rhs_1)
        else:
            z = tf.tanh(lhs_1 + rhs_1)
        # f
        # if config.fuse_gate_KR_1_0:
        #     lhs_2 = linear(lhs, dim ,True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        #     rhs_2 = linear(rhs, dim ,True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=config.wd, input_keep_prob=1.0, is_train=is_train)
        # else:
        lhs_2 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_2", squeeze=False, wd=config.wd,
                       input_keep_prob=config.keep_rate, is_train=is_train)
        rhs_2 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_2", squeeze=False, wd=config.wd,
                       input_keep_prob=config.keep_rate, is_train=is_train)
        f = tf.sigmoid(lhs_2 + rhs_2)

        if config.two_gate_fuse_gate:
            lhs_3 = linear(lhs, dim, True, bias_start=0.0, scope="lhs_3", squeeze=False, wd=config.wd,
                           input_keep_prob=config.keep_rate, is_train=is_train)
            rhs_3 = linear(rhs, dim, True, bias_start=0.0, scope="rhs_3", squeeze=False, wd=config.wd,
                           input_keep_prob=config.keep_rate, is_train=is_train)
            f2 = tf.sigmoid(lhs_3 + rhs_3)
            out = f * lhs + f2 * z
        else:
            out = f * lhs + (1 - f) * z

        return out


