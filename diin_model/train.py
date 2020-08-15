import os
import argparse
from diin_model.model import DIINModel
from esim_model.utils import *
from diin_model.dataset import *


tf.reset_default_graph()
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger.info(basedir)


parser = argparse.ArgumentParser()
pa = parser.add_argument

pa("--valid_dir", type=str, default="/Users/jxz/PycharmProjects/query2query/eval_data")
pa("--train_dir", type=str, default="/Users/jxz/PycharmProjects/query2query/eval_data")
pa("--word2id_file", type=str, default=os.path.join(basedir, "data", "word2id.txt"))

pa("--emb_dim", default=64, help="Call if you want to make your word embeddings trainable.")
pa("--hidden_dim", default=100, help="hidden_dim")
pa("--seq_length", type=int, default=30, help="Max sequence length")
pa("--n_classes", type=int, default=2, help="")
pa("--optimizer", default="adam")
pa("--clip_value", default=5)
pa("--n_vocab", default=84000, help="n_tokens")
pa("--learning_rate", type=float, default=0.0005, help="Learning rate for model")
pa("--save_model", type=str, default=os.path.join(basedir, "diin_model", "save_model_tfrecord"))
pa("--steps", type=int, default=100)
pa("--model_dir", type=str, default=os.path.join(basedir, "diin_model", "model_dir_tfrecord"))
pa("--keep_rate", type=float, default=1.0, help="Keep rate for dropout in the model")
pa("--dropout_decay_step",  type=int, default=10000, help='dropout_decay_step') ##
pa("--dropout_decay_rate",  type=float, default=0.977, help='dropout_decay_rate') ##
pa("--dropout_keep_prob", type=float, default=0.3)
pa("--eval_after_sec", type=int, default=300)
pa("--start_delay_secs", type=int, default=100)
pa("--epoch", type=int, default=10)
pa("--shuffle_size", type=int, default=1024 * 10)
pa("--embedding_size", type=int, default=64)

pa("--input_keep_rate", type=float, default=0.8, help='keep rate for embedding')
pa("--use_input_dropout", action='store_true', help='use input dropout')
pa("--batch_size", default=64, help="batch_size")
pa("--highway_num_layers", default=3, help="highway_num_layers")
pa("--wd", type=float, default=0.0, help='weight decay')
pa("--self_att_enc_layers", type=int, default=1, help='num layers self att enc') ##
pa("--self_att_logit_func", type=str, default="tri_linear", help='logit function')
pa("--self_att_wo_residual_conn", action='store_true', help='self att without residual connection')
pa("--self_att_fuse_gate_residual_conn", action='store_false', help='self att fuse gate residual connection') ##
pa("--self_att_fuse_gate_relu_z", action='store_true', help='relu instead of tanh')
pa("--conv_fuse_gate_out_origx_base", action='store_true', help='conv_fuse_gate_out_origx_base')
pa("--conv_fuse_gate_out_newx_base", action='store_true', help='conv_fuse_gate_out_newx_base')
pa("--cross_att_fuse_gate_residual_conn", action='store_true', help='cross att fuse gate residual connection')
pa("--two_gate_fuse_gate", action='store_false', help='inside fuse gate we have two f gates') ##
pa("--transitioning_conv_blocks", action='store_true', help='transitioning conv blocks')
pa("--use_dense_net", action='store_false', help='use dense net') ##
pa("--dense_net_growth_rate", type=int, default=20, help='dense net growth rate') ##
pa("--first_transition_growth_rate", type=int, default=2, help='first_transition_growth_rate')
pa("--dense_net_layers", type=int, default=8, help='dense net layers') ##
pa("--dense_net_bottleneck_size", type=int, default=500, help='dense net bottleneck size')
pa("--dense_net_transition_rate", type=float, default=0.5, help='dense_net_transition_rate') ##
pa("--dense_net_transition_layer_max_pooling", action='store_false', help='dense net transition layer max pooling') ##
pa("--dense_net_wo_bottleneck", action='store_false', help='dense net without bottleneck') ##
pa("--dense_net_act_before_conv", action='store_true', help='dense_net_act_before_conv')
pa("--dense_net_kernel_size", default=3, help='dense net kernel size')
pa("--rm_first_transition_layer", action='store_true', help='rm_first_transition_layer')
pa("--first_scale_down_layer", action='store_false', help='first_scale_down_layer') ##
pa("--first_scale_down_layer_relu", action='store_true', help='first_scale_down_layer_relu')
pa("--first_scale_down_kernel", type=int, default=1, help='first_scale_down_kernel') ##

pa("--dense_net_first_scale_down_ratio", type=float, default=0.3, help='dense_net_first_scale_down_ratio') ##


def create_estimator(config, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config)


def model_fn(features, labels, mode, params):
    premise = features['premise']
    hypothesis = features['hypothesis']
    print("==============1===============")
    print(labels)

    model = DIINModel(args)

    if mode == tf.estimator.ModeKeys.PREDICT:
        print("==============predict================")
        print(mode)
        model.build(premise_x=premise, hypothesis_x=hypothesis, is_train=0)
        predictions = tf.argmax(model.logits, axis=1)
        predictions = {
            'predictions': predictions,
            'logits': model.logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    model.build(premise_x=premise, hypothesis_x=hypothesis, is_train=1)
    predictions = tf.argmax(model.logits, axis=1)
    loss = model.build_loss(labels)
    train_op = model.train_op()

    if mode == tf.estimator.ModeKeys.TRAIN:
        print("==============train================")
        print(mode)
        # train_op = tf.contrib.layers.optimize_loss(loss,
        #                                            tf.train.get_global_step(),
        #                                            optimizer=args.optimizer,
        #                                            learning_rate=args.learning_rate,
        #                                            summaries=['loss'],
        #                                            name="train_op")
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    if mode == tf.estimator.ModeKeys.EVAL:
        print("==============eval================")
        print(mode)
        eval_metrics = {
            'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op'),
            'auc': tf.metrics.auc(labels=labels, predictions=predictions, name='auc_op')
        }
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            eval_metric_ops=eval_metrics)


def export_helper(args):
    input_fn = tf.estimator.export.build_raw_serving_input_receiver_fn({
        "premise": tf.placeholder(tf.int64, [None, args.seq_length], name='premise'),
        "hypothesis": tf.placeholder(tf.int64, [None, args.seq_length], name='hypothesis')
    })
    return input_fn


if __name__ == '__main__':
    args = parser.parse_args()
    word2id = read_word2id(args.word2id_file)

    # train_files = get_tfrecord_files(args.train_dir)
    train_files = ["/Users/jxz/PycharmProjects/query2query/eval_data/part-r-00001-2"]
    print("files of train_files: {}".format(len(train_files)))
    # valid_files = get_tfrecord_files(args.valid_dir)[:2]
    valid_files = ["/Users/jxz/PycharmProjects/query2query/eval_data/part-r-00001-2"]
    print("files of train_files: {}".format(len(valid_files)))

    SCHEMA = {
        'premise': tf.io.FixedLenFeature([args.seq_length], tf.int64),
        'premise_mask': tf.io.FixedLenFeature((), tf.int64),
        'hypothesis': tf.io.FixedLenFeature([args.seq_length], tf.int64),
        'hypothesis_mask': tf.io.FixedLenFeature((), tf.int64),
        'label': tf.io.FixedLenFeature((), tf.int64)
    }
    train_parser = data_parser(SCHEMA, is_pred=False)
    valid_parser = data_parser(SCHEMA, is_pred=False)

    session_config = tf.ConfigProto(log_device_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.5

    run_config = tf.estimator.RunConfig(
      keep_checkpoint_max=2, save_checkpoints_steps=args.steps)
    run_config = run_config.replace(session_config=session_config)

    estimator = create_estimator(run_config, args.model_dir)

    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(
            train_files,
            train_parser,
            args.batch_size,
            args.shuffle_size,
            args.epoch,
            use_gzip=False
        ),
        max_steps=1000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            valid_files,
            valid_parser,
            args.batch_size,
            epoch=None,
            use_gzip=False
        ),
        steps=None,
        start_delay_secs=args.start_delay_secs,
        throttle_secs=args.eval_after_sec
    )

    results = tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    print(results)

    if args.save_model:
        estimator.export_saved_model(args.save_model,
                                     export_helper(args))
