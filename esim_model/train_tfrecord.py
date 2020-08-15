import pandas as pd
import argparse
from esim_model.model_tfrecord import ESIM
from esim_model.utils import *
from esim_model.dataset import *
from sklearn.model_selection import train_test_split


tf.reset_default_graph()
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger.info(basedir)


def args_parser():
  parser = argparse.ArgumentParser()

  # model parameters
  parser.add_argument("--batch_size", default=16, type=int)
  parser.add_argument("--n_vocab", default=84000, type=int)
  parser.add_argument("--word2id_file", type=str, default=os.path.join(basedir, "data", "word2id.txt"))
  parser.add_argument("--embedding_size", type=int, default=64)
  parser.add_argument("--hidden_size", type=int, default=64)
  parser.add_argument("--attention_size", type=int, default=64)
  parser.add_argument("--n_classes", type=int, default=2)
  parser.add_argument("--clip_value", type=int, default=5)

  # data parameters
  parser.add_argument("--model_dir", type=str, default=os.path.join(basedir, "esim_model", "model_dir_tfrecord"))
  parser.add_argument("--valid_dir", type=str, default="/Users/jxz/PycharmProjects/query2query/eval_data")
  parser.add_argument("--train_dir", type=str, default="/Users/jxz/PycharmProjects/query2query/eval_data")

  # training parameters
  parser.add_argument("--learning_rate", type=float, default=5e-5)
  parser.add_argument("--epoch", type=int, default=10)
  parser.add_argument("--shuffle_size", type=int, default=1024 * 10)
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--l2", type=float, default=0.01)
  parser.add_argument("--sentence_len", type=int, default=30)

  # save dir
  parser.add_argument("--dropout_keep_prob", type=float, default=0.3)
  parser.add_argument("--save_model", type=str, default=os.path.join(basedir, "esim_model", "save_model_tfrecord"))
  parser.add_argument("--optimizer", type=str, default='Adam')
  parser.add_argument("--eval_after_sec", type=int, default=100)
  return parser


def create_estimator(config, model_dir):
    return tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=config)


def model_fn(features, labels, mode, params):
    premise = features['premise']
    premise_mask = features['premise_mask']
    hypothesis = features['hypothesis']
    hypothesis_mask = features['hypothesis_mask']
    print("==============1===============")
    print(labels)
    labels_1 = tf.one_hot(labels, args.n_classes)
    print("==============2===============")
    print(labels)

    model = ESIM(args)

    if mode == tf.estimator.ModeKeys.PREDICT:
        logits = model.build(premise, premise_mask, hypothesis, hypothesis_mask, 1)
        predictions = tf.argmax(logits, axis=1)
        predictions = {
            'predictions': predictions,
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    logits = model.build(premise, premise_mask, hypothesis, hypothesis_mask, args.dropout_keep_prob)
    predictions = tf.argmax(logits, axis=1)

    losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_1, logits=logits)
    loss = tf.reduce_mean(losses, name='loss_val')
    weights = [v for v in tf.trainable_variables() if ('w' in v.name) or ('kernel' in v.name)]
    l2_loss = tf.add_n([tf.nn.l2_loss(w) for w in weights]) * args.l2
    loss += l2_loss

    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.contrib.layers.optimize_loss(loss,
                                                   tf.train.get_global_step(),
                                                   optimizer=args.optimizer,
                                                   learning_rate=args.learning_rate,
                                                   summaries=['loss'],
                                                   name="train_op")
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op)
    else:
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
        "premise": tf.placeholder(tf.int64, [None, args.sentence_len], name='premise'),
        "premise_mask": tf.placeholder(tf.int64, [None,], name='premise_mask'),
        "hypothesis": tf.placeholder(tf.int64, [None, args.sentence_len], name='hypothesis'),
        "hypothesis_mask": tf.placeholder(tf.int64, [None, ], name='hypothesis_mask')
    })
    return input_fn


if __name__ == '__main__':
    arg_parser = args_parser()
    args = arg_parser.parse_args()
    word2id = read_word2id(args.word2id_file)

    train_files = get_tfrecord_files(args.train_dir)
    print("files of train_files: {}".format(len(train_files)))
    valid_files = get_tfrecord_files(args.valid_dir)[:2]
    print("files of train_files: {}".format(len(valid_files)))

    SCHEMA = {
        'premise': tf.io.FixedLenFeature([args.sentence_len], tf.int64),
        'premise_mask': tf.io.FixedLenFeature((), tf.int64),
        'hypothesis': tf.io.FixedLenFeature([args.sentence_len], tf.int64),
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
            use_gzip=True
        ),
        max_steps=10000000
    )

    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(
            valid_files,
            valid_parser,
            args.batch_size,
            epoch=None,
            use_gzip=True
        ),
        steps=100,
        throttle_secs=args.eval_after_sec
    )

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    if args.save_model:
        estimator.export_saved_model(args.save_model,
                                     export_helper(args))


