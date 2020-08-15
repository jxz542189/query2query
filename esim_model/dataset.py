import os
import tensorflow as tf


def get_tfrecord_files(dir_list, sep=","):
    dirs = dir_list.strip().split(sep)
    rlt = []

    files = tf.io.gfile.listdir(dir_list)
    rlt.extend([
        os.path.join(dir_list, f)
        for f in files
        if f.endswith(".tfrecord") or f.endswith(".gz")
        ])
    return rlt


def input_fn(files,
             parse_fn,
             batch_size=64,
             shuffle_size=None,
             epoch=None,
             use_gzip=None):
    dataset = tf.data.TFRecordDataset(
        files,
        compression_type="GZIP") if use_gzip else tf.data.TFRecordDataset(files)
    # shuffle sample, by default shuffle size is batch_size
    dataset = dataset.shuffle(shuffle_size) if shuffle_size else dataset.shuffle(
        batch_size * 10)
    # repeat dataset
    dataset = dataset.repeat(epoch) if epoch else dataset.repeat()

    dataset = dataset.map(
        parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset


def data_parser(schema, is_pred=False):
    def __parser(example_proto):
        features = tf.io.parse_single_example(example_proto, schema)
        train_features = {
            "premise": features["premise"],
            "premise_mask": features["premise_mask"],
            "hypothesis": features["hypothesis"],
            "hypothesis_mask": features["hypothesis_mask"]
        }
        if is_pred:
            return train_features
        label = features["label"]
        return train_features,  label
    return __parser