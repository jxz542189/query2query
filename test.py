import tensorflow as tf
from esim_model.dataset import *


def parse_label_column(label_string_tensor):
    table = tf.contrib.lookup.index_table_from_tensor(tf.constant(2))
    return table.lookup(label_string_tensor)


SCHEMA = {
    'premise': tf.io.FixedLenFeature([args.sentence_len], tf.int64),
    'premise_mask': tf.io.FixedLenFeature((), tf.int64),
    'hypothesis': tf.io.FixedLenFeature([args.sentence_len], tf.int64),
    'hypothesis_mask': tf.io.FixedLenFeature((), tf.int64),
    'label': tf.io.FixedLenFeature([1], tf.int64)
}

train_files = get_tfrecord_files("/Users/jxz/PycharmProjects/query2query/train_data")
train_parser = data_parser(SCHEMA, is_pred=False)
valid_parser = data_parser(SCHEMA, is_pred=False)
batch_size = 64
shuffle_size = 10000

dataset = tf.data.TFRecordDataset(files)
# shuffle sample, by default shuffle size is batch_size
dataset = dataset.shuffle(10000) if shuffle_size else dataset.shuffle(
    batch_size * 10)
# repeat dataset
dataset = dataset.repeat(10) if epoch else dataset.repeat()

dataset = dataset.map(
    train_parser, num_parallel_calls=tf.data.experimental.AUTOTUNE)
dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)