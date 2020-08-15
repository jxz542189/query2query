import tensorflow as tf

files = ["/Users/jxz/PycharmProjects/query2query/eval_data/part-r-00001-2"]
tf.compat.v1.enable_eager_execution()
raw_dataset = tf.data.TFRecordDataset(files)

for raw_record in raw_dataset.take(2):
  example = tf.train.Example()
  example.ParseFromString(raw_record.numpy())
  print(type(example))


