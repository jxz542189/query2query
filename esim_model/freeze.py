import numpy as np

import tensorflow as tf


def freeze_graph(input_checkpoint, output_graph):
  '''
    :param input_checkpoint:
    :param output_graph: PB模型保存路径
    :return:
    '''
  # checkpoint = tf.train.get_checkpoint_state(model_folder) #检查目录下ckpt文件状态是否可用
  # input_checkpoint = checkpoint.model_checkpoint_path #得ckpt文件路径

  # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
  output_node_names = "prob"
  saver = tf.train.import_meta_graph(
      input_checkpoint + '.meta', clear_devices=True)
  graph = tf.get_default_graph()  # 获得默认的图
  input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图

  with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)  #恢复图并得到数据
    output_graph_def = tf.graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
        sess=sess,
        input_graph_def=input_graph_def,  # 等于:sess.graph_def
        output_node_names=output_node_names.split(","))  # 如果有多个输出节点，以逗号隔开

    with tf.gfile.GFile(output_graph, "wb") as f:  #保存模型
      f.write(output_graph_def.SerializeToString())  #序列化输出
    print("%d ops in the final graph." %
          len(output_graph_def.node))  #得到当前图有几个操作节点

    # for op in graph.get_operations():
    #     print(op.name, op.values())


def save_model2pb(save_model_dir, pb_path, output_node, tags=['serve']):
  with tf.Session(graph=tf.Graph()) as sess, tf.device("/cpu:0"):
    tf.saved_model.loader.load(sess, tags, save_model_dir)
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess=sess,
        input_graph_def=sess.graph_def,
        output_node_names=output_node)

    with tf.gfile.GFile(pb_path, "wb") as f:
      f.write(output_graph_def.SerializeToString())

    print("%d ops in the final graph." % len(output_graph_def.node))


def get_variable_from_ckpt(checkpoint, variable_name, save_file):

  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, checkpoint)
    variable = sess.run(variable_name)
    np.save(save_file, variable)


def print_variable_from_ckpt(checkpoint):

  with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint + '.meta', clear_devices=True)
    saver.restore(sess, checkpoint)
    for var in tf.global_variables():
      print(var.name)


def load_pb(file):
  with tf.Session() as sess:
    print("load graph...")
    with tf.gfile.GFile(file, 'rb') as f:
      graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    graph_nodes = [n for n in graph_def.node]
    for t in graph_nodes:
      print(t.name)
