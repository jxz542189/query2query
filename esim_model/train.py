import pandas as pd
from esim_model.model import ESIM
from esim_model.utils import *
from sklearn.model_selection import train_test_split


tf.reset_default_graph()
basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
logger.info(basedir)

tf.flags.DEFINE_integer("batch_size", 16, "Batch Size (default: 64)")

tf.flags.DEFINE_string("word2id_file", os.path.join(basedir, "data", "word2id.txt"), "字典数据")
tf.flags.DEFINE_string("data_file", os.path.join(basedir, "data", "2020-06-12.csv"), "data file")
tf.flags.DEFINE_string("model_dir", os.path.join(basedir, "esim_model", "model_dir"), "model_dir")
tf.flags.DEFINE_string("best_path", os.path.join(basedir, "esim_model", "model_dir", "best_checkpoint"), "best_path")
tf.flags.DEFINE_string("model_file", os.path.join(basedir, "esim_model", "model_dir", "model_file"), "model_file")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 200)")

tf.flags.DEFINE_float("dropout_keep_prob", 0.1, "dropout_keep_prob")

tf.flags.DEFINE_integer("embedding_size", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("eval_batch", 1000, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("early_stop_step", 100000, "Dimensionality of character embedding (default: 128)")

tf.flags.DEFINE_integer("seq_length", 30, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("n_vocab", 83145, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("hidden_size", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("attention_size", 64, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("n_classes", 2, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_integer("clip_value", 5, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("optimizer", 'adam', "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_float("learning_rate", 5e-5, "learning_rate")
tf.flags.DEFINE_float("l2", 0.01, "learning_rate")
tf.flags.DEFINE_float("test_size", 0.1, "learning_rate")


FLAGS = tf.flags.FLAGS

word2id = read_word2id(FLAGS.word2id_file)


def feed_data(model, premise, hypothesis,
              y_batch, dropout_keep_prob,
              is_train=True):
    premise_ids, premise_mask = sentence_to_ids(premise, word_to_ids=word2id, sentence_len=FLAGS.seq_length)
    hypothesis_ids, hypothesis_mask = sentence_to_ids(hypothesis, word_to_ids=word2id, sentence_len=FLAGS.seq_length)
    if is_train:
        batch_labels = convert2onehot(y_batch)
        feed_dict = {
            model.premise: premise_ids,
            model.premise_mask: premise_mask,
            model.hypothesis: hypothesis_ids,
            model.hypothesis_mask: hypothesis_mask,
            model.y: batch_labels,
            model.dropout_keep_prob: dropout_keep_prob
        }
    else:
        feed_dict = {
            model.premise: premise,
            model.premise_mask: premise_mask,
            model.hypothesis: hypothesis,
            model.hypothesis_mask: hypothesis_mask,
            model.dropout_keep_prob: 1
        }
    return feed_dict


def evaluate(sess, model, premise, hypothesis, y):
    data_nums = len(premise)
    batchNums = int(data_nums // int(FLAGS.batch_size))
    total_loss, total_acc = 0.0, 0.0
    for i in range(batchNums):
        batch_premises = premise[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
        batch_hypothesis = hypothesis[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
        batch_labels = y[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
        batch_premises, batch_premises_mask = sentence_to_ids(batch_premises, word_to_ids=word2id, sentence_len=30)
        batch_hypothesis, batch_hypothesis_mask = sentence_to_ids(batch_hypothesis, word_to_ids=word2id, sentence_len=30)
        feed_dict = feed_data(model, batch_premises,
                              batch_hypothesis,
                              batch_labels,
                              dropout_keep_prob=1.0,
                              is_train=True)
        _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
        total_loss += batch_loss
        total_acc += batch_acc
    return total_loss / batchNums, total_acc / batchNums


def train(model):
    logger.info('Loading training and validation data ...')

    saver = tf.train.Saver(max_to_keep=2)
    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)
    logger.info('Configuring TensorBoard and Saver ...')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    total_parameters = count_parameters()
    logger.info('Total trainable parameters : {}'.format(total_parameters))
    logger.info('Start training and evaluating ...')
    start_time = time.time()
    total_batch = 0
    best_acc_val = 0.0
    last_improved_batch = 0
    isEarlyStop = False
    data_nums = 0
    for epoch in range(FLAGS.num_epochs):
        logger.info("Epoch : {}".format(epoch + 1))
        reader = pd.read_csv(FLAGS.data_file, chunksize=100000, sep='\u0001')
        for df in reader:
            logger.info(f"当前分批的大小： {df.shape}")
            records = df.to_dict('records')
            premises, hypythosises, labels = [], [], []
            for record in records:
                label, premise, hypothesis = record['label'], record['name'], record['query']
                premises.append(premise)
                hypythosises.append(hypothesis)
                labels.append(label)
            logger.info("===========================================")
            logger.info(f"当前训练数据: {len(premises)}")
            premises_train, premises_test = train_test_split(premises, test_size=FLAGS.test_size, random_state=1234)
            premises_train, premises_test = np.array(premises_train), np.array(premises_test)
            hypothesis_train, hypothesis_test = train_test_split(hypythosises, test_size=FLAGS.test_size, random_state=1234)
            hypothesis_train, hypothesis_test = np.array(hypothesis_train), np.array(hypothesis_test)
            labels_train, labels_test = train_test_split(labels, test_size=FLAGS.test_size, random_state=1234)
            labels_train, labels_test = np.array(labels_train), np.array(labels_test)

            sampleNums = len(premises_train)
            batchNums = int((sampleNums - 1) / FLAGS.batch_size) + 1
            data_nums += sampleNums

            indices = np.random.permutation(np.arange(sampleNums))
            premises_train = premises_train[indices]
            hypothesis_train = hypothesis_train[indices]
            labels_train = labels_train[indices]
            total_loss, total_acc = 0.0, 0.0
            for i in range(batchNums):
                batch_premises_np_train = premises_train[i * FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                batch_hypothesis_np_train = hypothesis_train[i * FLAGS.batch_size: (i+1)*FLAGS.batch_size]
                batch_labels_train = labels_train[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                batch_nums = FLAGS.batch_size
                feed_dict = feed_data(model, batch_premises_np_train,
                                      batch_hypothesis_np_train,
                                      batch_labels_train,
                                      FLAGS.dropout_keep_prob, is_train=True)
                _, batch_loss, batch_acc = sess.run([model.train, model.loss, model.acc], feed_dict=feed_dict)
                total_loss += batch_loss * batch_nums
                total_acc += batch_acc * batch_nums

                if total_batch % FLAGS.eval_batch == 0:
                    loss_val, acc_val = evaluate(sess, model, premises_test,
                                                 hypothesis_test,
                                                 labels_test)

                    saver.save(sess=sess, save_path=FLAGS.model_file + '_dev_loss_{:.4f}.ckpt'.format(loss_val))
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        last_improved_batch = total_batch
                        saver.save(sess=sess, save_path=FLAGS.best_path)
                        improved_flag = '*'
                    else:
                        improved_flag = ''
                    time_diff = get_time_diff(start_time)
                    msg = 'Epoch : {0:>3}, Batch : {1:>8}, Train Batch Loss : {2:>6.2}, Train Batch Acc : {3:>6.2%}, Dev Loss : {4:>6.2}, Dev Acc : {5:>6.2%}, Time : {6} {7}'
                    logger.info(msg.format(epoch + 1, total_batch, batch_loss, batch_acc, loss_val, acc_val, time_diff,
                                         improved_flag))
                total_batch += 1
                if total_batch - last_improved_batch > FLAGS.early_stop_step:
                    logger.info('No optimization for a long time, auto-stopping ...')
                    isEarlyStop = True
                    break
            if isEarlyStop:
                break
        time_diff = get_time_diff(start_time)
        total_loss, total_acc = total_loss / data_nums, total_acc / data_nums
        msg = '** Epoch : {0:>2} finished, Train Loss : {1:>6.2}, Train Acc : {2:6.2%}, Time : {3}'
        logger.info(msg.format(epoch + 1, total_loss, total_acc, time_diff))


if __name__ == '__main__':
    model = ESIM(FLAGS.seq_length, FLAGS.n_vocab, FLAGS.embedding_size,
                 FLAGS.hidden_size, FLAGS.attention_size, FLAGS.n_classes,
                 FLAGS.batch_size, FLAGS.learning_rate, FLAGS.optimizer,
                 FLAGS.l2, FLAGS.clip_value)
    train(model)





