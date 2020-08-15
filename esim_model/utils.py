import os
import re
import time
import logging
import pkuseg
import numpy as np
import tensorflow as tf
from termcolor import colored
from datetime import timedelta


seg = pkuseg.pkuseg()


def set_logger(context, verbose=False):
    if os.name == 'nt':  # for Windows
        return NTLogger(context, verbose)

    logger = logging.getLogger(context)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    formatter = logging.Formatter(
        '%(levelname)-.1s:' + context + ':[%(filename).10s:%(funcName).20s:%(lineno)3d]:%(message)s', datefmt=
        '%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(console_handler)
    return logger


class NTLogger:
    def __init__(self, context, verbose):
        self.context = context
        self.verbose = verbose

    def info(self, msg, **kwargs):
        print('I:%s:%s' % (self.context, msg), flush=True)

    def debug(self, msg, **kwargs):
        if self.verbose:
            print('D:%s:%s' % (self.context, msg), flush=True)

    def error(self, msg, **kwargs):
        print('E:%s:%s' % (self.context, msg), flush=True)

    def warning(self, msg, **kwargs):
        print('W:%s:%s' % (self.context, msg), flush=True)


logger = set_logger(colored('oredict', 'yellow'), False)


def print_shape(varname, var):
    """
    :param varname: tensor name
    :param var: tensor variable
    """
    logger.info('{0} : {1}'.format(varname, var.get_shape()))


def read_word2id(file_name):
    word2id = {}
    with open(file_name) as f:
        lines = f.readlines()
        for line in lines:
            line = re.sub('\n', '', line)
            if len(line.split(' ')) == 2:
                word, id = line.split(' ')
                word2id[word] = int(id)
    return word2id


def sentence_to_ids(sentences, word_to_ids, sentence_len=30):
    res_words_id = []
    res_words_mask = []
    for sentence in list(sentences):
        words = seg.cut(str(sentence))
        words_id = []
        for word in words:
            if word in word_to_ids:
                words_id.append(word_to_ids[word])
            else:
                words_id.append(0)
        true_len = len(words) if len(words) <= sentence_len else sentence_len
        words_id = words_id[:sentence_len] if len(words_id) >= sentence_len else words_id + [0] * (sentence_len - len(words_id))
        # words_mask = [1] * true_len + [0] * (sentence_len - true_len)
        res_words_id.append(np.array(words_id, dtype=np.int32))
        res_words_mask.append(true_len)
    return np.array(res_words_id), np.array(res_words_mask)


def convert2onehot(y):
    labels = []
    for label in list(y):
        try:
            if int(label) == 1:
                labels.append(np.array([0, 1], dtype=np.int32))
            else:
                labels.append(np.array([1, 0], dtype=np.int32))
        except:
            labels.append(np.array([1, 0], dtype=np.int32))
            continue
    return np.array(labels)


def read_file(file_name):
    premises, hypythosises, labels = [], [], []
    with open(file_name) as f:
        for line in f:
            if len(line.split('\u0001')) != 3:
                continue
            label, premise, hypothesis = line.split('\u0001')
            premises.append(premise)
            hypythosises.append(hypothesis)
            labels.append(label)
    return premises, hypythosises, labels


def count_parameters():
    totalParams = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variableParams = 1
        for dim in shape:
            variableParams *= dim.value
        totalParams += variableParams
    return totalParams


def get_time_diff(startTime):
    endTime = time.time()
    diff = endTime - startTime
    return timedelta(seconds=int(round(diff)))

