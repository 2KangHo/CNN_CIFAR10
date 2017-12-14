import numpy as np
import tensorflow as tf

flag = tf.app.flags


flag.DEFINE_string('log_dir', './logs', 'log directory')
flag.DEFINE_string('save', './ckpt/LeeNet.ckpt', 'checkpoint save file path')

flag.DEFINE_boolean('is_training', True, 'train or test')
flag.DEFINE_string('labels', './input/batches.meta', 'label dictionary data file path')
flag.DEFINE_string('data1', './input/data_batch_1', 'train data file path')
flag.DEFINE_string('data2', './input/data_batch_2', 'train data file path')
flag.DEFINE_string('data3', './input/data_batch_3', 'train data file path')
flag.DEFINE_string('data4', './input/data_batch_4', 'train data file path')
flag.DEFINE_string('data5', './input/data_batch_5', 'train data file path')
flag.DEFINE_string('testdata', './input/test_batch', 'test data file path')

# hyper parameters
flag.DEFINE_float('l_rate', 0.001, 'learning rate')
flag.DEFINE_integer('b_size', 200, 'mini batch size')
flag.DEFINE_integer('epoch', 32, 'epoch number')


cfg = flag.FLAGS
