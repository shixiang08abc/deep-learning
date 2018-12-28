# ==============================================================================
# Author: shixiang08abc@gmail.com
# Copyright 2018 Sogou Inc. All Rights Reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import re
import sys
import time
import WordEmbedding

from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import init_ops
from tensorflow.core.protobuf import saver_pb2

class Config(object):
  def __init__(self):
    self.batch_size = 200
    self.max_length = 30
    self.learning_rate = 0.0001
    self.momentum = 0.9
    self.max_epoch = 100
    self.target_delay = 0
    self.vocab_size = 450000
    self.embedding_dim = 300
    self.tag_size = 32
    self.tag_dim = 10
    self.cell_size = 128
    self.target_size = 1
    self.filter_width = 2
    self.pooling = 'fo'

def getFileNames(mydir):
  filenames = []
  for filename in os.listdir(os.path.dirname(mydir)):
    if re.match('^.+[0-9]+\.txt$',filename):
      filenames.append(mydir+filename)
  return filenames

class QRNNPoolCell(tf.contrib.rnn.RNNCell):
  def __init__(self, num_units, pooling='fo'):
    assert(pooling in ['f', 'fo', 'ifo'])
    self._num_units = num_units
    self.pooling = pooling
    func_map = {'f': self._f_pooling, 'fo': self._fo_pooling, 'ifo': self._ifo_pooling}
    self.pooling_func = func_map[self.pooling]

  @property
  def state_size(self):
    return self._num_units if self.pooling == 'f' else (self._num_units, self._num_units)

  @property
  def output_size(self):
    return self._num_units if self.pooling == 'f' else (self._num_units, self._num_units)

  def _f_pooling(self, inputs, state):
    assert(self.pooling == 'f')
    assert(len(inputs) == 2)
    z_t, f_t = inputs  # (bs, units), (bs, units)
    h_t_1 = state  # (bs, units) no cell units here
    h_t = f_t * h_t_1 + (1.0 - f_t) * z_t
    return h_t, h_t

  def _fo_pooling(self, inputs, state):
    assert(self.pooling == 'fo')
    assert(len(inputs) == 3)
    z_t, f_t, o_t = inputs  # cell gate, forget gate, output gate here
    c_t_1, _ = state  # (bs, units) in this case, state is cell. hid is output
    c_t = f_t * c_t_1 + (1.0 - f_t) * z_t
    h_t = o_t * c_t
    return (c_t, h_t), (c_t, h_t)  # (c, h)

  def _ifo_pooling(self, inputs, state):
    assert(self.pooling == 'ifo')
    assert(len(inputs) == 4)
    z_t, f_t, o_t, i_t = inputs
    c_t_1, _ = state  # (c, h)
    c_t = f_t * c_t_1 + i_t * z_t
    h_t = o_t * c_t
    return (c_t, h_t), (c_t, h_t)

  def __call__(self, inputs, state, scope=None):
    return self.pooling_func(inputs, state)

class createModel(object):
  def __init__(self, config, embedding):
    self.batch_size = config.batch_size
    self.num_steps = config.max_length
    self.learning_rate = config.learning_rate
    self.momentum = config.momentum
    self.target_delay = config.target_delay
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim

    self.cell_size = config.cell_size
    self.target_size = config.target_size

    self.filter_width = config.filter_width
    self.pooling = config.pooling

    self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps+self.target_delay])
    self._targets = tf.placeholder(tf.float32, [self.batch_size, self.num_steps+self.target_delay])
    self._lengths = tf.placeholder(tf.int32, [self.batch_size])
    self._frame_weight = tf.placeholder(tf.float32, [self.batch_size, self.num_steps+self.target_delay])

    word_embedding = tf.nn.embedding_lookup(embedding.id2embedding, self._input_data)
    self.targets = tf.reshape(tf.concat(self._targets, 0), [self.batch_size*(self.num_steps+self.target_delay)])
    self.frame_weight = tf.reshape(tf.concat(self._frame_weight, 0), [self.batch_size*(self.num_steps+self.target_delay)])

    self.hid_states, hid_final, O, cell_states = self.QRNNLayer(word_embedding, self._frame_weight, self._lengths)
    softmax_fw_w = tf.get_variable("softmax_fw_w", [2*self.cell_size, self.target_size])
    softmax_fw_b = tf.get_variable("softmax_fw_b", [self.target_size], initializer=init_ops.constant_initializer(0.0))
    self.logits_fw = tf.matmul(tf.reshape(self.hid_states, [-1, 2*self.cell_size]), softmax_fw_w) + softmax_fw_b
    self.logits = tf.sigmoid(tf.reshape(self.logits_fw, [self.batch_size*(self.num_steps+self.target_delay)]))
    self.mse_lose = (self.logits - self.targets) ** 2 * self.frame_weight / 2
    self.true_lose = tf.reduce_sum(self.mse_lose) / tf.reduce_sum(self.frame_weight)

    optimizer = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate)
    self._train_op = optimizer.minimize(self.mse_lose)

    self.var_list = tf.global_variables()
    self.print_variable()

  def QRNNLayer(self, seq_enc, seq_mask, seq_length, scope='QuasiRNN'):
    assert (self.pooling in ['o', 'f', 'fo', 'ifo'])
    seq_enc = seq_enc * seq_mask[:, :, None]
    Z, F, O, I = 0, 0, 0, 0
    with tf.variable_scope(scope):
      if self.pooling=='o':
        O = self.Masked1dConvOps(seq_enc, seq_mask, [tf.nn.sigmoid], scope='ConvO')
        fake_output = tf.constant(0.0, dtype=tf.float32)
        return fake_output, fake_output, O, fake_output
      elif self.pooling=='f':
        Z, F = self.Masked1dConvOps(seq_enc, seq_mask, seq_length, [tf.nn.tanh, tf.nn.sigmoid], scope='ConvZF')
      elif self.pooling=='fo':
        Z, F, O = self.Masked1dConvOps(seq_enc, seq_mask, seq_length, [tf.nn.tanh, tf.nn.sigmoid, tf.nn.sigmoid], scope='ConvZFO')
      elif self.pooling=='ifo':
        Z, F, O, I = self.Masked1dConvOps(seq_enc, seq_mask, seq_length, [tf.nn.tanh, tf.nn.sigmoid, tf.nn.sigmoid, tf.nn.sigmoid], scope='ConvZFOI')
      else:
        sys.stderr.write('>>>?????>>>')
      
      if self.pooling=='f':
        recurrent_inp = [Z, F]
      elif self.pooling=='fo':
        recurrent_inp = [Z, F, O]
      elif self.pooling=='ifo':
        recurrent_inp = [Z, F, O, I]
      else:
        sys.stderr.write('>>>?????>>>')

      pooling_cell = QRNNPoolCell(self.cell_size, self.pooling)
      states_fw, final_fw = tf.nn.dynamic_rnn(pooling_cell, recurrent_inp, seq_length, dtype=tf.float32)

      recurrent_inp_re = []
      for i in range(len(recurrent_inp)):
        temp = tf.reverse(recurrent_inp[i], [1])
        recurrent_inp_re.append(temp)
      states_bw, final_bw = tf.nn.dynamic_rnn(pooling_cell, recurrent_inp_re, seq_length, dtype=tf.float32)
      
      if self.pooling=='f':
        hid_final = tf.concat([final_fw, final_bw], 1)
        cell_states = tf.zeros([], dtype=tf.float32)
        hid_states = tf.concat([states_fw, tf.reverse(states_bw, [1])], 2)
      elif self.pooling=='fo' or self.pooling=='ifo':
        _, hid_final_fw = final_fw
        _, hid_final_bw = final_bw
        hid_final = tf.concat([hid_final_fw, hid_final_bw], 1)
        cell_states_fw, hid_states_fw = states_fw
        cell_states_bw, hid_states_bw = states_bw
        cell_states = tf.concat([cell_states_fw, tf.reverse(cell_states_bw, [1])], 2)
        hid_states = tf.concat([hid_states_fw, tf.reverse(hid_states_bw, [1])], 2)
      else:
        sys.stderr.write('>>>?????>>>')

      return hid_states, hid_final, O, cell_states

  def Masked1dConvOps(self, seq_enc, seq_mask, seq_length,  non_linear_list=[tf.nn.tanh, tf.nn.sigmoid], scope='Masked1dConv'):
    n_splits = len(non_linear_list)
    
    left_pad = self.filter_width - 1
    right_pad = 0
    seq_enc_pad = tf.pad(seq_enc, [[0, 0], [left_pad, right_pad], [0, 0]])

    with tf.variable_scope(scope):
      conv_w = tf.get_variable('conv_w', [self.filter_width, self.embedding_dim, self.cell_size*n_splits], dtype=tf.float32)
      conv_b = tf.get_variable('conv_b', [self.cell_size*n_splits], dtype=tf.float32)
    
      conv = tf.nn.conv1d(seq_enc_pad, conv_w, stride=1, padding='VALID')
      conv = conv[:, :self.num_steps+self.target_delay, :]
      conv += conv_b[None, None, :]
      split_conv = tf.split(conv, n_splits, 2)
      conv_out = []
      for i, f in enumerate(non_linear_list):
        split_conv_nonlinear = f(split_conv[i])
        split_conv_nonlinear *= seq_mask[:, :, None]
        conv_out.append(split_conv_nonlinear)
      return conv_out

  def print_variable(self):
    print("---------- Model Variabels -----------")
    cnt = 0
    for var in self.var_list:
      cnt += 1
      try:
        var_shape = var.get_shape()
      except:
        var_shape = var.dense_shape
      str_line = str(cnt) + '. ' + str(var.name) + '\t' + str(var.device) \
                 + '\t' + str(var_shape) + '\t' + str(var.dtype.base_dtype)
      print(str_line)
    print('------------------------')

def decode(session, model, w2v, test_file):
  total_lose = 0.0
  total_length = 0
  fin = open(test_file,"r")

  while True:
    success,data_list,tag_list,target_list,length_list,frame_weight = w2v.getImportanceBatchData(fin)
    if not success:
      break
    cost , logits = session.run([model.true_lose, model.logits],
                           {model._input_data: data_list,
                            model._targets: target_list,
                            model._lengths: length_list,
                            model._frame_weight: frame_weight})

    for i in range(model.batch_size):
      if length_list[i]==0:
        break
      total_length += length_list[i] - model.target_delay
      data_line = ""
      org_target = ""
      pre_weight = ""
      for j in range(length_list[i]):
        if j<(length_list[i] - model.target_delay):
          data_line = data_line + " " + w2v.getWord(data_list[i,j])
        if j>=model.target_delay:
          org_target = org_target + " " + str(target_list[i,j])
          pre_weight = pre_weight + " " + str(logits[i*(model.num_steps+model.target_delay)+j])
      print(data_line.strip()+"\t"+org_target.strip()+"\t"+pre_weight)
      sys.stdout.flush()

  err_line = 'avg_lose=%.8f\n' % (total_lose/total_length)
  sys.stderr.write(err_line)
  fin.close()

def main(unused_args):
  myconfig = Config()
  w2v = WordEmbedding.Word2Vec(myconfig)
  start_time = time.time()
  w2v.loadWordFile("word_table_merge")
  end_time = time.time()
  sys.stderr.write(' %.2f'%(end_time-start_time) + ' seconds escaped...\n')

  os.environ["CUDA_VISIBLE_DEVICES"] = "3"
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True

  with tf.Session(config=configproto) as sess:
    model = createModel(myconfig, w2v)
    loader = tf.train.Saver()
    loader.restore(sess, "models/qrnn_imp_bi_refine_009")

    decode(sess, model, w2v, 'src_code/input.demo')

if __name__=="__main__":
  tf.app.run()
