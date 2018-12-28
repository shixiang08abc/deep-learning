# ==============================================================================
# Author: shixiang08abc@gmail.com
# Copyright 2017 Sogou Inc. All Rights Reserved.
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
    self.target_delay = 5
    self.vocab_size = 450000
    self.embedding_dim = 100
    self.tag_size = 32
    self.tag_dim = 10
    self.cell_size = 256
    self.target_size = 1
    self.project_size = 128

class createLstmModel(object):
  def __init__(self,config,embedding):
    #self.tensor_table = {}
    self.batch_size = config.batch_size
    self.num_steps = config.max_length
    self.learning_rate = config.learning_rate
    self.momentum = config.momentum
    self.target_delay = config.target_delay
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.tag_size = config.tag_size
    self.tag_dim = config.tag_dim

    self.cell_size = config.cell_size
    self.target_size = config.target_size
    self.project_size = config.project_size

    self._input_data = tf.placeholder(tf.int32, [self.batch_size, self.num_steps+self.target_delay])
    self._input_tag = tf.placeholder(tf.int32, [self.batch_size, self.num_steps+self.target_delay])
    self._targets = tf.placeholder(tf.float32, [self.batch_size, self.num_steps+self.target_delay])
    self._lengths = tf.placeholder(tf.int32, [self.batch_size])
    self._frame_weight = tf.placeholder(tf.float32, [self.batch_size, self.num_steps+self.target_delay])

    lstm_cell = tf.contrib.rnn.LSTMCell(self.cell_size, num_proj=self.project_size, use_peepholes=True, forget_bias=0.0)

    word_embedding = tf.nn.embedding_lookup(embedding.id2embedding, self._input_data)
    tag_embedding = tf.nn.embedding_lookup(embedding.id2tagembedding, self._input_tag)
    concat_embedding = tf.concat([word_embedding, tag_embedding], 2)
    self.lengths = tf.reshape(self._lengths, [self.batch_size])
    self.targets = tf.reshape(tf.concat(self._targets, 0), [self.batch_size*(self.num_steps+self.target_delay)])
    self.frame_weight = tf.reshape(tf.concat(self._frame_weight, 0), [self.batch_size*(self.num_steps+self.target_delay)])
    self.output, _ = tf.nn.dynamic_rnn(lstm_cell, concat_embedding, sequence_length=self.lengths, dtype=tf.float32)
    
    softmax_fw_w = tf.get_variable("softmax_fw_w", [self.project_size, self.target_size])
    softmax_fw_b = tf.get_variable("softmax_fw_b", [self.target_size], initializer=init_ops.constant_initializer(0.0))
    self.logits_fw = tf.matmul(tf.reshape(self.output, [-1, self.project_size]), softmax_fw_w) + softmax_fw_b
    #self.logits_tar = tf.reshape(self.logits_fw, [self.batch_size*(self.num_steps+self.target_delay)])
    self.logits = tf.sigmoid(tf.reshape(self.logits_fw, [self.batch_size*(self.num_steps+self.target_delay)]))
    self.mse_lose = (self.logits - self.targets) ** 2 * self.frame_weight / 2
    #self.lose = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.targets, logits=self.logits_tar)
    #self._lose = tf.reduce_sum(self.lose)
    self.true_lose = tf.reduce_sum(self.mse_lose) / tf.reduce_sum(self.frame_weight)

    tvars = tf.trainable_variables()
    self.grads = tf.gradients(self.mse_lose,tvars)
    optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
    self._train_op = optimizer.apply_gradients(zip(self.grads,tvars))

    #self.init_tensor_table()
    self.var_list = tf.global_variables()
    self.print_variable()

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

  #def init_tensor_table(self):
  #  for v in tf.all_variables():
  #    if re.match('^.+/W_0:0$',v.name):
  #      self.tensor_table['FW_W_0'] = v
  #    elif re.match('^.+/W_P_0:0$',v.name):
  #      self.tensor_table['FW_W_P'] = v
  #    elif re.match('^.+/B:0$',v.name):
  #      self.tensor_table['FW_B'] = v
  #    elif re.match('^.+/W_I_diag:0$',v.name):
  #      self.tensor_table['FW_W_I'] = v
  #    elif re.match('^.+/W_F_diag:0$',v.name):
  #      self.tensor_table['FW_W_F'] = v
  #    elif re.match('^.+/W_O_diag:0$',v.name):
  #      self.tensor_table['FW_W_O'] = v
  #    elif re.match('^softmax_fw_w:0$',v.name):
  #      self.tensor_table['FW_W_sm'] = v
  #    elif re.match('^softmax_fw_b:0$',v.name):
  #      self.tensor_table['FW_B_sm'] = v

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
                            model._input_tag: tag_list,
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

  os.environ["CUDA_VISIBLE_DEVICES"] = "4"
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True

  with tf.Session(config=configproto) as sess:
    #w2v.loadEmbedding(sess,"models/lstmp_outter_embedding_refine_001")
    #end_time = time.time()
    #sys.stderr.write(' %.2f'%(end_time-start_time) + ' seconds escaped...\n')

    lstm_model = createLstmModel(myconfig,w2v)
    loader = tf.train.Saver()
    loader.restore(sess, "models/lstmp_imp_refine_010")

    decode(sess, lstm_model, w2v, "src_code/input.demo")

if __name__=="__main__":
  tf.app.run()
