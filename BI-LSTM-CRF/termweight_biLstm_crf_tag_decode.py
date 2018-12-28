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
    self.batch_size = 1
    self.max_length = 30
    self.learning_rate = 0.0001
    self.momentum = 0.9
    self.max_epoch = 200
    self.target_delay = 0
    self.vocab_size = 450000
    self.embedding_dim = 100
    self.tag_size = 32
    self.tag_dim = 10
    self.cell_size = 256
    self.target_size = 3
    self.project_size = 128

def getFileNames(mydir):
  filenames = []
  for filename in os.listdir(os.path.dirname(mydir)):
    if re.match('^.+[0-9]+\.txt$',filename):
      filenames.append(mydir+filename)
  return filenames

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

    self._input_data = tf.placeholder(tf.int32,[self.batch_size,self.num_steps+self.target_delay])
    self._input_tag = tf.placeholder(tf.int32, [self.batch_size, self.num_steps+self.target_delay])
    self._targets = tf.placeholder(tf.int32,[self.batch_size,self.num_steps+self.target_delay])
    self._lengths = tf.placeholder(tf.int32,[self.batch_size])
    self._frame_weight = tf.placeholder(tf.float32,[self.batch_size,self.num_steps+self.target_delay])

    forward_cell = tf.contrib.rnn.LSTMCell(self.cell_size, num_proj=self.project_size, use_peepholes=True, forget_bias=0.0)
    backward_cell = tf.contrib.rnn.LSTMCell(self.cell_size, num_proj=self.project_size, use_peepholes=True, forget_bias=0.0)

    word_embedding = tf.nn.embedding_lookup(embedding.id2embedding,self._input_data)
    tag_embedding = tf.nn.embedding_lookup(embedding.id2tagembedding, self._input_tag)
    concat_embedding = tf.concat([word_embedding, tag_embedding], 2)
    self.lengths = tf.reshape(self._lengths,[self.batch_size])
    self.targets = tf.reshape(tf.concat(self._targets,0),[self.batch_size*(self.num_steps+self.target_delay)])
    self.onehot = tf.one_hot(self.targets,self.target_size,on_value=1.0,off_value=0.0)
    self.frame_weight = tf.reshape(tf.concat(self._frame_weight,0),[self.batch_size*(self.num_steps+self.target_delay)])
    (self.output_fw,self.output_bw),_ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, concat_embedding, sequence_length=self.lengths, dtype=tf.float32)

    softmax_fw_w = tf.get_variable("softmax_fw_w",[2*self.project_size,self.target_size])
    softmax_fw_b = tf.get_variable("softmax_fw_b",[self.target_size],initializer=init_ops.constant_initializer(0.0))
    self.logits_fw = tf.matmul(tf.reshape(tf.concat([self.output_fw,self.output_bw],2),[-1, 2*self.project_size]),softmax_fw_w) + softmax_fw_b
    self.logits = tf.sigmoid(self.logits_fw)

    #self.mse_lose = (self.logits - self.targets) ** 2 * self.frame_weight / 2
    #self.lose = tf.nn.sigmoid_cross_entropy_with_logits((self.logits_fw),self.onehot) * self.frame_weight
    self.lose = tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot, logits=self.logits_fw) * self.frame_weight
    self._lose = tf.reduce_sum(self.lose)
    #self.true_lose = tf.reduce_sum(self.mse_lose) / tf.reduce_sum(self.frame_weight)

    #add crf
    self.logits_crf = tf.reshape(self.logits_fw,[self.batch_size, self.num_steps+self.target_delay, self.target_size])
    self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits_crf, self._targets, self.lengths)
    self._crflose = tf.reduce_sum(-(self.log_likelihood))
    self.crf_tag, self.crf_socre = tf.contrib.crf.crf_decode(self.logits_crf, self.transition_params, self.lengths) 
    #self.crf_score = []
    #for i in range(self.batch_size):
    #  _logits_crf = self.logits_crf[i]
    #  _length = self.lengths[i]
    #  one_score = _logits_crf[:_length]
    #  viterbi, _ = tf.contrib.crf.viterbi_decode(one_score, self.transition_params)
    #  self.crf_score.append(viterbi)
    #optimizer_crf = tf.train.AdamOptimizer(self.learning_rate)
    #self.train_op_crf = optimizer_crf.minimize(self._crflose)

    regularization_cost = 0.01* tf.reduce_sum([tf.nn.l2_loss(softmax_fw_w)])

    optimizer_crf = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate)
    self.train_op_crf = optimizer_crf.minimize(self._crflose + regularization_cost)

    #tvars = tf.trainable_variables()
    #self.grads = tf.gradients(self._lose,tvars)
    #self.grads = tf.gradients(self._crflose,tvars)
    #optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
    #self._train_op = optimizer.apply_gradients(zip(self.grads,tvars))

    self.var_list = tf.global_variables()
    #self.print_variable()

#  def print_variable(self):
#    print("---------- Model Variabels -----------")
#    cnt = 0
#    for var in self.var_list:
#      cnt += 1
#      try:
#        var_shape = var.get_shape()
#      except:
#        var_shape = var.dense_shape
#      str_line = str(cnt) + '. ' + str(var.name) + '\t' + str(var.device) \
#                 + '\t' + str(var_shape) + '\t' + str(var.dtype.base_dtype)
#      print(str_line)
#    print('------------------------')

def decode(session, model, w2v, decode_file):
  start_time = time.time()
  totle_length = 0

  fin = open(decode_file,"r")
  while True:
    success,data_list,tag_list,target_list,length_list,frame_weight = w2v.getTermWeightBactchDate(fin)
    if not success:
      break
    cost , result_list = session.run([model._crflose, model.crf_tag],
                           {model._input_data: data_list,
                            model._input_tag: tag_list,
                            model._targets: target_list,
                            model._lengths: length_list,
                            model._frame_weight: frame_weight})
    for i in range(model.batch_size):
      data_line = ""
      target_line = ""
      outer_line = ""
      for j in range(length_list[i]):
        if j<(length_list[i]-model.target_delay):
          data_line = data_line + " " + w2v.getWord(data_list[i,j])
        if j>=model.target_delay:
          target_line = target_line + " " + str(target_list[i,j])
          outer_line = outer_line + " " + str(result_list[i,j])
      print(data_line.strip()+"\n"+target_line.strip()+"\n"+outer_line.strip())
      print("\n------------------------------------\n")
      sys.stdout.flush()

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
    #w2v.loadEmbeddings(sess,'word_embedding.tensorflow')
    #end_time = time.time()
    #sys.stderr.write(' %.2f'%(end_time-start_time) + ' seconds escaped...\n')

    lstm_model = createLstmModel(myconfig,w2v)
    loader = tf.train.Saver()
    loader.restore(sess, "models/lstmp_tw_refine_199")

    decode(sess, lstm_model, w2v, 'data/test.txt')

if __name__=="__main__":
  tf.app.run()

