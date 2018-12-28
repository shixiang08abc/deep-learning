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
    self.target_delay = 0
    self.vocab_size = 450000
    self.embedding_dim = 100

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

    cell_size = 256
    target_size = 3
    project_size = 128

    self._input_data = tf.placeholder(tf.int32,[self.batch_size,self.num_steps+self.target_delay])
    self._targets = tf.placeholder(tf.int32,[self.batch_size,self.num_steps+self.target_delay])
    self._lengths = tf.placeholder(tf.int32,[self.batch_size])
    self._frame_weight = tf.placeholder(tf.float32,[self.batch_size,self.num_steps+self.target_delay])

    #forward = tf.nn.rnn_cell.LSTMCell(cell_size,num_proj=project_size,use_peepholes=True,forget_bias=0.0)
    forward_cell = tf.contrib.rnn.LSTMCell(cell_size, num_proj=project_size, use_peepholes=True, forget_bias=0.0)
    backward_cell = tf.contrib.rnn.LSTMCell(cell_size, num_proj=project_size, use_peepholes=True, forget_bias=0.0)

    '''
    embedding_table = tf.split(tf.nn.embedding_lookup(embedding.id2embedding,self._input_data),self.num_steps+self.target_delay,1)
    concat_embeddings = []
    for i in range(self.num_steps+self.target_delay):
      concat_embeddings.append(tf.reshape(embedding_table[i],[self.batch_size,self.embedding_dim]))
    '''
    concat_embeddings = tf.nn.embedding_lookup(embedding.id2embedding,self._input_data)
    self.lengths = tf.reshape(self._lengths,[self.batch_size])
    self.targets = tf.reshape(tf.concat(self._targets,0),[self.batch_size*(self.num_steps+self.target_delay)])
    self.onehot = tf.one_hot(self.targets,target_size,on_value=1.0,off_value=0.0)
    self.frame_weight = tf.reshape(tf.concat(self._frame_weight,0),[self.batch_size*(self.num_steps+self.target_delay)])
    #self.output = rnn.rnn(forward,concat_embeddings,dtype=tf.float32,sequence_length=self.lengths)
    (self.output_fw,self.output_bw),_ = tf.nn.bidirectional_dynamic_rnn(forward_cell, backward_cell, concat_embeddings, sequence_length=self.lengths, dtype=tf.float32)

    softmax_fw_w = tf.get_variable("softmax_fw_w",[2*project_size,target_size])
    softmax_fw_b = tf.get_variable("softmax_fw_b",[target_size],initializer=init_ops.constant_initializer(0.0))
    self.logits_fw = tf.matmul(tf.reshape(tf.concat([self.output_fw,self.output_bw],2),[-1, 2*project_size]),softmax_fw_w) + softmax_fw_b
    self.logits = tf.sigmoid(self.logits_fw)

    #self.mse_lose = (self.logits - self.targets) ** 2 * self.frame_weight / 2
    #self.lose = tf.nn.sigmoid_cross_entropy_with_logits((self.logits_fw),self.onehot) * self.frame_weight
    self.lose = tf.nn.softmax_cross_entropy_with_logits(labels=self.onehot, logits=self.logits_fw) * self.frame_weight
    self._lose = tf.reduce_sum(self.lose)
    #self.true_lose = tf.reduce_sum(self.mse_lose) / tf.reduce_sum(self.frame_weight)

    #add crf
    self.logits_crf = tf.reshape(self.logits_fw,[self.batch_size, self.num_steps+self.target_delay, target_size])
    self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.logits_crf, self._targets, self.lengths)
    self._crflose = tf.reduce_sum(-(self.log_likelihood))
    #optimizer_crf = tf.train.AdamOptimizer(self.learning_rate)
    optimizer_crf = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate)
    self.train_op_crf = optimizer_crf.minimize(self._crflose)

    tvars = tf.trainable_variables()
    #self.grads = tf.gradients(self._lose,tvars)
    self.grads = tf.gradients(self._crflose,tvars)
    optimizer = tf.train.MomentumOptimizer(self.learning_rate,self.momentum)
    self._train_op = optimizer.apply_gradients(zip(self.grads,tvars))

    self.var_list = tf.global_variables()
    self.print_variable()
    #self.init_tensor_table()
    #self.saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

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

def run_epoch(session,model,w2v,train_file,epoch_id):
  print("file=%s, epoch=%d begins:" % (train_file,epoch_id))
  start_time = time.time()
  costs = 0.0
  steps = 0

  fin = open(train_file,"r")
  while True:
    success,data_list,target_list,length_list,frame_weight = w2v.getTermWeightBactchDate(fin,model.target_delay)
    if not success:
      break
    #cost , _ = session.run([model._lose, model._train_op],
    cost , _= session.run([model._crflose, model.train_op_crf],
                           {model._input_data: data_list,
                            model._targets: target_list,
                            model._lengths: length_list,
                            model._frame_weight: frame_weight})
    costs += cost
    steps += 1

    if steps%100==0:
      print("avg cost after %5d batches: cur_loss=%.6f, avg_loss=%.6f, %5.2f seconds elapsed ..." % (steps, cost, (costs/steps), (time.time()-start_time)))
      sys.stdout.flush()

  fin.close()
  saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
  saver.save(session, 'models/lstmp_tw_refine_'+'%03d'%epoch_id, write_meta_graph=False)
  #model.saver.save(session, 'models/lstmp_tw_refine_'+'%03d'%epoch_id, write_meta_graph=False)
  #saver = tf.train.Saver(model.tensor_table.values())
  #saver.save(session, 'models/lstmp_tw_refine_'+'%03d'%epoch_id, write_meta_graph=False)
  saver_emb = tf.train.Saver({'embedding':w2v.id2embedding}, write_version=saver_pb2.SaverDef.V1)
  saver_emb.save(session, 'models/lstmp_outter_embedding_refine_'+'%03d'%epoch_id, write_meta_graph=False)

def main(unused_args):
  myconfig = Config()
  w2v = WordEmbedding.Word2Vec(myconfig)
  start_time = time.time()
  w2v.loadWordFile("word_table_merge")
  end_time = time.time()
  sys.stderr.write(' %.2f'%(end_time-start_time) + ' seconds escaped...\n')
  trainnames = getFileNames("data/")
  trainnames.sort()
  print("traindata:\t",trainnames)

  os.environ["CUDA_VISIBLE_DEVICES"] = "5"
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True

  with tf.Session(config=configproto) as sess:
    filenum = len(trainnames)
    lstm_model = createLstmModel(myconfig,w2v)
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    sess.run(init)

    #w2v.loadEmbeddings(sess,'word_embedding.tensorflow')
    #end_time = time.time()
    #sys.stderr.write(' %.2f'%(end_time-start_time) + ' seconds escaped...\n')

    for i in range(myconfig.max_epoch):
      run_epoch(sess,lstm_model,w2v,trainnames[i%filenum],i+1)

if __name__=="__main__":
  tf.app.run()

