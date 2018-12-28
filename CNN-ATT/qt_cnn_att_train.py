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
    self.learning_rate = 0.001
    self.l2_rate = 1e-8
    self.momentum = 0.9
    self.max_epoch = 100
    self.vocab_size = 450000
    self.embedding_dim = 100
    self.margin = 0.1
    self.core_nums = 400
    self.conv_window = 3
    self.num_layers = [300, 300, 200]

def getFileNames(mydir):
  filenames = []
  for filename in os.listdir(os.path.dirname(mydir)):
    if re.match('^.+[0-9]+\.txt$',filename):
      filenames.append(mydir+filename)
  return filenames

class createModel(object):
  def __init__(self, config, embedding):
    self.batch_size = config.batch_size
    self.num_steps = config.max_length
    self.learning_rate = config.learning_rate
    self.l2_rate = config.l2_rate
    self.momentum = config.momentum
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.margin = config.margin
    
    self.core_nums = config.core_nums
    self.conv_window = config.conv_window
    self.num_layers = config.num_layers

    self.query_data_ = tf.placeholder(tf.int32, [self.batch_size, self.num_steps])
    self.query_mask_ = tf.placeholder(tf.float32, [self.batch_size, self.num_steps])
    self.query_length_ = tf.placeholder(tf.int32, [self.batch_size])
    self.title_data_ = tf.placeholder(tf.int32, [self.batch_size, 2, self.num_steps])
    self.title_mask_ = tf.placeholder(tf.float32, [self.batch_size, 2, self.num_steps])
    self.title_length_ = tf.placeholder(tf.int32, [self.batch_size, 2])

    query_length = tf.reshape(self.query_length_, [self.batch_size])
    query_mask = tf.reshape(tf.tile(tf.expand_dims(self.query_mask_, 1), [1,2,1]), [-1, self.num_steps])#(2*bs, ns)
    query_embedding = tf.nn.embedding_lookup(embedding.id2embedding, self.query_data_) #(bs,ns,dm)
    query_embeddings = tf.reshape(tf.tile(tf.expand_dims(query_embedding, 1), [1,2,1,1]), [-1, self.num_steps, self.embedding_dim]) #(2*bs,ns,dm)

    self.title_data = tf.reshape(self.title_data_, [-1, self.num_steps])
    title_length = tf.reshape(self.title_length_, [self.batch_size*2])
    title_mask = tf.reshape(self.title_mask_, [-1, self.num_steps]) #(2*bs, ns)
    title_embeddings = tf.nn.embedding_lookup(embedding.id2embedding, self.title_data) #(2*bs,ns,dm)

    #attention
    query_states, title_states, self.att = self.EmbdAttention(query_embeddings, query_mask, title_embeddings, title_mask)

    #merge embedding and att
    query_embd_att = self.MergeSim(query_embeddings, query_states, 'QueryMerge')
    title_embd_att = self.MergeSim(title_embeddings, title_states, 'TitleMerge')

    #conv layer and fully connect layer
    query_enc = self.TextConvPool1D(query_embd_att, query_mask, 'QueryConv')
    title_enc = self.TextConvPool1D(title_embd_att, title_mask, 'TitleConv')

    query_enc = tf.reshape(query_enc, [self.batch_size, -1, self.num_layers[-1]])
    title_enc = tf.reshape(title_enc, [self.batch_size, -1, self.num_layers[-1]])
    self._score = self.BiLinerScore(query_enc, title_enc)
    gap = self._score[:, 1] - self._score[:, 0]
    self.pair_loss = tf.reduce_mean(tf.nn.relu(gap + self.margin))
    self.regu_loss = self.l2_rate * tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    self.lose = self.pair_loss + self.regu_loss

    tvars = tf.trainable_variables()
    self.grads = tf.gradients(self.lose, tvars)
    #optimizer = tf.contrib.opt.LazyAdamOptimizer(self.learning_rate)
    #optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
    optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
    self._train_op = optimizer.apply_gradients(zip(self.grads, tvars))
    
    self.var_list = tf.global_variables()
    self.print_variable()


  def TextConvPool1D(self, inputs, mask, scope='TextCNN'):
    with tf.variable_scope(scope, initializer=tf.random_normal_initializer(0.0, 1e-2), regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      inputs = inputs * mask[:, :, None]
      text_conv = tf.contrib.layers.conv2d(inputs, self.core_nums, self.conv_window, padding='SAME')
      print("text_conv shape:", text_conv.get_shape())
      text_conv = text_conv * mask[:, :, None]
      text_conv = tf.expand_dims(text_conv, -2)
      text_pool = tf.contrib.layers.max_pool2d(text_conv, [self.num_steps, 1])
      print ("text_pool shape:", text_pool.get_shape())
      text_enc = tf.contrib.layers.flatten(text_pool)
      print ("text_enc shape:", text_enc.get_shape())
      for i, layers_size in enumerate(self.num_layers):
        suffix = '' if i == 0 else str(i)
        text_enc = tf.contrib.layers.fully_connected(text_enc, layers_size, scope="text_enc"+suffix)
      return text_enc

  def MergeSim(self, embeddings, states, scope='MergeSim'):
    with tf.variable_scope(scope, initializer=tf.random_normal_initializer(0.0, 1e-2), regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      U_merge_embed = tf.get_variable('U_merge_embed', shape=[self.embedding_dim, self.embedding_dim])
      U_merge_state = tf.get_variable('U_merge_state', shape=[self.embedding_dim, self.embedding_dim])
      text_embed = tf.reshape(tf.matmul(tf.reshape(embeddings, [-1, self.embedding_dim]), U_merge_embed), [-1, self.num_steps, self.embedding_dim])
      text_state = tf.reshape(tf.matmul(tf.reshape(states, [-1, self.embedding_dim]), U_merge_state), [-1, self.num_steps, self.embedding_dim])
      return tf.nn.relu(text_embed + text_state)
 
  def EmbdAttention(self, query_embeddings, query_mask, title_embeddings, title_mask, scope='EmbdAtt'):
    with tf.variable_scope(scope, initializer=tf.random_normal_initializer(0.0, 1e-2), regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      U_query_att = tf.get_variable('U_query_att', shape=[self.num_steps, self.embedding_dim])
      U_title_att = tf.get_variable('U_title_att', shape=[self.num_steps, self.embedding_dim])
      mask = query_mask[:, None, :] * title_mask[:, :, None] #(2*bs, tns, qns)
      csim = self.EmbdCosDis(query_embeddings, title_embeddings) #(2*bs, tns, qns)
      exp_a_mask = csim * mask
      query_att = tf.nn.relu(tf.reshape(tf.matmul(tf.reshape(tf.transpose(exp_a_mask, [0, 2, 1]), [-1, self.num_steps]), U_query_att), [-1, self.num_steps, self.embedding_dim]))
      title_att = tf.nn.relu(tf.reshape(tf.matmul(tf.reshape(exp_a_mask, [-1, self.num_steps]), U_title_att), [-1, self.num_steps, self.embedding_dim]))
      return query_att, title_att, exp_a_mask

  def EmbdCosDis(self, query_embeddings, title_embeddings):
    q_sqrt = tf.square(query_embeddings)
    t_sqrt = tf.square(title_embeddings)
    q_sum = tf.reduce_sum(q_sqrt, 2)
    t_sum = tf.reduce_sum(t_sqrt, 2)
    normal_q = query_embeddings / tf.sqrt(q_sum)[:, :, None]
    normal_t = title_embeddings / tf.sqrt(t_sum)[:, :, None]
    cosDot = tf.reduce_sum(normal_q[:, None, :, :] * normal_t[:, :, None, :], 3)
    print("cosDot shape:", cosDot.get_shape())
    return cosDot

  def BiLinerScore(self, query_enc, title_enc, scope="LogLinear"):
    with tf.variable_scope(scope, initializer=tf.random_normal_initializer(0.0, 1e-2), regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      U = tf.get_variable('bi_liner_U', shape=[self.num_layers[-1], self.num_layers[-1]], initializer=tf.random_normal_initializer(0.0, 1e-2))
      V = tf.get_variable('bi_liner_V', shape=[2*self.num_layers[-1]], initializer=tf.random_normal_initializer(0.0, 1e-2))
      B = tf.get_variable('bi_liner_B', shape=[], initializer=init_ops.constant_initializer(0.0))
      feat_concat = tf.concat([query_enc, title_enc], 2)
      score = B + tf.reshape(tf.matmul(tf.reshape(feat_concat, [-1, 2*self.num_layers[-1]]), V[:, None]), [self.batch_size, -1])
      mid = tf.reshape(tf.matmul(tf.reshape(query_enc, [-1, self.num_layers[-1]]), U), [self.batch_size, -1, self.num_layers[-1]])
      score += tf.reduce_sum(mid * title_enc, 2)
      return tf.nn.sigmoid(score)
    
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


def run_epoch(session, model, w2v, train_file, epoch_id):
  print("file=%s, epoch=%d begins:" % (train_file,epoch_id))
  start_time = time.time()
  costs = 0.0
  steps = 0

  fin = open(train_file,"r")
  while True:
    success, query_list, query_length_list, query_mask_weight, title_list, title_length_list, title_mask_weight = w2v.getQueryTitleBatchData(fin)
    if not success:
      break
    cost, score, _ = session.run([model.lose, model._score, model._train_op],
                           {model.query_data_: query_list,
                            model.query_mask_: query_mask_weight,
                            model.query_length_: query_length_list,
                            model.title_data_: title_list,
                            model.title_mask_: title_mask_weight,
                            model.title_length_: title_length_list})
    costs += cost
    steps += 1

    if steps%100==0:
      print("avg cost after %5d batches: cur_loss=%.6f, avg_loss=%.6f, %5.2f seconds elapsed ..." % (steps, cost, (costs/steps), (time.time()-start_time)))
      if steps%10000==0:
        for i in range(model.batch_size):
          if i%10==0:
            print("batch:%d\tscore1:%f\tscore2:%f" % (i, score[i][0], score[i][1]))
      sys.stdout.flush()

  fin.close()

  saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)
  saver.save(session, 'models/qt_cnn_att_refine_'+'%03d'%epoch_id, write_meta_graph=False)
  saver_emb = tf.train.Saver({'embedding':w2v.id2embedding}, write_version=saver_pb2.SaverDef.V1)
  saver_emb.save(session, 'models/qt_cnn_att_embedding_refine_'+'%03d'%epoch_id, write_meta_graph=False)


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

  os.environ["CUDA_VISIBLE_DEVICES"] = "2"
  configproto = tf.ConfigProto()
  configproto.gpu_options.allow_growth = True
  configproto.allow_soft_placement = True

  with tf.Session(config=configproto) as sess:
    filenum = len(trainnames)
    model = createModel(myconfig, w2v)
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(myconfig.max_epoch):
      run_epoch(sess, model, w2v, trainnames[i%filenum], i+1)

if __name__=="__main__":
  tf.app.run()

