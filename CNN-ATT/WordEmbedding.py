# ==============================================================================
# Author: sxron
# E-mail: shixiang08abc@gmail.com
# Copyright 2017 Sogou Inc. All Rights Reserved.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import os
import re
import sys


class Word2Vec(object):
  def __init__(self,config):
    self.word2id = {}
    self.id2word = []
    self.num_steps = config.max_length
    self.batch_size = config.batch_size
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    with tf.device("cpu:0"):
      self.id2embedding = tf.get_variable("embedding",[self.vocab_size+1,self.embedding_dim],dtype=tf.float32)
      #self.id2embedding = tf.get_variable("embedding",[self.vocab_size+1,self.embedding_dim],dtype=tf.float32,trainable=False)

  def loadWordFile(self,filename):
    sys.stderr.write("\nloading word table...")
    fin = open(filename,'r')
    while True:
      word = fin.readline()
      word = word.strip()
      if word=="":
        break
      index = len(self.id2word)
      self.id2word.append(word)
      self.word2id[word] = index
    fin.close()
    sys.stderr.write("\nloading word table finished!!!")

  def loadEmbeddings(self,session,filepath):
    sys.stderr.write("\nloading word embeddings form tensor file...")
    saver = tf.train.Saver({'embedding':self.id2embedding})
    saver.restore(session,filepath)
    sys.stderr.write("\nloading word embeddings finished !!!")

  def getWid(self,term):
    if self.word2id.has_key(term):
      return self.word2id[term]
    else:
      return self.vocab_size

  def getWord(self,wid):
    assert wid<=self.vocab_size
    return self.id2word[wid]

  def getQueryTitleBatchData(self,fin):
    batch_cursor = 0
    query_list = np.zeros([self.batch_size, self.num_steps], dtype=np.int32)
    query_length_list = np.zeros([self.batch_size], dtype=np.int32)
    query_mask_weight = np.zeros([self.batch_size, self.num_steps], dtype=np.float32)
    title_list = np.zeros([self.batch_size, 2, self.num_steps], dtype=np.int32)
    title_length_list = np.zeros([self.batch_size, 2], dtype=np.int32)
    title_mask_weight = np.zeros([self.batch_size, 2, self.num_steps], dtype=np.float32)

    while True:
      line = fin.readline()
      if line=="":
        if batch_cursor==0:
          return False, query_list, query_length_list, query_mask_weight, title_list, title_length_list, title_mask_weight
        else:
          return True, query_list, query_length_list, query_mask_weight, title_list, title_length_list, title_mask_weight

      line = line.strip()
      tokens = line.split("\t")
      if len(tokens)!=9:
        print ("Invalid line: %s , must have 9 fields." % line)
        continue
      querys = tokens[0].strip().split(" ")
      titles1 = tokens[1].strip().split(" ")
      titles2 = tokens[5].strip().split(" ")

      if len(querys)<=0 or len(titles1)<=0 or len(titles2)<=0:
        print ("Invalid line: %s , query or title miss." % line)
        continue

      for idx, word in enumerate(querys):
        if idx>=self.num_steps:
            break
        query_list[batch_cursor, idx] = self.getWid(querys[idx].strip())
        query_mask_weight[batch_cursor, idx] = 1.0
      query_length_list[batch_cursor] = len(querys) if len(querys)<=self.num_steps else self.num_steps

      for idx, word in enumerate(titles1):
        if idx>=self.num_steps:
            break
        title_list[batch_cursor, 0, idx] = self.getWid(titles1[idx].strip())
        title_mask_weight[batch_cursor, 0, idx] = 1.0
      title_length_list[batch_cursor, 0] = len(titles1) if len(titles1)<=self.num_steps else self.num_steps

      for idx, word in enumerate(titles2):
        if idx>=self.num_steps:
            break
        title_list[batch_cursor, 1, idx] = self.getWid(titles2[idx].strip())
        title_mask_weight[batch_cursor, 1, idx] = 1.0
      title_length_list[batch_cursor, 1] = len(titles2) if len(titles2)<=self.num_steps else self.num_steps

      batch_cursor += 1
      if batch_cursor==self.batch_size:
        return True, query_list, query_length_list, query_mask_weight, title_list, title_length_list, title_mask_weight

