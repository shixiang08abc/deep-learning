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

class TransFormerLayer(object):
  def __init__(self, config):
    self.batch_size = config.batch_size
    self.num_steps = config.max_length
    self.learning_rate = config.learning_rate
    self.target_delay = config.target_delay
    self.vocab_size = config.vocab_size
    self.embedding_dim = config.embedding_dim
    self.tag_size = config.tag_size
    self.tag_dim = config.tag_dim
    self.parse_size = config.parse_size
    self.parse_dim = config.parse_dim

    self.out_dim = config.out_dim
    self.feed_size = config.feed_size
    self.model_size = config.model_size
    self.num_attentions = config.num_attentions

  def create_initializer(self, initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)

  def gelu(self, input_tensor):
    cdf = 0.5 * (1.0 + tf.erf(input_tensor / tf.sqrt(2.0)))
    return input_tensor*cdf

  def ln(self, tensor, scope='NormLayer', epsilon=1e-5):
    m, v = tf.nn.moments(tensor, [-1], keep_dims=True)
    with tf.variable_scope(scope):
      scale = tf.get_variable(name='scale', shape=[tensor.get_shape()[-1]], initializer=tf.constant_initializer(1.0))
      shift = tf.get_variable(name='shift', shape=[tensor.get_shape()[-1]], initializer=tf.constant_initializer(0))
      LN_initial = (tensor - m) / tf.sqrt(v + epsilon)
      return LN_initial*scale + shift

  def multiHeadAtt(self, Q, Qm, K=None, Km=None, d_model=self.model_size, d_out=self.out_dim, num_att=self.num_attentions, keep_prob=0.9, is_training=True, mode='residual', scope='MultiHeadAtt', att_fun=tf.nn.relu):
    d_k = int(d_model / num_att)
    assert(d_k*num_att == d_model)
    batch_size = tf.shape(Q)[0]
    cur_q_length = tf.shape(Qm)[-1]

    if K is None:
      K, V = Q, Q
      Km = Qm
      cur_k_length = cur_q_length
    else:
      V = K
      cur_k_length = tf.shape(Km)[-1]

    with tf.variable_scope(scope, regularizer=tf.contrib.layers.sum_regularizer([tf.contrib.layers.l2_regularizer(1.0), tf.contrib.layers.l1_regularizer(0.5)])):
      QW = tf.contrib.layers.fully_connected(Q, num_att*d_k, att_fun, scope='LinearQ', weights_initializer=create_initializer())
      QW = tf.reshape(QW, [batch_size, cur_q_length, num_att, d_k])
      QW = tf.transpose(QW, [0, 2, 1, 3])

      KW = tf.contrib.layers.fully_connected(K, num_att*d_k, att_fun, scope='LinearK', weights_initializer=create_initializer())
      KW = tf.reshape(KW, [batch_size, cur_k_length, num_att, d_k])
      KW = tf.transpose(KW, [0, 2, 1, 3])

      VW = tf.contrib.layers.fully_connected(V, num_att*d_k, att_fun, scope='LinearV', weights_initializer=create_initializer())
      VW = tf.reshape(VW, [batch_size, cur_k_length, num_att, d_k])
      VW = tf.transpose(VW, [0, 2, 1, 3])

      a = tf.batch_matmul(QW, tf.transpose(KW, [0, 1, 3, 2]), name='scaled_dot')
      a /= tf.sqrt(1e-6 + tf.to_float(d_k))
      mask_a = tf.batch_matmul(tf.reshape(Qm, [batch_size, cur_q_length, 1]), tf.reshape(Km, [batch_size, 1, cur_k_length]), name='dot_mask')
      a *= mask_a[:, None, :, :]
      tao = tf.get_variable(name='temperature', shape=[1], initializer=tf.constant_initializer(0.5))
      a /= (tf.abs(tao) + 1e-4)

      valid_offset = tf.reduce_max(a - 10.0 * (1.0 - mask_a[:, None, :, :]), 3, True, name='valid_offset')
      exp_a = tf.exp(a - valid_offset, name='exp_a') + 1e-8
      exp_a_masked = exp_a * mask_a[:, None, :, :]
      e = exp_a_masked / (1e-8 + tf.reduce_sum(exp_a_masked, 3))[:,:,:,None]

      c = tf.batch_matmul(e, VW)
      c = tf.reshape(tf.transpose(c, [0, 2, 1, 3]), [batch_size * cur_q_length, num_att * d_k])
      O = tf.contrib.layers.fully_connected(c, d_out, att_fun, scope='LinearO', weights_initializer=create_initializer())
      mh = tf.reshape(O, [batch_size, cur_q_length, d_out])
      mh *= Qm[:,:,None]
      mh = tf.contrib.layers.dropout(mh, keep_prob=keep_prob, is_training=is_training)
      if mode=='residual':
        residual_output = mh + Q
        residual_output = ln(residual_output, scope=scope)
      elif mode=='concat':
        residual_output = tf.concat([Q, mh], 2)
        residual_output = ln(residual_output, scope=scope)
      else:
        residual_output = mh
        residual_output = ln(residual_output, scope=scope)
      residual_output *= Qm[:, :, None]
     
      return residual_output

  def posWiseForawrd(self, seq_inp, seq_mask, d_out=self.out_dim, d_ff=self.feed_size, keep_prob=0.9, is_training=True, mode='residual', scope='posFeedForward', ff_fun=tf.nn.tanh):
    with tf.variable_scope(scope, regularizer=tf.contrib.layers.l2_regularizer(1.0)):
      relu_layer = tf.contrib.layers.fully_connected(seq_inp, d_ff, self.gelu, scope='Linear1', weights_initializer=self.create_initializer())
      relu_layer *= seq_mask[:,:,None]
      final_layer = tf.contrib.layers.fully_connected(relu_layer, d_out, ff_fun, scope='Linear2', weights_initializer=self.create_initializer())
      final_layer *= seq_mask[:,:,None]
      final_layer = tf.contrib.layers.dropout(final_layer, keep_prob=keep_prob, is_training=is_training)
      if mode=='residual':
        residual_output = final_layer + seq_inp
        residual_output = self.ln(residual_output, scope=scope+'layer_norm')
      elif mode=='concat':
        residual_output = tf.concat([final_layer, seq_inp], 2)
        residual_output = self.ln(residual_output, scope=scope+'layer_norm')
      else:
        residual_output = final_layer
        residual_output = self.ln(residual_output, scope=scope+'layer_norm')
      residual_output *= seq_mask[:, :, None]

      return residual_output

  def posEncoding(self, max_length=self.num_steps, emb_dim=self.out_dim, mode='sinusoid', scope='PosEmbd_Lookup'):
    with tf.variable_scope(scope):
      if mode=='sinusoid':
        P = tf.range(max_length, dtype=tf.float32)
        P = tf.tile(tf.reshape(P, [max_length, 1]), [1, emb_dim])
        D = tf.range(emb_dim, dtype=tf.float32)
        D_mod_2 = D - tf.to_float(tf.floor(D / 2.0) * 2.0)
        D = 10000.0 ** ((D - D_mod_2) / tf.to_float(emb_dim + 1e-6))
        D = tf.tile(tf.reshape(D, [1, emb_dim]), [max_length, 1])
        PE = tf.sin(P / D + 0.5 * np.pi * D_mod_2[None, :], name='pos_embedding')
      else:
        PE = tf.get_variable('pos_embedding', [max_length, emb_dim], dtype=tf.float32, initializer=self.create_initializer())
    return PE

  def transformerLayer(self, inputs_proj, inputs_mask, layer_id, num_att=self.num_attentions, d_model=self.model_size, d_out=self.out_dim, d_ff=self.feed_size, keep_prob=0.9, is_training=True, mode='residual', scope='Transformer', att_fun=tf.nn.relu, ff_fun=tf.nn.relu):
    with tf.variable_scope(scope+str(layer_id)):
      sub_layer_1 = multiHeadAtt(self, inputs_proj, inputs_mask, d_model, d_out, num_att, keep_prob, is_training, mode, 'MultiHeadAtt_'+scope+str(layer_id), att_fun)
      sub_layer_2 = posWiseForawrd(self, sub_layer_1, inputs_mask, d_out, d_ff, keep_prob, is_training, mode, 'PosFeedForward_'+scope+str(layer_id), ff_fun)
    return sub_layer_2
 
