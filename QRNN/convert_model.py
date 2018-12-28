# ==============================================================================
# Author: shixiang08abc@gmail.com
# Copyright 2018 Sogou Inc. All Rights Reserved.
# ==============================================================================

import struct
import tensorflow as tf

if __name__=="__main__":
  word_tabel = []
  rfin = open("word_table_merge", 'r')
  lines = rfin.readlines()
  for line in lines:
    line = line.strip()
    if line=="":
      continue
    word_tabel.append(line)
  rfin.close()

  wfin = open("qrnn.model",'w')
  wbfin = open("qrnn.model.bin",'wb')
  tf_ckpt = '/data/shixiang/qrnn_imp/qanchor_train/models/qrnn_imp_upweight_refine_100'
  reader = tf.train.NewCheckpointReader(tf_ckpt)

  #cell_size filter_width word_embedding target_num
  wfin.write(str(256) + " " + str(2) + " " + str(300) + " " + str(1) + "\n")

  cell_size = struct.pack("i", 256)
  wbfin.write(cell_size)
  filter_width = struct.pack("i", 2)
  wbfin.write(filter_width)
  word_dim = struct.pack("i", 300)
  wbfin.write(word_dim)
  target_num = struct.pack("i", 1)
  wbfin.write(target_num)

  ####################convert conv_w#####################
  mat = reader.get_tensor("QuasiRNN/ConvZFO/conv_w")
  row = 600
  col = 768
  wfin.write("conv_w" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("conv_w ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row/2):
    dim1 = mat[0][i]
    dim2 = mat[1][i]
    scores = ""
    for j in range(col):
      wbfin.write(struct.pack("f", dim1[j]))
      scores = scores + " " + str(dim1[j])
    wfin.write(scores.strip()+"\n")
    scores = ""
    for j in range(col):
      wbfin.write(struct.pack("f", dim2[j]))
      scores = scores + " " + str(dim2[j])
    wfin.write(scores.strip()+"\n")

  ####################convert conv_b#####################
  mat = reader.get_tensor("QuasiRNN/ConvZFO/conv_b")
  row = 1
  col = 768
  wfin.write("conv_b" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("conv_b ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    wbfin.write(struct.pack("f", mat[i]))
    scores = scores + " " + str(mat[i])
  wfin.write(scores.strip()+"\n")
    
  ####################convert softmax_fw_w#####################
  mat = reader.get_tensor("softmax_fw_w")
  row = 1
  col = 256
  wfin.write("softmax_fw_w" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_w ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    wbfin.write(struct.pack("f", mat[i][0]))
    scores = scores + " " + str(mat[i][0])
  wfin.write(scores.strip()+"\n")
    
  ####################convert softmax_fw_b#####################
  mat = reader.get_tensor("softmax_fw_b")
  row = 1
  col = 1
  wfin.write("softmax_fw_b" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_b ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    wbfin.write(struct.pack("f", mat[i]))
    scores = scores + " " + str(mat[i])
  wfin.write(scores.strip()+"\n")

  ##################convert word embedding#################
  mat = reader.get_tensor("embedding")
  row = 450001
  col = 301
  wfin.write("embedding" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("embedding ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = word_tabel[i]
    wbfin.write(word_tabel[i]+" ")
    for j in range(col-1):
      wbfin.write(struct.pack("f", dim[j]))
      scores = scores + " " + str(dim[j])
    wfin.write(scores.strip()+"\n")
  
  wfin.close()
  wbfin.close()

