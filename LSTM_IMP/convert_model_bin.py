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

  wfin = open("model.tag.parse.lstm",'wb')
  wbfin = open("model.tag.parse.lstm.bin",'wb')
  tf_ckpt = '/data/shixiang/parse_imp/new3_train/models/lstmp_imp_new_refine_223'
  reader = tf.train.NewCheckpointReader(tf_ckpt)

  #cell_size proj_num word_embedding tag_embedding parse_embedding target_num
  wfin.write(str(256) + " " + str(128) + " " + str(100) + " " + str(10) + " " + str(20) + " " + str(1) + "\n")

  cell_size = struct.pack("i", 256)
  wbfin.write(cell_size)

  proj_num = struct.pack("i", 128)
  wbfin.write(proj_num)

  word_dim = struct.pack("i", 100)
  wbfin.write(word_dim)

  tag_dim = struct.pack("i", 10)
  wbfin.write(tag_dim)

  parse_dim = struct.pack("i", 20)
  wbfin.write(parse_dim)

  target_num = struct.pack("i", 1)
  wbfin.write(target_num)

  ##################convert word embedding#################
  mat = reader.get_tensor("embedding")
  row = 450002
  col = 101
  wfin.write("embedding" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("embedding ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = word_tabel[i]
    wbfin.write(word_tabel[i]+" ")
    for j in range(col-1):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")

  ##################convert tag embedding#################
  mat = reader.get_tensor("tag_embedding")
  row = 33
  col = 10
  wfin.write("tag_embedding" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("tag_embedding ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")
  
  ##################convert parse embedding#################
  mat = reader.get_tensor("parse_embedding")
  row = 16
  col1 = 100
  col2 = 20
  wfin.write("parse_embedding" + " " + str(row) + " " + str(col1*col2) + "\n")
  wbfin.write("parse_embedding ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col1*col2))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col1):
      dim2 = dim[j]
      for k in range(col2):
        scores = scores + " " + str(dim2[k])
        wbfin.write(struct.pack("f", dim2[k]))
    wfin.write(scores.strip()+"\n")
  
  ##################convert lstm_cell kernel#################
  mat = reader.get_tensor("rnn/lstm_cell/kernel")
  row = 130
  col = 1024
  wfin.write("lstm_cell_x" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_x ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")

  row2 = 128
  wfin.write("lstm_cell_h" + " " + str(row2) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_h ")
  wbfin.write(struct.pack("i", row2))
  wbfin.write(struct.pack("i", col))
  for i in range(row2):
    dim = mat[i+row]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")

  ##################convert lstm_cell bias#################
  mat = reader.get_tensor("rnn/lstm_cell/bias")
  row = 1
  col = 1024
  wfin.write("lstm_cell_bias" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_bias ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert w_f_diag#################
  mat = reader.get_tensor("rnn/lstm_cell/w_f_diag")
  row = 1
  col = 256
  wfin.write("w_f_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("w_f_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert w_i_diag#################
  mat = reader.get_tensor("rnn/lstm_cell/w_i_diag")
  row = 1
  col = 256
  wfin.write("w_i_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("w_i_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert w_o_diag#################
  mat = reader.get_tensor("rnn/lstm_cell/w_o_diag")
  row = 1
  col = 256
  wfin.write("w_o_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("w_o_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert projection#################
  mat = reader.get_tensor("rnn/lstm_cell/projection/kernel")
  row = 256
  col = 128
  wfin.write("projection" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("projection ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")
  
  ##################convert softmax_fw_w#################
  mat = reader.get_tensor("softmax_fw_w")
  row = 1
  col = 128
  wfin.write("softmax_fw_w" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_w ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    dim = mat[i]
    for j in range(1):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert softmax_fw_b#################
  mat = reader.get_tensor("softmax_fw_b")
  row = 1
  col = 1
  wfin.write("softmax_fw_b" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_b ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")

  wfin.close()
  wbfin.close()
