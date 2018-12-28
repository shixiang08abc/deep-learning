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

  wfin = open("model.crf.lstm",'w')
  wbfin = open("model.crf.lstm.bin", 'wb')
  tf_ckpt = '/data/shixiang/termweight/online_train/models/lstmp_tw_refine_199'
  reader = tf.train.NewCheckpointReader(tf_ckpt)

  #cell_size proj_num word_embedding target_num
  wfin.write(str(256) + " " + str(128) + " " + str(100) + " " + str(3) + "\n")

  cell_size = struct.pack("i", 256)
  wbfin.write(cell_size)

  proj_num = struct.pack("i", 128)
  wbfin.write(proj_num)

  word_dim = struct.pack("i", 100)
  wbfin.write(word_dim)

  target_num = struct.pack("i", 3)
  wbfin.write(target_num)
  
  ##################convert word embedding#################
  mat = reader.get_tensor("embedding")
  row = 450001
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
  
  ##################convert fw lstm_cell kernel#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/kernel")
  row = 100
  col = 1024
  wfin.write("lstm_cell_fw_x" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_fw_x ")
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
  wfin.write("lstm_cell_fw_h" + " " + str(row2) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_fw_h ")
  wbfin.write(struct.pack("i", row2))
  wbfin.write(struct.pack("i", col))
  for i in range(row2):
    dim = mat[i+row]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")

  ##################convert fw lstm_cell bias#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/bias")
  row = 1
  col = 1024
  wfin.write("lstm_cell_fw_bias" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_fw_bias ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert fw w_f_diag#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/w_f_diag")
  row = 1
  col = 256
  wfin.write("fw_w_f_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("fw_w_f_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert fw w_i_diag#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/w_i_diag")
  row = 1
  col = 256
  wfin.write("fw_w_i_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("fw_w_i_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert fw w_o_diag#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/w_o_diag")
  row = 1
  col = 256
  wfin.write("fw_w_o_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("fw_w_o_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert fw projection#################
  mat = reader.get_tensor("bidirectional_rnn/fw/lstm_cell/projection/kernel")
  row = 256
  col = 128
  wfin.write("fw_projection" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("fw_projection ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")
  
  ##################convert bw lstm_cell kernel#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/kernel")
  row = 100
  col = 1024
  wfin.write("lstm_cell_bw_x" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_bw_x ")
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
  wfin.write("lstm_cell_bw_h" + " " + str(row2) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_bw_h ")
  wbfin.write(struct.pack("i", row2))
  wbfin.write(struct.pack("i", col))
  for i in range(row2):
    dim = mat[i+row]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")

  ##################convert bw lstm_cell bias#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/bias")
  row = 1
  col = 1024
  wfin.write("lstm_cell_bw_bias" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("lstm_cell_bw_bias ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert bw w_f_diag#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/w_f_diag")
  row = 1
  col = 256
  wfin.write("bw_w_f_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("bw_w_f_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert bw w_i_diag#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/w_i_diag")
  row = 1
  col = 256
  wfin.write("bw_w_i_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("bw_w_i_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert bw w_o_diag#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/w_o_diag")
  row = 1
  col = 256
  wfin.write("bw_w_o_diag" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("bw_w_o_diag ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
  
  ##################convert bw projection#################
  mat = reader.get_tensor("bidirectional_rnn/bw/lstm_cell/projection/kernel")
  row = 256
  col = 128
  wfin.write("bw_projection" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("bw_projection ")
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
  row = 3
  col = 256
  wfin.write("softmax_fw_w" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_w ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    scores = ""
    for j in range(col):
      dim = mat[j]
      scores = scores + " " + str(dim[i])
      wbfin.write(struct.pack("f", dim[i]))
    wfin.write(scores.strip()+"\n")
  
  ##################convert softmax_fw_b#################
  mat = reader.get_tensor("softmax_fw_b")
  row = 1
  col = 3
  wfin.write("softmax_fw_b" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("softmax_fw_b ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  scores = ""
  for i in range(col):
    scores = scores + " " + str(mat[i])
    wbfin.write(struct.pack("f", mat[i]))
  wfin.write(scores.strip()+"\n")
 
  ##################convert transitions#################
  mat = reader.get_tensor("transitions")
  row = 3
  col = 3
  wfin.write("transitions" + " " + str(row) + " " + str(col) + "\n")
  wbfin.write("transitions ")
  wbfin.write(struct.pack("i", row))
  wbfin.write(struct.pack("i", col))
  for i in range(row):
    dim = mat[i]
    scores = ""
    for j in range(col):
      scores = scores + " " + str(dim[j])
      wbfin.write(struct.pack("f", dim[j]))
    wfin.write(scores.strip()+"\n")
  
  wfin.close()
  wbfin.close()
