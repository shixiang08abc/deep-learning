// =====================================================================
// Author: shixiang08abc@gmail.com
// Copyright 2017 Sogou Inc. All Rights Reserved.
// =====================================================================
#ifndef LSTM_TAG_CRF_DECODE_H
#define LSTM_TAG_CRF_DECODE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

class LstmTwTagDecode
{
	public:
		LstmTwTagDecode();
		~LstmTwTagDecode();
		bool split(const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens);
		bool ReadModelBin(const std::string& model_path);
		size_t getWordId(std::string word);
		size_t getTagIdx(std::string hexTag);
		void getTermWeight(std::vector<std::string>& words, std::vector<size_t>& tags, std::vector<size_t>& labels);
		void PropagateFw(const float* mat_fw_input, size_t input_row, size_t input_col, float* mat_fw_output, size_t output_row, size_t output_col);
		void PropagateBw(const float* mat_bw_input, size_t input_row, size_t input_col, float* mat_bw_output, size_t output_row, size_t output_col);
		void PropagateCrf(const float* mat_fw_output, const float* mat_bw_output, size_t output_row, size_t output_col, std::vector<size_t>& labels);
		void printLstmPareMeter();

	private:
		//embedding
		std::vector<float* > word_embedding_;
		std::vector<float* > tag_embedding_;

		//cell kernel
		float* mat_fw_lstm_w_ijfo_x_;	//recurrent fw lstm cell: from x to [i, j, f, o]
		float* mat_fw_lstm_w_ijfo_h_; 	//recurrent fw lstm cell: from h to [i, j, f, o]
		float* vec_fw_lstm_bias_ijfo_;	//fw biases of [i, j, f, o]
		float* mat_bw_lstm_w_ijfo_x_;	//recurrent bw lstm cell: from x to [i, j, f, o]
		float* mat_bw_lstm_w_ijfo_h_; 	//recurrent bw lstm cell: from h to [i, j, f, o]
		float* vec_bw_lstm_bias_ijfo_;	//bw biases of [i, j, f, o]

		//peephole from c to f, i, o
		float* vec_fw_lstm_peephole_f_c_;
		float* vec_fw_lstm_peephole_i_c_;
		float* vec_fw_lstm_peephole_o_c_;
		float* vec_bw_lstm_peephole_f_c_;
		float* vec_bw_lstm_peephole_i_c_;
		float* vec_bw_lstm_peephole_o_c_;
		
		//projection
		float* mat_fw_lstm_w_proj_;		//fw from r to h
		float* mat_bw_lstm_w_proj_;		//bw from r to h

		//affine
		float* mat_affine_w_;
		float* vec_affine_b_;

		//transitions
		float* mat_trans_;

		std::map<std::string, size_t> word2id_;
		std::vector<std::string> id2word_;
		size_t word_dim_;
		size_t word_size_;
		size_t tag_dim_;
		size_t tag_size_;
			
		size_t cell_size_;
		size_t proj_num_;
		size_t target_num_;
};

#endif
