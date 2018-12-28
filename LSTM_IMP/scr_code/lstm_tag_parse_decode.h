// =====================================================================
// Author: shixiang08abc@gmail.com
// Copyright 2017 Sogou Inc. All Rights Reserved.
// =====================================================================
#ifndef LSTM_TAG_PARSE_DECODE_H
#define LSTM_TAG_PARSE_DECODE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

class LstmTagParseDecode
{
	public:
		LstmTagParseDecode();
		~LstmTagParseDecode();
		bool split(const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens);
		bool ReadModelBin(const std::string& model_path);
		size_t getWordId(std::string word);
		size_t getTagIdx(std::string hexTag);
		void getImportant(std::vector<std::string>& words, std::vector<size_t>& tags,  std::vector<size_t>& heads, std::vector<size_t>& deprels, std::vector<float>& scores);
		void Propagate(const float* mat_input, size_t input_row, size_t input_col, float* mat_output, size_t output_row, size_t output_col);
		void printLstmPareMeter();

	private:
		//embedding
		std::vector<float* > word_embedding_;
		std::vector<float* > tag_embedding_;
		std::vector<float* > parse_embedding_;

		//cell kernel
		float* mat_lstm_w_ijfo_x_;	//recurrent lstm cell: from x to [i, j, f, o]
		float* mat_lstm_w_ijfo_h_; 	//recurrent lstm cell: from h to [i, j, f, o]
		float* vec_lstm_bias_ijfo_;	//biases of [i, j, f, o]

		//peephole from c to f, i, o
		float* vec_lstm_peephole_f_c_;
		float* vec_lstm_peephole_i_c_;
		float* vec_lstm_peephole_o_c_;
		
		//projection
		float* mat_lstm_w_proj_;		//from r to h

		//affine
		float* mat_affine_w_;
		float* vec_affine_b_;

		std::map<std::string, size_t> word2id_;
		std::vector<std::string> id2word_;
		size_t word_dim_;
		size_t word_size_;
		size_t tag_dim_;
		size_t tag_size_;
		size_t parse_dim_;
		size_t parse_size_;
			
		size_t cell_size_;
		size_t proj_num_;
		size_t target_num_;
		size_t target_delay_;
};

#endif
