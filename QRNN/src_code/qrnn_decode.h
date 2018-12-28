// =====================================================================
// Author: shixiang08abc@gmail.com
// Copyright 2018 Sogou Inc. All Rights Reserved.
// =====================================================================
#ifndef QRNN_DECODE_H
#define QRNN_DECODE_H

#include <iostream>
#include <string>
#include <vector>
#include <map>

class QrnnDecode
{
	public:
		QrnnDecode();
		~QrnnDecode();
		bool split(const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens);
		bool ReadModelBin(const std::string& model_path);
		bool CreateModelDict(const std::string& model_path, const std::string& model_dict_path);
		size_t getWordId(std::string word);
		void getImportant(std::vector<std::string>& words, std::vector<float>& scores);
		void Propagate(const float* mat_input, size_t input_row, size_t input_col, float* mat_output, size_t output_row, size_t output_col);
		void printQrnnParaMeter();
		void init_fast_tanh();
		float fast_tanh(float score);

	private:
		//embedding
		std::vector<float* > word_embedding_;

		//conv layer
		float* mat_conv_w_zfo_;
		float* vec_conv_bias_zfo_;

		//affine
		float* mat_affine_w_;
		float* vec_affine_b_;

		std::map<std::string, size_t> word2id_;
		std::vector<std::string> id2word_;
		size_t word_dim_;
		size_t word_size_;

		size_t cell_size_;
		size_t filter_width_;
		size_t target_num_;

		size_t tanh_size_;
		float* tanh_cache_;
};

#endif
