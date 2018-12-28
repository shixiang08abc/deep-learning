// =====================================================================
// Author: shixiang08abc@gmail.com
// Copyright 2018 Sogou Inc. All Rights Reserved.
// =====================================================================
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include "qrnn_decode.h"

QrnnDecode::QrnnDecode()
{
	mat_conv_w_zfo_ = NULL;
	vec_conv_bias_zfo_ = NULL;
	
	mat_affine_w_ = NULL;
	vec_affine_b_ = NULL;

	word_dim_ = 0;
	word_size_ = 0;
	
	cell_size_ = 0;
	target_num_ = 0;
	filter_width_ = 0;
}

QrnnDecode::~QrnnDecode()
{
	if(mat_conv_w_zfo_!=NULL)
		delete mat_conv_w_zfo_;
	if(vec_conv_bias_zfo_!=NULL)
		delete vec_conv_bias_zfo_;

	if(mat_affine_w_!=NULL)
		delete mat_affine_w_;
	if(vec_affine_b_!=NULL)
		delete vec_affine_b_;
}

bool QrnnDecode::split(const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens)
{
	if (str.empty())
		return false;
	tokens.clear();
	std::string::size_type lastPos = 0;
	std::string::size_type pos = str.find(delimiter, lastPos);
	if(pos == std::string::npos)
	{
		tokens.push_back(str);
		return true;
	}

	while (std::string::npos != pos)
	{
		tokens.push_back(str.substr(lastPos, pos - lastPos));
		lastPos = pos+delimiter.size();
		pos = str.find(delimiter, lastPos);	
	}
	std::string last = str.substr(lastPos);
	if(!last.empty())
		tokens.push_back(last);
	return true;
}

bool QrnnDecode::CreateModelDict(const std::string& model_path, const std::string& model_dict_path)
{
	std::ifstream ifs;
	ifs.open(model_path.c_str(), std::ifstream::in|std::ifstream::binary);
	if(!ifs)
	{
		std::cerr << "Fail to open " << model_path << std::endl;
		return false;
	}

	std::ofstream ofs;
	ofs.open(model_dict_path.c_str(), std::ofstream::out|std::ofstream::binary);
	if(!ofs)
	{
		std::cerr << "Fail to create " << model_dict_path << std::endl;
		return false;
	}
	
	int feat = 0;

	// load qrnn parameter
	ifs.read((char*)&feat, sizeof(int));
	cell_size_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	filter_width_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	word_dim_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	target_num_ = feat;
	
	ofs.write((char*)&cell_size_, sizeof(int));
	ofs.write((char*)&filter_width_, sizeof(int));
	ofs.write((char*)&word_dim_, sizeof(int));
	ofs.write((char*)&target_num_, sizeof(int));

	std::string str;
	// load qrnn conv weight z,f,o
	str.clear();
	ifs >> str;
	if(str == "conv_w")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		str = str + " ";	
		ofs.write(str.c_str(), sizeof(char)*(str.size()));
		ofs.write((char*)&row, sizeof(int));
		ofs.write((char*)&column, sizeof(int));
		
		if((row==word_dim_*filter_width_) && (column==3*cell_size_))
		{
			mat_conv_w_zfo_ = new float[row * column];
			ifs.read((char*)mat_conv_w_zfo_, row*column*sizeof(float));

			ofs.write((char*)mat_conv_w_zfo_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, conv_w parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, conv_w is not expect!!!" << std::endl;
		return false;
	}

	// load qrnn conv biases z,f,o
	str.clear();
	ifs >> str;
	if(str == "conv_b")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		str = str + " ";	
		ofs.write(str.c_str(), sizeof(char)*(str.size()));
		ofs.write((char*)&row, sizeof(int));
		ofs.write((char*)&column, sizeof(int));
		
		if((row==1) && (column==3*cell_size_))
		{
			vec_conv_bias_zfo_ = new float[row*column];
			ifs.read((char*)vec_conv_bias_zfo_, row*column*sizeof(float));

			ofs.write((char*)vec_conv_bias_zfo_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, conv_b parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, conv_b is not expect!!!" << std::endl;
		return false;
	}

	// load affine weights
	str.clear();
	ifs >> str;
	if(str == "softmax_fw_w")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		str = str + " ";	
		ofs.write(str.c_str(), sizeof(char)*(str.size()));
		ofs.write((char*)&row, sizeof(int));
		ofs.write((char*)&column, sizeof(int));
		
		if((row==1) && (column==cell_size_))
		{
			mat_affine_w_ = new float[row*column];
			ifs.read((char*)mat_affine_w_, row*column*sizeof(float));

			ofs.write((char*)mat_affine_w_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, softmax_fw_w parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, softmax_fw_w is not expect!!!" << std::endl;
		return false;
	}

	// load affine biase
	str.clear();
	ifs >> str;
	if(str == "softmax_fw_b")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		str = str + " ";	
		ofs.write(str.c_str(), sizeof(char)*(str.size()));
		ofs.write((char*)&row, sizeof(int));
		ofs.write((char*)&column, sizeof(int));
		
		if((row==1) && (column==target_num_))
		{
			vec_affine_b_ = new float[row*column];
			ifs.read((char*)vec_affine_b_, row*column*sizeof(float));

			ofs.write((char*)vec_affine_b_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, softmax_fw_b parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, softmax_fw_b is not expect!!!" << std::endl;
		return false;
	}

	// load word embedding
	str.clear();
        ifs >> str;

	if(str == "embedding")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		word_size_ = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;
		
		str = "embedding_dict ";
		size_t outcol = cell_size_*filter_width_*3 + 1;		//3 is GRU model gate, 1 is word
		ofs.write(str.c_str(), sizeof(char)*(str.size()));
		ofs.write((char*)&word_size_, sizeof(int));
		ofs.write((char*)&outcol, sizeof(int));
		
		if(column == word_dim_+1)
		{
			for(size_t i=0; i<word_size_; i++)
			{
				str.clear();
				ifs >> str;
				ifs.get();

				float* embedding = new float[word_dim_];
				ifs.read((char*)embedding, word_dim_*sizeof(float));
				
				float* embedding_dict = new float[outcol-1];
				for(size_t j=0; j<filter_width_; j++)
				{
					for(size_t k=0; k<3*cell_size_; k++)
					{
						float sum = 0.0;
						for(size_t m=0; m<word_dim_; m++)
						{
							sum += embedding[m] * mat_conv_w_zfo_[(m*filter_width_+j)*3*cell_size_ + k];
						}
						embedding_dict[k + j*3*cell_size_] = sum;
					}
					
				}
				
				str = str + " ";
				ofs.write(str.c_str(), sizeof(char)*(str.size()));
				ofs.write((char*)embedding_dict, (outcol-1)*sizeof(float));
				std::cout << str;
				for(size_t j=0; j<outcol-1; j++)
					std::cout << " " << embedding_dict[j];
				std::cout << std::endl;

				delete[] embedding;
				delete[] embedding_dict;
			}
		}
		else
		{
			std::cerr << "Model format error, embedding parameter colomn is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, embedding is not expect!!!" << std::endl;
		return false;
	}

	ifs.close();
	ofs.close();
	return true;
}

bool QrnnDecode::ReadModelBin(const std::string& model_path)
{

	std::ifstream ifs;
	ifs.open(model_path.c_str(), std::ifstream::in|std::ifstream::binary);
	if(!ifs)
	{
		std::cerr << "Fail to open " << model_path << std::endl;
		return false;
	}
	
	int feat = 0;

	// load qrnn parameter
	ifs.read((char*)&feat, sizeof(int));
	cell_size_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	filter_width_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	word_dim_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	target_num_ = feat;

	//std::cout << cell_size_ << "\t" << filter_width_ << "\t" << word_dim_ << "\t" << target_num_ << std::endl;

	std::string str;
	// load qrnn conv weight z,f,o
	str.clear();
	ifs >> str;
	if(str == "conv_w")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==word_dim_*filter_width_) && (column==3*cell_size_))
		{
			mat_conv_w_zfo_ = new float[row * column];
			ifs.read((char*)mat_conv_w_zfo_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, conv_w parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, conv_w is not expect!!!" << std::endl;
		return false;
	}

	// load qrnn conv biases z,f,o
	str.clear();
	ifs >> str;
	if(str == "conv_b")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==3*cell_size_))
		{
			vec_conv_bias_zfo_ = new float[row*column];
			ifs.read((char*)vec_conv_bias_zfo_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, conv_b parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, conv_b is not expect!!!" << std::endl;
		return false;
	}

	// load affine weights
	str.clear();
	ifs >> str;
	if(str == "softmax_fw_w")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==cell_size_))
		{
			mat_affine_w_ = new float[row*column];
			ifs.read((char*)mat_affine_w_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, softmax_fw_w parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, softmax_fw_w is not expect!!!" << std::endl;
		return false;
	}

	// load affine biase
	str.clear();
	ifs >> str;
	if(str == "softmax_fw_b")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==target_num_))
		{
			vec_affine_b_ = new float[row*column];
			ifs.read((char*)vec_affine_b_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, softmax_fw_b parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, softmax_fw_b is not expect!!!" << std::endl;
		return false;
	}

	// load word embedding
	str.clear();
        ifs >> str;
	if(str == "embedding")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		word_size_ = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;
		if(column == word_dim_+1)
		{
			for(size_t i=0; i<word_size_; i++)
			{
				str.clear();
				ifs >> str;
				ifs.get();

				word2id_.insert(std::make_pair(str, id2word_.size()));
				id2word_.push_back(str);
				float* embedding = new float[word_dim_];
				ifs.read((char*)embedding, word_dim_*sizeof(float));
				word_embedding_.push_back(embedding);
			}
		}
		else
		{
			std::cerr << "Model format error, embedding parameter colomn is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, embedding is not expect!!!" << std::endl;
		return false;
	}

	ifs.close();
	return true;
}

size_t QrnnDecode::getWordId(std::string word)
{
	size_t wid = word_size_ - 1;
	std::map<std::string, size_t>::iterator iter = word2id_.find(word);
	if(iter != word2id_.end())
		wid = iter->second;
	return wid;
}

void QrnnDecode::Propagate(const float* mat_input, size_t input_row, size_t input_col, float* mat_output, size_t output_row, size_t output_col)
{
	float input_zfo_buf[(input_row-filter_width_+1)*3*cell_size_];
	float cell_buf[(input_row-filter_width_+2)*cell_size_];

	memset(input_zfo_buf, 0.0, (input_row-filter_width_+1)*3*cell_size_*sizeof(float));
	memset(cell_buf, 0.0, (input_row-filter_width_+2)*cell_size_);

	for(size_t i=0; i<input_row-filter_width_+1; i++)
		memcpy(input_zfo_buf+i*3*cell_size_, vec_conv_bias_zfo_, 3*cell_size_*sizeof(float));

	// cala conv layer 
	// input * _w_zfo_
	// mat_input[(words.size() + filter_width_ - 1)*word_dim_]
	// mat_conv_w_zfo_[(word_dim_*filter_width_)*3*cell_size_]
	// input_zfo_buf[(input_row-filter_width_+1)*3*cell_size_]
	for(size_t i=0; i<input_row-filter_width_+1; i++)
	{
		for(size_t j=0; j<3*cell_size_; j++)
		{
			float sum = 0.0;
			for(size_t k=0; k<word_dim_; k++)
			{
				for(size_t m=0; m<filter_width_; m++)
				{
					sum += mat_input[(i+m)*word_dim_+k]*mat_conv_w_zfo_[(k*filter_width_+m)*3*cell_size_+j];
				}
			}
			input_zfo_buf[i*3*cell_size_+j] += sum;
		}
	}

	//std::cout << "input_zfo_buf" << std::endl;
	//for(size_t i=0; i<input_row-filter_width_+1; i++)
	//{
	//	for(size_t j=0; j<3*cell_size_; j++)
	//		std::cout << input_zfo_buf[i*3*cell_size_+j] << " ";
	//	std::cout << std::endl;
	//}

	//cala input_zfo_buf tanh z, sigmoid f,o 
	for(size_t i=0; i<input_row-filter_width_+1; i++)
	{
		for(size_t j=0; j<cell_size_; j++)
		{
			input_zfo_buf[i*3*cell_size_+j] = tanh(input_zfo_buf[i*3*cell_size_+j]);
			input_zfo_buf[i*3*cell_size_+cell_size_+j] = 0.5*(tanh(input_zfo_buf[i*3*cell_size_+cell_size_+j]*0.5) + 1);
			input_zfo_buf[i*3*cell_size_+2*cell_size_+j] = 0.5*(tanh(input_zfo_buf[i*3*cell_size_+2*cell_size_+j]*0.5) + 1);
		}
	}
	
	//std::cout << "input_zfo_buf after sigmoid tanh" << std::endl;
	//for(size_t i=0; i<input_row-filter_width_+1; i++)
	//{
	//	for(size_t j=0; j<3*cell_size_; j++)
	//		std::cout << input_zfo_buf[i*3*cell_size_+j] << " ";
	//	std::cout << std::endl;
	//}
	
	//cala cell_buf
	//cell_buf[i] = (sigmoid_f*cell_buf[i-1] + (1.0-sigmoid_f)*tanh_z)*sigmoid_o
	for(size_t i=0; i<input_row-filter_width_+1; i++)
	{
		for(size_t j=0; j<cell_size_; j++)
		{
			cell_buf[(i+1)*cell_size_+j] = input_zfo_buf[i*3*cell_size_+cell_size_+j]*cell_buf[i*cell_size_+j] + (1.0-input_zfo_buf[i*3*cell_size_+cell_size_+j])*input_zfo_buf[i*3*cell_size_+j];
		}
	}

	//std::cout << "cell_buf" << std::endl;
	//for(size_t i=0; i<input_row-filter_width_+2; i++)
	//{
	//	for(size_t j=0; j<cell_size_; j++)
	//		std::cout << cell_buf[i*cell_size_+j] << " ";
	//	std::cout << std::endl;
	//}

	//cala mat_output
	//cell_buf[(input_row-filter_width_+2)*cell_size_]
	//mat_affine_w_[target_num_*cell_size_]
	//mat_input[(input_row-filter_width_+1)*target_num_]
	for(size_t i=0; i<input_row-filter_width_+1; i++)
	{
		for(size_t j=0; j<target_num_; j++)
		{
			float sum = vec_affine_b_[j];
			for(size_t k=0; k<cell_size_; k++)
			{
				sum += cell_buf[(i+1)*cell_size_+k]*input_zfo_buf[i*3*cell_size_+2*cell_size_+k]*mat_affine_w_[j*cell_size_+k];
			}
			mat_output[i*target_num_+j] = 0.5*(tanh(sum*0.5) + 1);
		}
	}

	//std::cout << "mat_output" << std::endl;
	//for(size_t i=0; i<input_row-filter_width_+1; i++)
	//{
	//	for(size_t j=0; j<target_num_; j++)
	//		std::cout << mat_output[i*target_num_+j] << " ";
	//	std::cout << std::endl;
	//}
}

void QrnnDecode::getImportant(std::vector<std::string>& words, std::vector<float>& scores)
{
	size_t input_row = words.size() + filter_width_ - 1;
	size_t input_col = word_dim_;
	float* mat_input = new float[input_row * input_col];

	size_t output_row = words.size();
	size_t output_col = target_num_;
	float* mat_output = new float[output_row * output_col];

	memset(mat_input, 0.0, input_row*input_col*sizeof(float));
	memset(mat_output, 0.0, output_row*output_col*sizeof(float));

	for(size_t i=0; i<words.size(); i++)
	{
		float* embedding = word_embedding_[getWordId(words[i])];
		for(size_t j=0; j<word_dim_; j++)
			mat_input[(i+filter_width_ - 1)*input_col + j] = embedding[j];
	}

	//std::cout << "mat_input" << std::endl;
	//for(size_t i=0; i<input_row; i++)
	//{
	//	for(size_t j=0; j<input_col; j++)
	//		std::cout << mat_input[i*input_col +j] << " ";
	//	std::cout << std::endl;
	//}

	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	Propagate(mat_input, input_row, input_col, mat_output, output_row, output_col);
	gettimeofday(&tv2, NULL);
	size_t span = (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
	//std::cout << "CPU time used: " << span << " us" << std::endl;

	/*std::cout << "mat_output:";
	for(size_t i=0; i<output_row*output_col; i++)
		std::cout << " " << mat_output[i];
	std::cout << std::endl;*/

	for(size_t i=0; i<words.size(); i++)
		scores.push_back(mat_output[i]);

	delete[] mat_input;
	delete[] mat_output;
}

void QrnnDecode::printQrnnParaMeter()
{
	std::cout << "word_dim_: " << word_dim_ << std::endl;
	std::cout << "word_size_: " << word_size_ << std::endl;
	std::cout << "cell_size_: " << cell_size_ << std::endl;
	std::cout << "filter_width_: " << filter_width_ << std::endl;
	std::cout << "target_num_: " << target_num_ << std::endl;

	std::cout << "word_embedding_" << std::endl;
	for(size_t i=0; i<word_size_; i++)
	{
		std::cout << i << "\t" << id2word_[i] << "\t" << word2id_[id2word_[i]] << "\t";
		for(size_t j=0; j<word_dim_; j++)
			std::cout << word_embedding_[i][j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "mat_conv_w_zfo_" << std::endl;
	for(size_t i=0; i<word_dim_*filter_width_; i++)
	{
		for(size_t j=0; j<3*cell_size_; j++)
			std::cout << mat_conv_w_zfo_[i*3*cell_size_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "vec_conv_bias_zfo_" << std::endl;
	for(size_t i=0; i<3*cell_size_; i++)
		std::cout << vec_conv_bias_zfo_[i] << " ";
	std::cout << std::endl;	

	std::cout << "mat_affine_w_" << std::endl;
	for(size_t i=0; i<target_num_; i++)
	{
		for(size_t j=0; j<cell_size_; j++)
			std::cout << mat_affine_w_[i*cell_size_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "vec_affine_b_" << std::endl;
	for(size_t i=0; i<target_num_; i++)
		std::cout << vec_affine_b_[i] << " ";
	std::cout << std::endl;	
}

int main(void)
{
	QrnnDecode model;

	/*if(!model.CreateModelDict("/data/shixiang/qrnn_imp/qrnn.model.bin", "/data/shixiang/qrnn_imp/qrnn.model.dict.bin"))
	{
		std::cout << "create model dict error!!!" << std::endl;
		return 0;
	}*/

	if(!model.ReadModelBin("/data/shixiang/qrnn_imp/qrnn.model.bin"))
	//if(!model.ReadModelDictBin("/data/shixiang/qrnn_imp/qrnn.model.dict.bin"))
	{
		std::cout << "read model error!!!" << std::endl;
		return 0;
	}
	//model.printQrnnParaMeter();

	std::string line;
	while(std::getline(std::cin, line))
	{
		std::vector<std::string> tokens;
		std::vector<std::string> words;
		model.split(line, "\t", tokens);
		if(tokens.size()!=3)
		{
			std::cerr << "input format error, must 3 tokens!!" << std::endl;
			continue;
		}

		model.split(tokens[0], " ", words);

		std::vector<float> scores;
		struct timeval tv1, tv2;
		gettimeofday(&tv1, NULL);
		model.getImportant(words, scores);
		gettimeofday(&tv2, NULL);
		size_t span = (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
		std::cout << "CPU time used: " << span << " us" << std::endl;

		std::cout << line << "\t";
		for(size_t i=0;i<scores.size();i++)
			std::cout << " " << scores[i];
		std::cout << std::endl;
	}

	std::cout << "pass~~~" << std::endl;
	return 0;
}
