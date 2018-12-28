// =====================================================================
// Author: shixiang08abc@gmail.com
// Copyright 2017 Sogou Inc. All Rights Reserved.
// =====================================================================
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <cstring>
#include <sys/time.h>
#include "lstm_tag_parse_decode.h"
extern "C" 
{
#include "cblas.h"
}


LstmTagParseDecode::LstmTagParseDecode()
{
	mat_lstm_w_ijfo_x_ = NULL;
	mat_lstm_w_ijfo_h_ = NULL;
	vec_lstm_bias_ijfo_ = NULL;
	
	vec_lstm_peephole_i_c_ = NULL;
	vec_lstm_peephole_f_c_ = NULL;
	vec_lstm_peephole_o_c_ = NULL;

	mat_lstm_w_proj_ = NULL;
	mat_affine_w_ = NULL;
	vec_affine_b_ = NULL;

	word_dim_ = 0;
	word_size_ = 0;
	tag_dim_ = 0;
	tag_size_ = 0;
	parse_dim_ = 0;
	parse_size_ = 0;
	
	cell_size_ = 0;
	proj_num_ = 0;
	target_num_ = 0;
	target_delay_ = 5;
}

LstmTagParseDecode::~LstmTagParseDecode()
{
	if(mat_lstm_w_ijfo_x_!=NULL)
		delete mat_lstm_w_ijfo_x_;
	if(mat_lstm_w_ijfo_h_!=NULL)
		delete mat_lstm_w_ijfo_h_;
	if(vec_lstm_bias_ijfo_!=NULL)
		delete vec_lstm_bias_ijfo_;

	if(vec_lstm_peephole_i_c_!=NULL)
		delete vec_lstm_peephole_i_c_;
	if(vec_lstm_peephole_f_c_!=NULL)
		delete vec_lstm_peephole_f_c_;
	if(vec_lstm_peephole_o_c_!=NULL)
		delete vec_lstm_peephole_o_c_;

	if(mat_lstm_w_proj_!=NULL)
		delete mat_lstm_w_proj_;
	if(mat_affine_w_!=NULL)
		delete mat_affine_w_;
	if(vec_affine_b_!=NULL)
		delete vec_affine_b_;
}

bool LstmTagParseDecode::split(const std::string& str, const std::string& delimiter, std::vector<std::string>& tokens)
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

bool LstmTagParseDecode::ReadModelBin(const std::string& model_path)
{
	std::ifstream ifs;
	ifs.open(model_path.c_str(),std::ifstream::in|std::ifstream::binary);
	if(!ifs)
	{
		std::cerr << "Fail to open " << model_path << std::endl;
		return false;
	}

	int feat = 0;
	// load lstm cell parameter
	ifs.read((char*)&feat, sizeof(int));
	cell_size_ = feat;

	ifs.read((char*)&feat, sizeof(int));
	proj_num_ = feat;
	
	ifs.read((char*)&feat, sizeof(int));
	word_dim_ = feat;
	
	ifs.read((char*)&feat, sizeof(int));
	tag_dim_ = feat;
	
	ifs.read((char*)&feat, sizeof(int));
	parse_dim_ = feat;
	
	ifs.read((char*)&feat, sizeof(int));
	target_num_ = feat;
	//std::cout << cell_size_ << "\t" << proj_num_ << "\t" << word_dim_ << "\t" << tag_dim_ << "\t" << target_num_ << std::endl; 

	std::string str;
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

	// load tag embedding
	str.clear();
	ifs >> str;
	if(str == "tag_embedding")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		tag_size_ = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;
		if(column == tag_dim_)
		{
			for(size_t i=0; i<tag_size_; i++)
			{
				float* embedding = new float[tag_dim_];
				ifs.read((char*)embedding, tag_dim_*sizeof(float));
				tag_embedding_.push_back(embedding);
			}
		}
		else
		{
			std::cerr << "Model format error, tag_embedding parameter colomn is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, tag_embedding is not expect!!!" << std::endl;
		return false;
	}

	// load parse embedding
	str.clear();
	ifs >> str;
	if(str == "parse_embedding")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		parse_size_ = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;
		if(column == word_dim_*parse_dim_)
		{
			for(size_t i=0; i<parse_size_; i++)
			{
				float* embedding = new float[column];
				ifs.read((char*)embedding, column*sizeof(float));
				parse_embedding_.push_back(embedding);
			}
		}
		else
		{
			std::cerr << "Model format error, parse_embedding parameter colomn is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, parse_embedding is not expect!!!" << std::endl;
		return false;
	}

	// load lstm weight i,j,f,o inputs
	str.clear();
	ifs >> str;
	if(str == "lstm_cell_x")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==word_dim_+tag_dim_+parse_dim_) && (column==4*cell_size_))
		{
			mat_lstm_w_ijfo_x_ = new float[row * column];
			ifs.read((char*)mat_lstm_w_ijfo_x_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, lstm_cell_x parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, lstm_cell_x is not expect!!!" << std::endl;
		return false;
	}

	// load lstm weight i,j,f,o hiddens
	str.clear();
	ifs >> str;
	if(str == "lstm_cell_h")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==proj_num_) && (column==4*cell_size_))
		{
			mat_lstm_w_ijfo_h_ = new float[row * column];
			ifs.read((char*)mat_lstm_w_ijfo_h_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, lstm_cell_h parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, lstm_cell_h is not expect!!!" << std::endl;
		return false;
	}

	// load lstm weight i,j,f,o biases
	str.clear();
	ifs >> str;
	if(str == "lstm_cell_bias")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==4*cell_size_))
		{
			vec_lstm_bias_ijfo_ = new float[row*column];
			ifs.read((char*)vec_lstm_bias_ijfo_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, lstm_cell_bias parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, lstm_cell_bias is not expect!!!" << std::endl;
		return false;
	}

	// load lstm peephole f
	str.clear();
	ifs >> str;
	if(str == "w_f_diag")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==cell_size_))
		{
			vec_lstm_peephole_f_c_ = new float[row*column];
			ifs.read((char*)vec_lstm_peephole_f_c_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, w_f_diag parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, w_f_diag is not expect!!!" << std::endl;
		return false;
	}

	// load lstm peephole i
	str.clear();
	ifs >> str;
	if(str == "w_i_diag")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==cell_size_))
		{
			vec_lstm_peephole_i_c_ = new float[row*column];
			ifs.read((char*)vec_lstm_peephole_i_c_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, w_i_diag parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, w_i_diag is not expect!!!" << std::endl;
		return false;
	}

	// load lstm peephole o
	str.clear();
	ifs >> str;
	if(str == "w_o_diag")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==1) && (column==cell_size_))
		{
			vec_lstm_peephole_o_c_ = new float[row*column];
			ifs.read((char*)vec_lstm_peephole_o_c_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, w_o_diag parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, w_o_diag is not expect!!!" << std::endl;
		return false;
	}

	// load projection weights
	str.clear();
	ifs >> str;
	if(str == "projection")
	{
		ifs.get();
		ifs.read((char*)&feat, sizeof(int));
		size_t row = feat;
		ifs.read((char*)&feat, sizeof(int));
		size_t column = feat;

		if((row==cell_size_) && (column==proj_num_))
		{
			mat_lstm_w_proj_ = new float[row*column];
			ifs.read((char*)mat_lstm_w_proj_, row*column*sizeof(float));
		}
		else
		{
			std::cerr << "Model format error, projection parameter is not expect!!!" << std::endl;
			return false;
		}
	}
	else
	{
		std::cerr << "Model format error, projection is not expect!!!" << std::endl;
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

		if((row==1) && (column==proj_num_))
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

		if((row==1) && (column==1))
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

	ifs.close();
	return true;
}

size_t LstmTagParseDecode::getWordId(std::string word)
{
	size_t wid = word_size_ - 1;
	std::map<std::string, size_t>::iterator iter = word2id_.find(word);
	if(iter != word2id_.end())
		wid = iter->second;
	return wid;
}

size_t LstmTagParseDecode::getTagIdx(std::string hexTag)
{
	size_t tid = 0;
	size_t number = 0;
	size_t data = hexTag[0] - '0';
	while(data>0)
	{
		number++;
		data /= 2;
	}
	tid = (hexTag.length()-1)*4 + number;
	//std::cout << hexTag << "\t" << data << "\t" << number << "\t" << tid << std::endl;
	if(tid>tag_size_ || tid<0)
		tid = 0;

	return tid;
}

void LstmTagParseDecode::Propagate(const float* mat_input, size_t input_row, size_t input_col, float* mat_output, size_t output_row, size_t output_col)
{
	float input_ijfo_buf[(input_row+1)*4*cell_size_];
	float project_buf[(input_row+1)*proj_num_];
	float cell_buf[(input_row+1)*cell_size_];

	memset(input_ijfo_buf, 0, (input_row+1)*4*cell_size_*sizeof(float));
	memset(project_buf, 0, (input_row+1)*proj_num_*sizeof(float));
	memset(cell_buf, 0, (input_row+1)*cell_size_*sizeof(float));

	for(size_t i=1; i<input_row+1; i++)
		memcpy(input_ijfo_buf+i*4*cell_size_, vec_lstm_bias_ijfo_, 4*cell_size_*sizeof(float));

	/*std::cout << "input_ijfo_buf after bias" << std::endl;
	for(size_t idx=0; idx<input_row+1; idx++)
	{
		for(size_t jdx=0; jdx<4*cell_size_; jdx++)
			std::cout << input_ijfo_buf[idx*4*cell_size_+jdx] << " ";
		std::cout << std::endl;
	}*/

	//calc input*_w_ijfo_x_
	//mat_input[(word+target_delay_) * (word_dim_+tag_dim_)]
	//mat_lstm_w_ijfo_x_[(word_dim_+tag_dim_) * (4*cell_size_)]
	//input_ijfo_buf[(word+target_delay_+1)*(4*cell_size_)]
	/*for(size_t i=0; i<input_row; i++)
	{
		for(size_t j=0; j<4*cell_size_; j++)
		{
			float sum = 0.0;
			for(size_t k=0; k<input_col; k++)
			{
				sum += mat_input[i*input_col+k]*mat_lstm_w_ijfo_x_[k*4*cell_size_+j];
			}
			input_ijfo_buf[(i+1)*4*cell_size_+j] += sum;
		}
	}*/
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
			input_row, 4*cell_size_, input_col,
			1.0, mat_input, input_col, mat_lstm_w_ijfo_x_, 4*cell_size_,
			1.0, input_ijfo_buf+4*cell_size_, 4*cell_size_);

	/*std::cout << "input_ijfo_buf after input" << std::endl;
	for(size_t idx=0; idx<input_row+1; idx++)
	{
		for(size_t jdx=0; jdx<4*cell_size_; jdx++)
			std::cout << input_ijfo_buf[idx*4*cell_size_+jdx] << " ";
		std::cout << std::endl;
	}*/

	for(size_t i=0; i<input_row; i++)
	{
		/*std::cout << "running step:" << i << std::endl;*/
		//calc hidden*_w_ijfo_h_
		//project_buf[(word+target_delay_+1)*proj_num_]
		//mat_lstm_w_ijfo_h_[proj_num_ * (4*cell_size_)]
		//input_ijfo_buf[(word+target_delay_+1)*(4*cell_size_)]
		/*for(size_t j=0; j<4*cell_size_; j++)
		{
			float sum = 0.0;
			for(size_t k=0; k<proj_num_; k++)
			{
				sum += project_buf[i*proj_num_+k]*mat_lstm_w_ijfo_h_[k*4*cell_size_+j];
			}
			input_ijfo_buf[(i+1)*4*cell_size_+j] += sum;
		}*/
		
		cblas_sgemv(CblasRowMajor, CblasTrans, proj_num_, 4*cell_size_,
			1.0, mat_lstm_w_ijfo_h_, 4*cell_size_, project_buf+i*proj_num_,
			1, 1.0, input_ijfo_buf+(i+1)*4*cell_size_, 1);

		/*std::cout << "input_ijfo_buf after hidden" << std::endl;
		for(size_t idx=0; idx<4*cell_size_; idx++)
			std::cout << input_ijfo_buf[(i+1)*4*cell_size_+idx] << " ";
		std::cout << std::endl;*/

		//calc peephole f
		//cell_buf[(word+target_delay_+1)*cell_size_]
		//vec_lstm_peephole_f_c_[cell_size_]
		//input_ijfo_buf[(word+target_delay_+1)*(4*cell_size_)]
		float peephole_f_c[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			peephole_f_c[j] = cell_buf[i*cell_size_+j]*vec_lstm_peephole_f_c_[j];
			input_ijfo_buf[(i+1)*4*cell_size_+2*cell_size_+j] += peephole_f_c[j];
		}

		/*std::cout << "peephole_f" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << peephole_f_c[idx] << " ";
		std::cout << std::endl;*/

		//calc peephole i
		//cell_buf[(word+target_delay_+1)*cell_size_]
		//vec_lstm_peephole_i_c_[cell_size_]
		//input_ijfo_buf[(word+target_delay_+1)*(4*cell_size_)]
		float peephole_i_c[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			peephole_i_c[j] = cell_buf[i*cell_size_+j]*vec_lstm_peephole_i_c_[j];
			input_ijfo_buf[(i+1)*4*cell_size_+j] += peephole_i_c[j];
		}

		/*std::cout << "peephole_i" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << peephole_i_c[idx] << " ";
		std::cout << std::endl;*/

		/*std::cout << "input_ijfo_buf after peephole i,f" << std::endl;
		for(size_t idx=0; idx<4*cell_size_; idx++)
			std::cout << input_ijfo_buf[(i+1)*4*cell_size_+idx] << " ";
		std::cout << std::endl;*/

		//cala sigmoid f,i  tanh j
		float sigmoid_f[cell_size_];
		float sigmoid_i[cell_size_];
		float tanh_j[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			sigmoid_f[j] = 0.5*(tanh(input_ijfo_buf[(i+1)*4*cell_size_+2*cell_size_+j]*0.5) + 1);
			sigmoid_i[j] = 0.5*(tanh(input_ijfo_buf[(i+1)*4*cell_size_+j]*0.5) + 1);
			tanh_j[j] = tanh(input_ijfo_buf[(i+1)*4*cell_size_+cell_size_+j]);
		}
		/*std::cout << "sigmoid_f" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << sigmoid_f[idx] << " ";
		std::cout << std::endl;

		std::cout << "sigmoid_i" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << sigmoid_i[idx] << " ";
		std::cout << std::endl;

		std::cout << "tanh_j" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << tanh_j[idx] << " ";
		std::cout << std::endl;*/

		//calc cell_buf
		//cell_buf[i] = sigmoid_f * cell_buf[i-1] + sigmoid_i * tanh_j
		for(size_t j=0; j<cell_size_; j++)
		{
			cell_buf[(i+1)*cell_size_+j] = sigmoid_f[j]*cell_buf[i*cell_size_+j] + sigmoid_i[j]*tanh_j[j];
		}

		/*std::cout << "cell_buf" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << cell_buf[(i+1)*cell_size_+idx] << " ";
		std::cout << std::endl;*/

		//cala tanh h
		float tanh_h[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			tanh_h[j] = tanh(cell_buf[(i+1)*cell_size_+j]);
		}

		/*std::cout << "tanh_h" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << tanh_h[idx] << " ";
		std::cout << std::endl;*/

		//calc peephole o
		//cell_buf[(word+target_delay_+1)*cell_size_]
		//vec_lstm_peephole_o_c_[cell_size_]
		//input_ijfo_buf[(word+target_delay_+1)*(4*cell_size_)]
		float peephole_o_c[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			peephole_o_c[j] = cell_buf[(i+1)*cell_size_+j]*vec_lstm_peephole_o_c_[j];
			input_ijfo_buf[(i+1)*4*cell_size_+3*cell_size_+j] += peephole_o_c[j];
		}

		/*std::cout << "peehole c" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << peephole_o_c[idx] << " ";
		std::cout << std::endl;

		std::cout << "input_ijfo_buf after peephole o" << std::endl;
		for(size_t idx=0; idx<4*cell_size_; idx++)
			std::cout << input_ijfo_buf[(i+1)*4*cell_size_+idx] << " ";
		std::cout << std::endl;*/

		//sigmoid o
		float sigmoid_o[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			sigmoid_o[j] = 0.5*(tanh(input_ijfo_buf[(i+1)*4*cell_size_+3*cell_size_+j]*0.5) + 1);
		}

		/*std::cout << "sigmoid_o" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << sigmoid_o[idx] << " ";
		std::cout << std::endl;*/

		//cala cell_h
		//cell_h = sigmoid_o*tanh_h
		float cell_h[cell_size_];
		for(size_t j=0; j<cell_size_; j++)
		{
			cell_h[j] = sigmoid_o[j] * tanh_h[j];
		}

		/*std::cout << "cell_h" << std::endl;
		for(size_t idx=0; idx<cell_size_; idx++)
			std::cout << cell_h[idx] << " ";
		std::cout << std::endl;*/

		//cala project_buf
		//mat_lstm_w_proj_[cell_size_*proj_num_]
		//cell_h[cell_size_]
		//project_buf[(word+target_delay_+1)*proj_num_]
		/*for(size_t j=0; j<proj_num_; j++)
		{
			float sum = 0.0;
			for(size_t k=0; k<cell_size_; k++)
			{
				sum += cell_h[k]*mat_lstm_w_proj_[k*proj_num_+j];
			}
			project_buf[(i+1)*proj_num_+j] = sum;
		}*/
		
		cblas_sgemv(CblasRowMajor, CblasTrans, cell_size_, proj_num_,
			1.0, mat_lstm_w_proj_, proj_num_, cell_h,
			1, 0.0, project_buf+(i+1)*proj_num_, 1);

		/*std::cout << "project_buf" << std::endl;
		for(size_t idx=0; idx<proj_num_; idx++)
			std::cout << project_buf[(i+1)*proj_num_+idx] << " ";
		std::cout << std::endl;*/

		//cala affine_h
		//project_buf[(word+target_delay_+1)*proj_num_]
		//mat_affine_w_[target_num_*proj_num_]
		//affine_h[target_num_]
		float affine_h[target_num_];
		memcpy(affine_h, vec_affine_b_, target_num_*sizeof(float));

		for(size_t j=0; j<target_num_; j++)
		{
			float sum = 0.0;
			for(size_t k=0; k<proj_num_; k++)
			{
				sum += project_buf[(i+1)*proj_num_+k]*mat_affine_w_[j*proj_num_+k];
			}
			affine_h[j] += sum;
		}

		/*cblas_sgemv(CblasRowMajor, CblasNoTrans, target_num_, proj_num_,
			1.0, mat_affine_w_, proj_num_, project_buf+(i+1)*proj_num_,
			1, 1.0, affine_h, 1);*/

		//cala mat_output
		for(size_t j=0; j<target_num_; j++)
		{
			mat_output[i*target_num_+j] = 0.5*(tanh(affine_h[j]*0.5) + 1);
		}

		/*std::cout << "affine_h, mat_output" << std::endl;
		for(size_t idx=0; idx<target_num_; idx++)
			std::cout << affine_h[idx] << " " << mat_output[i*target_num_+idx] << " ";
		std::cout << std::endl;*/

	}
}

void LstmTagParseDecode::getImportant(std::vector<std::string>& words, std::vector<size_t>& tags, std::vector<size_t>& heads, std::vector<size_t>& deprels, std::vector<float>& scores)
{
	size_t input_row = words.size() + target_delay_;
	size_t input_col = word_dim_ + tag_dim_ + parse_dim_;
	float* mat_input = new float[input_row * input_col];
	
	size_t output_row = words.size() + target_delay_;
	size_t output_col = target_num_;
	float* mat_output = new float[output_row * output_col];

	for(size_t i=0; i<words.size(); i++)
	{
		float* embedding = word_embedding_[getWordId(words[i])];
		for(size_t j=0; j<word_dim_; j++)
			mat_input[i*input_col + j] = embedding[j];
		float* t_embedding = tag_embedding_[tags[i]];
		for(size_t j=0; j<tag_dim_; j++)
			mat_input[i*input_col + word_dim_ + j] = t_embedding[j];
		
		float* p_embedding = parse_embedding_[deprels[i]];
		float* h_embedding = word_embedding_[getWordId(words[heads[i]])];
		if (deprels[i]==1 && heads[i]==0)
			h_embedding = word_embedding_[word_size_ - 2];
		for(size_t j=0; j<parse_dim_; j++)
		{
			float score = 0.0;
			for(size_t k=0; k<word_dim_; k++)
				score += h_embedding[k] * p_embedding[k*parse_dim_ + j];
			mat_input[i*input_col + word_dim_ + tag_dim_ + j] = score;
		}
	}
	for(size_t i=0; i<target_delay_; i++)
	{
		float* embedding = word_embedding_[0];
		for(size_t j=0; j<word_dim_; j++)
			mat_input[(i+words.size())*input_col + j] = embedding[j];
		float* t_embedding = tag_embedding_[0];
		for(size_t j=0; j<tag_dim_; j++)
			mat_input[(i+words.size())*input_col + word_dim_ + j] = t_embedding[j];

		float* p_embedding = parse_embedding_[0];
		float* h_embedding = word_embedding_[0];
		for(size_t j=0; j<parse_dim_; j++)
		{
			float score = 0.0;
			for(size_t k=0; k<word_dim_; k++)
				score += h_embedding[k] * p_embedding[k*parse_dim_ + j];
			mat_input[(i+words.size())*input_col + word_dim_ + tag_dim_ + j] = score;
		}
	}

	/*std::cout << "mat_input" << std::endl;
	for(size_t i=0; i<input_row; i++)
	{
		for(size_t j=0; j<input_col; j++)
			std::cout << mat_input[i*input_col +j] << " ";
		std::cout << std::endl;
	}*/

	struct timeval tv1, tv2;
	gettimeofday(&tv1, NULL);
	Propagate(mat_input, input_row, input_col, mat_output, output_row, output_col);
	gettimeofday(&tv2, NULL);
	size_t span = (tv2.tv_sec-tv1.tv_sec)*1000000+(tv2.tv_usec-tv1.tv_usec);
	std::cout << "CPU time used: " << span << " us" << std::endl;

	/*std::cout << "mat_output:";
	for(size_t i=0; i<output_row*output_col; i++)
		std::cout << " " << mat_output[i];
	std::cout << std::endl;*/

	for(size_t i=target_delay_; i<words.size() + target_delay_; i++)
		scores.push_back(mat_output[i]);

	delete[] mat_input;
	delete[] mat_output;
}

void LstmTagParseDecode::printLstmPareMeter()
{
	std::cout << "word_dim_:" << word_dim_ << std::endl;
	std::cout << "word_size_:" << word_size_ << std::endl;
	std::cout << "tag_dim_:" << tag_dim_ << std::endl;
	std::cout << "tag_size_:" << tag_size_ << std::endl;
	std::cout << "parse_dim_:" << parse_dim_ << std::endl;
	std::cout << "parse_size_:" << parse_size_ << std::endl;
	std::cout << "cell_size_:" << cell_size_ << std::endl;
	std::cout << "proj_num_:" << proj_num_ << std::endl;
	std::cout << "target_num_:" << target_num_ << std::endl;

	std::cout << "word_embedding_" << std::endl;
	for(size_t i=0; i<word_size_; i++)
	{
		std::cout << i << "\t" << id2word_[i] << "\t" << word2id_[id2word_[i]] << "\t";
		for(size_t j=0; j<word_dim_; j++)
			std::cout << word_embedding_[i][j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "tag_embedding_" << std::endl;
	for(size_t i=0; i<tag_size_; i++)
	{
		for(size_t j=0; j<tag_dim_; j++)
			std::cout << tag_embedding_[i][j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "parse_embedding_" << std::endl;
	for(size_t i=0; i<parse_size_; i++)
	{
		for(size_t j=0; j<parse_dim_*word_dim_; j++)
			std::cout << parse_embedding_[i][j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "mat_lstm_w_ijfo_x_" << std::endl;
	for(size_t i=0; i<word_dim_+tag_dim_+parse_dim_; i++)
	{
		for(size_t j=0; j<4*cell_size_; j++)
			std::cout << mat_lstm_w_ijfo_x_[i*4*cell_size_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "mat_lstm_w_ijfo_h_" << std::endl;
	for(size_t i=0; i<proj_num_; i++)
	{
		for(size_t j=0; j<4*cell_size_; j++)
			std::cout << mat_lstm_w_ijfo_h_[i*4*cell_size_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "vec_lstm_bias_ijfo_" << std::endl;
	for(size_t i=0; i<4*cell_size_; i++)
		std::cout << vec_lstm_bias_ijfo_[i] << " ";
	std::cout << std::endl;	

	std::cout << "vec_lstm_peephole_f_c_" << std::endl;
	for(size_t i=0; i<cell_size_; i++)
		std::cout << vec_lstm_peephole_f_c_[i] << " ";
	std::cout << std::endl;	

	std::cout << "vec_lstm_peephole_i_c_" << std::endl;
	for(size_t i=0; i<cell_size_; i++)
		std::cout << vec_lstm_peephole_i_c_[i] << " ";
	std::cout << std::endl;	

	std::cout << "vec_lstm_peephole_o_c_" << std::endl;
	for(size_t i=0; i<cell_size_; i++)
		std::cout << vec_lstm_peephole_o_c_[i] << " ";
	std::cout << std::endl;	

	std::cout << "mat_lstm_w_proj_" << std::endl;
	for(size_t i=0; i<cell_size_; i++)
	{
		for(size_t j=0; j<proj_num_; j++)
			std::cout << mat_lstm_w_proj_[i*proj_num_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "mat_affine_w_" << std::endl;
	for(size_t i=0; i<target_num_; i++)
	{
		for(size_t j=0; j<proj_num_; j++)
			std::cout << mat_affine_w_[i*proj_num_+j] << " ";
		std::cout << std::endl;	
	}

	std::cout << "vec_affine_b_" << std::endl;
	for(size_t i=0; i<target_num_; i++)
		std::cout << vec_affine_b_[i] << " ";
	std::cout << std::endl;	
}

int main(void)
{
	LstmTagParseDecode model;
	if(!model.ReadModelBin("/data/shixiang/parse_imp/model.tag.parse.lstm.bin"))
	{
		std::cout << "read model error" << std::endl;
		return 0;
	}
	//model.printLstmPareMeter();
	
	std::string line;
	while(std::getline(std::cin, line))
	{
		std::vector<std::string> tokens;
		std::vector<std::string> words;
		std::vector<std::string> tags;
		std::vector<std::string> headts;
		std::vector<std::string> deprelts;
		model.split(line, "\t", tokens);
		if(tokens.size()!=5)
		{
			std::cerr << "input format error, must 5 tokens!!" << std::endl;
			continue;
		}

		model.split(tokens[0], " ", words);
		model.split(tokens[2], " ", tags);
		model.split(tokens[3], " ", headts);
		model.split(tokens[4], " ", deprelts);
		if((words.size() != tags.size()) && (words.size() != headts.size()) && (words.size() != deprelts.size()))
		{
			std::cerr << "word size is not equal tag size!!" << std::endl;
			continue;
		}

		std::vector<size_t> vecTid;
		for(size_t i=0; i<tags.size(); i++)
			vecTid.push_back(model.getTagIdx(tags[i]));

		std::vector<size_t> heads;
		for(size_t i=0; i<headts.size(); i++)
			heads.push_back((size_t)atoi(headts[i].c_str()));

		std::vector<size_t> deprels;
		for(size_t i=0; i<deprelts.size(); i++)
			deprels.push_back((size_t)atoi(deprelts[i].c_str()));

		std::vector<float> scores;
		model.getImportant(words, vecTid, heads, deprels, scores);

		std::cout << line << "\n";
		for(size_t i=0;i<vecTid.size();i++)
			std::cout << " " << vecTid[i];
		std::cout << "\n";
		for(size_t i=0;i<heads.size();i++)
			std::cout << " " << heads[i];
		std::cout << "\n";
		for(size_t i=0;i<deprels.size();i++)
			std::cout << " " << deprels[i];
		std::cout << "\n";
		for(size_t i=0;i<scores.size();i++)
			std::cout << " " << scores[i];
		std::cout << std::endl;
		std::cout << std::endl;
	}

	std::cout << "pass~~~" << std::endl;
	return 0;
}
