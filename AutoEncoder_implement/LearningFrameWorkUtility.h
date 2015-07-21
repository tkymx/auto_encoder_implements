#include <iostream>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <map>

#include "utility.h"
#include "network.h"
#include "data.h"

#ifndef LEARNING_FRAME_WORK_UTILITY
#define LEARNING_FRAME_WORK_UTILITY

class LoadBase
{
protected:
	data_manager input_data;
	data_manager input_data_test;
	network_parameter3 *netp1;
	network_parameter3 *netp2;
	network_parameter5 *netpa;
	std::map<std::string, std::string> learn_map;
public:
	LoadBase(std::map<std::string, std::string> lm)
		: netp1(NULL)
		, netp2(NULL)
		, netpa(NULL)
		, learn_map(lm)
	{
	}
	virtual ~LoadBase()
	{
		//delete
		if (input_data.is_open())input_data.release();
		if (input_data_test.is_open())input_data_test.release();

		//network
		if (netp1 != 0)delete netp1;
		if (netp2 != 0)delete netp2;
		if (netpa != 0)delete netpa;
	}
public:
	virtual void load_data(std::string input_dir) = 0;
	virtual void create_netp1(int hidden, float lr, float lm, float mo , learn_info *info)
	{
		netp1 = new network_parameter3(
			input_data.get_input_node(),
			hidden,
			input_data.get_input_node(),
			lr, lm, mo);

		netp1->set_network_tool(
			info->get_network_tool());

		netp1->set_network_active_function( 
			get_cost_function(learn_map),
			get_active_function(learn_map, 0),
			get_active_function(learn_map, 3) );

		netp1->print_network_info();
	}
	virtual void create_netp2(int hidden1 , int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		netp2 = new network_parameter3(
			hidden1,
			hidden2,
			hidden1,
			lr,lm,mo);

		netp2->set_network_tool(
			info->get_network_tool());

		netp2->set_network_active_function(
			get_cost_function(learn_map),
			get_active_function(learn_map, 1),
			get_active_function(learn_map, 2));

		netp2->print_network_info();
	}
	virtual void create_netpa(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		//���ׂĂ̑w�̊w�K
		netpa = new network_parameter5(
			input_data.get_input_node(),
			hidden1,
			hidden2,
			hidden1,
			input_data.get_input_node(),
			lr,lm,mo);

		netpa->set_network_tool(
			info->get_network_tool());

		netpa->set_network_active_function(
			get_cost_function(learn_map),
			get_active_function(learn_map, 0),
			get_active_function(learn_map, 1),
			get_active_function(learn_map, 2),
			get_active_function(learn_map, 3));

		netpa->print_network_info();

	}
	virtual void create_netpa(float lr, float lm, float mo, learn_info *info)
	{
		if (netp1 == NULL)
		{
			std::cout << "netpa netp1 ���ݒu����Ă��܂���" << std::endl;
		}
		if (netp2 == NULL)
		{
			std::cout << "netpa netp2 ���ݒu����Ă��܂���" << std::endl;
		}

		//���ׂĂ̑w�̊w�K
		netpa = new network_parameter5(
			netp1, 
			netp2, 
			lr,lm,mo);

		netpa->set_network_tool(
			info->get_network_tool());

		netpa->set_network_active_function(
			get_cost_function(learn_map),
			get_active_function(learn_map, 0),
			get_active_function(learn_map, 1),
			get_active_function(learn_map, 2),
			get_active_function(learn_map, 3));

		netpa->print_network_info();

	}
public:
	data_manager* get_input_data()
	{
		return &input_data;
	}
	data_manager* get_input_data_test()
	{
		return &input_data_test;
	}
	network_parameter* get_netp1()
	{
		return netp1;
	}
	network_parameter* get_netp2()
	{
		return netp2;
	}
	network_parameter* get_netpa()
	{
		return netpa;
	}
};


class LoadProc : public LoadBase
{
protected:
	procrustes_parameter *train_proc;
	procrustes_parameter *test_proc;

public:
	LoadProc(std::map<std::string, std::string> lm)
		: LoadBase(lm)
		, train_proc(NULL)
		, test_proc(NULL)
	{

	}
	virtual ~LoadProc()
	{
		//�e�X�^�[	
		if (train_proc != 0)delete train_proc;
		if (test_proc != 0)delete test_proc;
	}
public:
	virtual void load_data(std::string input_dir)
	{
		load_proc(input_dir);
	}
	virtual void load_proc(std::string input_dir) = 0;
	virtual void create_netp1(int hidden, float lr, float lm, float mo, learn_info *info)
	{
		if (train_proc == NULL)
		{
			std::cout << "netp1 train_proc ���ݒu����Ă��܂���" << std::endl;
		}
		if (test_proc == NULL)
		{
			std::cout << "netp1 test_proc ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netp1(hidden, lr, lm, mo, info);

		netp1->set_tester(
			new tester_procrutes(train_proc, input_data.get_normal_param()),
			new tester_procrutes(test_proc, input_data.get_normal_param()));

	}
	virtual void create_netp2(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		if (train_proc == NULL)
		{
			std::cout << "netp2 train_proc ���ݒu����Ă��܂���" << std::endl;
		}
		if (test_proc == NULL)
		{
			std::cout << "netp2 test_proc ���ݒu����Ă��܂���" << std::endl;
		}
		if (netp1== NULL)
		{
			std::cout << "netp2 netp1 ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netp2(hidden1, hidden2, lr, lm, mo, info);

		netp2->set_tester(
			new tester_decode_procrutes(netp1, train_proc, input_data.get_normal_param()),
			new tester_decode_procrutes(netp1, test_proc, input_data.get_normal_param()));
	}
	virtual void create_netpa(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		if (train_proc == NULL)
		{
			std::cout << "netpa train_proc ���ݒu����Ă��܂���" << std::endl;
		}
		if (test_proc == NULL)
		{
			std::cout << "netpa test_proc ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netpa(hidden1, hidden2, lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes(train_proc, input_data.get_normal_param()),
			new tester_procrutes(test_proc, input_data.get_normal_param()));
	}
	virtual void create_netpa(float lr, float lm, float mo, learn_info *info)
	{
		if (train_proc == NULL)
		{
			std::cout << "netpa train_proc ���ݒu����Ă��܂���" << std::endl;
		}
		if (test_proc == NULL)
		{
			std::cout << "netpa test_proc ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netpa(lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes(train_proc, input_data.get_normal_param()),
			new tester_procrutes(test_proc, input_data.get_normal_param()));
	}
};

class LoadShape : public LoadProc
{
public:
	LoadShape(std::map<std::string, std::string> lm)
		: LoadProc(lm)
	{

	}
	virtual ~LoadShape()
	{

	}
public:
	virtual void load_data(std::string input_dir)
	{
		input_data = load_data_shape(input_dir + "//train.dat");
		if (!input_data.is_open())
		{
			std::cout << "train.dat������܂���" << std::endl;
			exit(1);
		}

		input_data_test = load_data_shape(input_dir + "//test.dat");
		if (!input_data_test.is_open())
		{
			std::cout << "test.dat������܂���" << std::endl;
			exit(1);
		}

		LoadProc::load_data(input_dir);
	}
	virtual void load_proc(std::string input_dir)
	{
		//�g���[�j���O���
		train_proc = new procrustes_parameter_shape();
		train_proc->load_data(input_dir + "//train.dat.train.dat");

		test_proc = new procrustes_parameter_shape();
		test_proc->load_data(input_dir + "//test.dat.train.dat");
	}


};

class LoadTexture : public LoadProc
{
public:
	LoadTexture(std::map<std::string, std::string> lm)
		: LoadProc(lm)
	{

	}
	virtual ~LoadTexture()
	{

	}
public:
	virtual void load_data(std::string input_dir)
	{
		input_data = load_data_texture(input_dir + "//train.dat");
		if (!input_data.is_open())
		{
			std::cout << "train.dat������܂���" << std::endl;
			exit(1);
		}

		input_data_test = load_data_texture(input_dir + "//test.dat");
		if (!input_data_test.is_open())
		{
			std::cout << "test.dat������܂���" << std::endl;
			exit(1);
		}

		LoadProc::load_data(input_dir);
	}
	virtual void load_proc(std::string input_dir)
	{
		//�g���[�j���O���
		train_proc = new procrustes_parameter_texture();
		train_proc->load_data(input_dir + "//train.dat.train.dat");

		test_proc = new procrustes_parameter_texture();
		test_proc->load_data(input_dir + "//test.dat.train.dat");
	}

};

class LoadAAM : public LoadBase
{
protected:
	data_manager data_shape;
	data_manager data_texture;

	network_parameter* shape_model;
	network_parameter* texture_model;

	procrustes_parameter_shape *shape_proc_train;
	procrustes_parameter_texture *texture_proc_train;
	procrustes_parameter_shape *shape_proc_test;
	procrustes_parameter_texture *texture_proc_test;

public:
	LoadAAM(std::map<std::string, std::string> lm)
		: LoadBase(lm)
		, shape_model(NULL)
		, texture_model(NULL)
		, shape_proc_train(NULL)
		, texture_proc_train(NULL)
		, shape_proc_test(NULL)
		, texture_proc_test(NULL)
	{

	}
	virtual ~LoadAAM()
	{
		if(shape_model!=NULL)delete shape_model;
		if (texture_model != NULL)delete texture_model;
		if (shape_proc_train != NULL)delete shape_proc_train;
		if (texture_proc_train != NULL)delete texture_proc_train;
		if (shape_proc_test != NULL)delete shape_proc_test;
		if (texture_proc_test != NULL)delete texture_proc_test;

		if (data_shape.is_open())data_shape.release();
		if (data_texture.is_open())data_texture.release();
	}
public:
	void param_chaeck()
	{
		if (shape_model == NULL)
		{
			std::cout << "netp shape_model ���ݒu����Ă��܂���" << std::endl;
		}
		if (texture_model == NULL)
		{
			std::cout << "netp texture_model ���ݒu����Ă��܂���" << std::endl;
		}
		if (shape_proc_train == NULL)
		{
			std::cout << "netp shape_proc_train ���ݒu����Ă��܂���" << std::endl;
		}
		if (texture_proc_train == NULL)
		{
			std::cout << "netp texture_proc_train ���ݒu����Ă��܂���" << std::endl;
		}
		if (shape_proc_test == NULL)
		{
			std::cout << "netp shape_proc_test ���ݒu����Ă��܂���" << std::endl;
		}
		if (texture_proc_train == NULL)
		{
			std::cout << "netp shape_proc_train ���ݒu����Ă��܂���" << std::endl;
		}

	}

public:
	virtual void load_data(std::string input_dir)
	{
		int shape_count = 0;

		input_data = load_data_aam(input_dir + "//train", shape_count);
		if (!input_data.is_open())
		{
			std::cout << "train.dat������܂���" << std::endl;
			exit(1);
		}

		input_data_test = load_data_aam(input_dir + "//test", shape_count);
		if (!input_data_test.is_open())
		{
			std::cout << "test.dat������܂���" << std::endl;
			exit(1);
		}

		data_shape = load_data_shape(input_dir + "//train.dat");
		data_texture = load_data_texture(input_dir + "//train.dat");
		data_shape.normalize(1);
		data_texture.normalize(1);

		shape_proc_train = new procrustes_parameter_shape(input_dir + "//train.dat.train.dat");
		texture_proc_train = new procrustes_parameter_texture(input_dir + "//train.dat.train.dat");

		shape_proc_test = new procrustes_parameter_shape(input_dir + "//test.dat.train.dat");
		texture_proc_test = new procrustes_parameter_texture(input_dir + "//test.dat.train.dat");

		shape_model = get_network_parameter("shape_model", learn_map, shape_proc_train->get_origin_shape().get_input_node());
		texture_model = get_network_parameter("texture_model", learn_map, texture_proc_train->get_origin_texture().get_input_node());
	}
	virtual void create_netp1(int hidden, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		LoadBase::create_netp1(hidden, lr, lm, mo, info);

		netp1->set_tester(
			new tester_procrutes_aam(
			shape_proc_train,
			texture_proc_train,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			),
			new tester_procrutes_aam(
			shape_proc_test,
			texture_proc_test,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			));
	}
	virtual void create_netp2(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		if (netp1 == NULL)
		{
			std::cout << "netp2 netp1 ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netp2(hidden1, hidden2, lr, lm, mo, info);

		netp2->set_tester(
			new tester_procrutes_aam_decode(
			shape_proc_train,
			texture_proc_train,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param(),
			netp1
			),
			new tester_procrutes_aam_decode(
			shape_proc_test,
			texture_proc_test,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param(),
			netp1
			));
	}
	virtual void create_netpa(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		LoadBase::create_netpa(hidden1, hidden2, lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes_aam(
			shape_proc_train,
			texture_proc_train,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			),
			new tester_procrutes_aam(
			shape_proc_test,
			texture_proc_test,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			));
	}
	virtual void create_netpa(float lr, float lm, float mo, learn_info *info)
	{
		LoadBase::create_netpa(lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes_aam(
			shape_proc_train,
			texture_proc_train,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			),
			new tester_procrutes_aam(
			shape_proc_test,
			texture_proc_test,
			shape_model,
			texture_model,
			data_shape.get_normal_param(),
			data_texture.get_normal_param(),
			input_data.get_normal_param()
			));
	}
};


class LoadShapeTexture : public LoadBase
{
protected:
	procrustes_parameter_shape *shape_proc_train;
	procrustes_parameter_texture *texture_proc_train;
	procrustes_parameter_shape *shape_proc_test;
	procrustes_parameter_texture *texture_proc_test;
	int m_sep_count;

public:
	LoadShapeTexture(std::map<std::string, std::string> lm)
		: LoadBase(lm)
		, shape_proc_train(NULL)
		, texture_proc_train(NULL)
		, shape_proc_test(NULL)
		, texture_proc_test(NULL)
	{

	}
	virtual ~LoadShapeTexture()
	{
		if (shape_proc_train != NULL)delete shape_proc_train;
		if (texture_proc_train != NULL)delete texture_proc_train;
		if (shape_proc_test != NULL)delete shape_proc_test;
		if (texture_proc_test != NULL)delete texture_proc_test;
	}
public:
	void param_chaeck()
	{
		if (shape_proc_train == NULL)
		{
			std::cout << "netp shape_proc_train ���ݒu����Ă��܂���" << std::endl;
		}
		if (texture_proc_train == NULL)
		{
			std::cout << "netp texture_proc_train ���ݒu����Ă��܂���" << std::endl;
		}
		if (shape_proc_test == NULL)
		{
			std::cout << "netp shape_proc_test ���ݒu����Ă��܂���" << std::endl;
		}
		if (texture_proc_train == NULL)
		{
			std::cout << "netp shape_proc_train ���ݒu����Ă��܂���" << std::endl;
		}

	}

public:
	virtual void load_data(std::string input_dir)
	{
		input_data = load_data_shape_texture(input_dir + "//train.dat" , m_sep_count);
		if (!input_data.is_open())
		{
			std::cout << "train.dat������܂���" << std::endl;
			exit(1);
		}

		input_data_test = load_data_shape_texture(input_dir + "//test.dat" , m_sep_count);
		if (!input_data_test.is_open())
		{
			std::cout << "test.dat������܂���" << std::endl;
			exit(1);
		}

		shape_proc_train = new procrustes_parameter_shape(input_dir + "//train.dat.train.dat");
		texture_proc_train = new procrustes_parameter_texture(input_dir + "//train.dat.train.dat");

		shape_proc_test = new procrustes_parameter_shape(input_dir + "//test.dat.train.dat");
		texture_proc_test = new procrustes_parameter_texture(input_dir + "//test.dat.train.dat");
	}
	virtual void create_netp1(int hidden, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		LoadBase::create_netp1(hidden, lr, lm, mo, info);

		netp1->set_tester(
			new tester_procrutes_sep(
			shape_proc_train,
			texture_proc_train,
			m_sep_count,
			input_data.get_normal_param()
			),
			new tester_procrutes_sep(
			shape_proc_test,
			texture_proc_test,
			m_sep_count,
			input_data.get_normal_param()
			));
	}
	virtual void create_netp2(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		if (netp1 == NULL)
		{
			std::cout << "netp2 netp1 ���ݒu����Ă��܂���" << std::endl;
		}

		LoadBase::create_netp2(hidden1, hidden2, lr, lm, mo, info);

		netp2->set_tester(
			new tester_procrutes_sep_decode(
			shape_proc_train,
			texture_proc_train,
			m_sep_count,
			input_data.get_normal_param(),
			netp1
			),
			new tester_procrutes_sep_decode(
			shape_proc_test,
			texture_proc_test,
			m_sep_count,
			input_data.get_normal_param(),
			netp1
			));
	}
	virtual void create_netpa(int hidden1, int hidden2, float lr, float lm, float mo, learn_info *info)
	{
		param_chaeck();

		LoadBase::create_netpa(hidden1, hidden2, lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes_sep(
			shape_proc_train,
			texture_proc_train,
			m_sep_count,
			input_data.get_normal_param()
			),
			new tester_procrutes_sep(
			shape_proc_test,
			texture_proc_test,
			m_sep_count,
			input_data.get_normal_param()
			));
	}
	virtual void create_netpa(float lr, float lm, float mo, learn_info *info)
	{
		LoadBase::create_netpa(lr, lm, mo, info);

		netpa->set_tester(
			new tester_procrutes_sep(
			shape_proc_train,
			texture_proc_train,
			m_sep_count,
			input_data.get_normal_param()
			),
			new tester_procrutes_sep(
			shape_proc_test,
			texture_proc_test,
			m_sep_count,
			input_data.get_normal_param()
			));
	}
};

#endif



