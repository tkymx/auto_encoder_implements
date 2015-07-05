#include "LearningFrameWorkUtility.h"

#ifndef LEARNING_FRAME_WORK
#define LEARNING_FRAME_WORK

class LearningFrameWork
{
private:
	float learning_rate;
	float lambda;
	float momentum;
	int epoch;

	std::string input_dir;
	
	std::string weight_dir_name;
	std::string weight_dir_name23;
	std::string weight_dir_nameall;
	
	data_manager input_data;
	data_manager input_data_test;
	
	procrustes_parameter *train_proc;
	procrustes_parameter *test_proc;

	pca_parameter *trained_pca_parameter;

	network_parameter3 *netp1;
	network_parameter3 *netp2;
	network_parameter5 *netpa;

	std::map<std::string, std::string> learn_map;

	learn_info *info;

	int layer_12_hidden_node;
	int layer_23_hidden_node;

	std::string data_structure;

	clock_t start_time;

	//aam

	data_manager data_shape;
	data_manager data_texture;

	network_parameter* shape_model;
	network_parameter* texture_model;

	procrustes_parameter_shape *shape_proc_train;
	procrustes_parameter_texture *texture_proc_train;
	procrustes_parameter_shape *shape_proc_test;
	procrustes_parameter_texture *texture_proc_test;

public:
	LearningFrameWork(int argc , char* argv[])
		: netp1(0)
		, netp2(0)
		, netpa(0)
	{
		if( argc < 15 )
		{
			std::cout << "AutoEncoder�̊w�K���ł��܂�" << std::endl;
			std::cout << "�m�����P�n�w�K��" << std::endl;
			std::cout << "�m�����Q�n�����W��" << std::endl;
			std::cout << "�m�����R�n���[�����^��" << std::endl;
			std::cout << "�m�����S�n�w�K��" << std::endl;
			std::cout << "�m�����T�n���B��w�̃m�[�h��" << std::endl;
			std::cout << "�m�����U�n���B��w�̃m�[�h��" << std::endl;
			std::cout << " [�����V�n�w�K�t�@�C���̃t�H���_" << std::endl;
			std::cout << " [�����W�n�w�K�d�݂̕ۑ��ǂݍ��ݏꏊ(12�w)" << std::endl;
			std::cout << " [�����X�n�w�K�d�݂̕ۑ��ǂݍ��ݏꏊ(23�w)" << std::endl;
			std::cout << "�m�����P�O�n�w�K�d�݂̕ۑ��ǂݍ��ݏꏊ(�S�w)" << std::endl;
			std::cout << "�m�����P�P�noption �w�K�̕��@" << std::endl;
			std::cout << "�m�����P�Q�noption �w�K�̌W��" << std::endl;
			std::cout << " [�����P�R�n�w�K������ shape texture aam" << std::endl;
			std::cout << " [�����P�S�n�w�K�ݒ�t�@�C��" << std::endl;
			exit(1);
		}

		learning_rate = static_cast<float>(atof(argv[1]));
		lambda = static_cast<float>(atof( argv[2] ));
		momentum = static_cast<float>(atof(argv[3]));
		epoch = atoi(argv[4]);
		layer_12_hidden_node = atoi(argv[5]);
		layer_23_hidden_node = atoi(argv[6]);
		input_dir = argv[7];
		weight_dir_name = argv[8];
		weight_dir_name23 = argv[9];
		weight_dir_nameall = argv[10];

		info = new learn_info(argv[11], static_cast<float>(atof(argv[12])) );

		data_structure = argv[13];

		learn_map = get_learn_dat( argv[14] );

		std::cout << "AutoEncoder�̊w�K�J�n" << std::endl;
		std::cout << "�w�K���F" << learning_rate << std::endl;
		std::cout << "�������W���F" << lambda << std::endl;
		std::cout << "���[�����^���F" << momentum << std::endl;
		std::cout << "�w�K�񐔁F" << epoch << std::endl; 
		std::cout << "�w�K�t�H���_: " << input_dir << std::endl;
		std::cout << "12�B��"  << layer_12_hidden_node << std::endl;
		std::cout << "23�B��"  << layer_23_hidden_node << std::endl;
		std::cout << "�w�K���@" << argv[11] << " " << argv[12] << std::endl;
		std::cout << "�w�K�f�[�^�\��" << data_structure << std::endl;
		std::cout << "�w�K�ݒ�t�@�C��" << argv[14] << std::endl;

		start_time = clock();

		//declaration

		if( data_structure == "shape" )
		{
			input_data = load_data_shape( input_dir + "//train.dat" );
			if( !input_data.is_open() )
			{
				std::cout << "train.dat������܂���" << std::endl;
				exit(1);
			}

			input_data_test = load_data_shape( input_dir + "//test.dat" );
			if( !input_data_test.is_open() )
			{
				std::cout << "test.dat������܂���" << std::endl;
				exit(1);
			}

			//�g���[�j���O���
			train_proc = new procrustes_parameter_shape();
			train_proc->load_data( input_dir + "//train.dat.train.dat");

			test_proc = new procrustes_parameter_shape();
			test_proc->load_data( input_dir + "//test.dat.train.dat");
		}
		else if( data_structure == "texture" )
		{
			input_data = load_data_texture( input_dir + "//train.dat" );
			if( !input_data.is_open() )
			{
				std::cout << "train.dat������܂���" << std::endl;
				exit(1);
			}

			input_data_test = load_data_texture( input_dir + "//test.dat" );
			if( !input_data_test.is_open() )
			{
				std::cout << "test.dat������܂���" << std::endl;
				exit(1);
			}

			//�g���[�j���O���
			train_proc = new procrustes_parameter_texture();
			train_proc->load_data( input_dir + "//train.dat.train.dat");

			test_proc = new procrustes_parameter_texture();
			test_proc->load_data( input_dir + "//test.dat.train.dat");
		}
		else if( data_structure == "aam" )
		{
			int shape_count = 0;

			input_data = load_data_aam( input_dir + "//train" , shape_count );
			if( !input_data.is_open() )
			{
				std::cout << "train.dat������܂���" << std::endl;
				exit(1);
			}

			input_data_test = load_data_aam( input_dir + "//test" , shape_count );
			if( !input_data_test.is_open() )
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

			shape_model = get_network_parameter("shape_model", learn_map, shape_proc_train->get_origin_shape().get_input_node() );
			texture_model = get_network_parameter("texture_model", learn_map, texture_proc_train->get_origin_texture().get_input_node() );

		}

		//PCA���
//		trained_pca_parameter = new pca_parameter();
//		trained_pca_parameter->load_pca_parameter( "Texture.eigen" , "Texture.mean" );

		//normalize
		input_data.normalize( 1 );
		input_data_test.normalize( 1 , input_data.get_normal_param() );

		//1�K�w�ڂ̊w�K
		netp1 = new network_parameter3( 
				input_data.get_input_node() , 
				layer_12_hidden_node , 
				input_data.get_input_node() , 
				learning_rate , lambda , momentum );

		netp1->set_network_tool( 
				info->get_network_tool( ) );
				

		//2�K�w�ڂ̊w�K
		netp2 = new network_parameter3(
			       	layer_12_hidden_node , 
				layer_23_hidden_node ,
			       	layer_12_hidden_node , 
				learning_rate , lambda , momentum );

		netp2->set_network_tool( 
				info->get_network_tool(  ) );
	
		//�e�X�^�[�̐ݒu
		if (data_structure == "aam")
		{
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
		else
		{
			netp1->set_tester(
				new tester_procrutes(train_proc, input_data.get_normal_param()),
				new tester_procrutes(test_proc, input_data.get_normal_param()));

			netp2->set_tester(
				new tester_decode_procrutes(netp1, train_proc, input_data.get_normal_param()),
				new tester_decode_procrutes(netp1, test_proc, input_data.get_normal_param()));
		}

		
	}
	~LearningFrameWork()
	{		
		//delete
		input_data.release();
		input_data_test.release();

		//�e�X�^�[	
		if(train_proc!=0)delete train_proc;
		if(test_proc!=0)delete test_proc;

		//network
		if(netp1!=0)delete netp1;
		if(netp2!=0)delete netp2;
		if(netpa!=0)delete netpa;

		if (data_structure == "aam")
		{
			delete shape_model;
			delete texture_model;
			delete shape_proc_train;
			delete texture_proc_train;
			delete shape_proc_test;
			delete texture_proc_test;

			data_shape.release();
			data_texture.release();
		}

		//PCA�p�����[�^
//		if(trained_pca_parameter!=0)delete trained_pca_parameter;

		//time
		std::cout << "time : " << ( clock() - start_time ) << std::endl;
	
	}
public:
	void Load12()
	{
		//��w�ڂ̃f�[�^�ǂݍ���
		netp1->input_weight( weight_dir_name );
		netp1->show_mse(input_data , input_data_test , "layer12");
	}

	void Learn12()
	{
		//��w�ڂ̊w�K
		netp1->learn( input_data , input_data_test , epoch );
		netp1->output_weight( weight_dir_name );
	}

	void Load23()
	{
		//�j�w�ڂ̃f�[�^�ǂݍ���
		netp2->input_weight( weight_dir_name23 );

		//2�K�w�ڂ̊w�K�f�[�^�̍쐬
		data_manager middle_data = netp1->foward_all_data( input_data );
		data_manager middle_data_test = netp1->foward_all_data( input_data_test );
	
		netp2->show_mse( middle_data , middle_data_test , "layer23" );

		middle_data.release();
		middle_data_test.release();

	}
	void Learn23()
	{
		//2�K�w�ڂ̊w�K�f�[�^�̍쐬
		data_manager middle_data = netp1->foward_all_data( input_data );
		data_manager middle_data_test = netp1->foward_all_data( input_data_test );

		//2�K�w�ڂ̊w�K
		netp2->learn( middle_data , middle_data_test , epoch );
		netp2->output_weight( weight_dir_name23 );

		middle_data.release();
		middle_data_test.release();
	}

	void LoadALL()
	{

		//���ׂĂ̑w�̊w�K
		netpa = new network_parameter5( 
				input_data.get_input_node() ,
				layer_12_hidden_node , 
				layer_23_hidden_node , 
				layer_12_hidden_node , 
				input_data.get_input_node() ,
			       	learning_rate , lambda , momentum );

		netpa->set_network_tool( 
				info->get_network_tool( ) );
	
		//�e�X�^�[�̐ݒu
		if (data_structure == "aam")
		{
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
		else
		{
			netpa->set_tester(
				new tester_procrutes(train_proc, input_data.get_normal_param()),
				new tester_procrutes(test_proc, input_data.get_normal_param()));
		}

		//�S�w�̓ǂݍ���
		netpa->input_weight( weight_dir_nameall );
	}

	void LearnAll()
	{
		if( netpa == 0 )
		{
			//���ׂĂ̑w�̊w�K
			netpa = new network_parameter5( netp1 , netp2 , learning_rate , lambda , momentum );

			netpa->set_network_tool( 
					info->get_network_tool( ) );

			//�e�X�^�[�̐ݒu
			if (data_structure == "aam")
			{
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
			else
			{
				netpa->set_tester(
					new tester_procrutes(train_proc, input_data.get_normal_param()),
					new tester_procrutes(test_proc, input_data.get_normal_param()));
			}
		}

		netpa->learn( input_data , input_data_test ,epoch );
		netpa->output_weight( weight_dir_nameall );
	}

	void output_aam_all()
	{
		//���ԃf�[�^�̏��
		data_manager middle_data = netpa->foward_all_data( input_data );
		data_manager middle_data_test = netpa->foward_all_data( input_data_test );

		//�o��(weight�ł͂Ȃ����f�[�^���A�m�[�h���A�f�[�^�̏��Ŋi�[�ł��邽��)
		output_weight( 
				middle_data.get_input_data() , 
				middle_data.get_data_count() ,
				middle_data.get_input_node() ,
				input_dir + "//train_"+data_structure+"_aam.dat"
				);

		std::cout << "output_aam : " << middle_data.get_data_count() << " : " << middle_data.get_input_node() << std::endl;	

		output_weight( 
				middle_data_test.get_input_data() , 
				middle_data_test.get_data_count() ,
				middle_data_test.get_input_node() ,
				input_dir + "//test_"+data_structure+"_aam.dat"
				);		
		
		std::cout << "output_aam_test : " << middle_data_test.get_data_count() << " : " << middle_data_test.get_input_node() << std::endl;	
	}

};


#endif
