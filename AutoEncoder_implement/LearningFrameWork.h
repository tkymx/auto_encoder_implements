#include "LearningFrameWorkUtility.h"

#ifndef LEARNING_FRAME_WORK
#define LEARNING_FRAME_WORK

class LearningFrameWork
{
private:

	int layer_12_hidden_node;
	int layer_23_hidden_node;

	float learning_rate;
	float lambda;
	float momentum;
	int epoch;

	std::string input_dir;
	std::string data_structure;

	std::string weight_dir_name;
	std::string weight_dir_name23;
	std::string weight_dir_nameall;
	

	learn_info *info;

	std::map<std::string, std::string> learn_map;

	LoadBase* m_loader;

	clock_t start_time;

public:
	LearningFrameWork(int argc , char* argv[])
		: m_loader(NULL)
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
			m_loader = new LoadShape();
		}
		else if( data_structure == "texture" )
		{
			m_loader = new LoadTexture();
		}
		else if( data_structure == "aam" )
		{
			m_loader = new LoadAAM();
		}

		m_loader->load_data(input_dir, learn_map);

		//normalize
		m_loader->get_input_data()->normalize( 1 );
		m_loader->get_input_data_test()->normalize(1, m_loader->get_input_data()->get_normal_param());

		//1�K�w�ڂ̊w�K
		m_loader->create_netp1(
			layer_12_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);
				

		//2�K�w�ڂ̊w�K
		m_loader->create_netp2(
			layer_12_hidden_node,
			layer_23_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);

		
	}
	~LearningFrameWork()
	{		
		//delete
		if(m_loader!=NULL)delete m_loader;

		//time
		std::cout << "time : " << ( clock() - start_time ) << std::endl;
	
	}
public:
	void Load12()
	{
		//��w�ڂ̃f�[�^�ǂݍ���
		m_loader->get_netp1()->input_weight( weight_dir_name );
		m_loader->get_netp1()->show_mse(*m_loader->get_input_data(), *m_loader->get_input_data_test(), "layer12");
	}

	void Learn12()
	{
		//��w�ڂ̊w�K
		m_loader->get_netp1()->learn(*m_loader->get_input_data(), *m_loader->get_input_data_test(), epoch);
		m_loader->get_netp1()->output_weight(weight_dir_name);
	}

	void Load23()
	{
		//�j�w�ڂ̃f�[�^�ǂݍ���
		m_loader->get_netp2()->input_weight(weight_dir_name23);

		//2�K�w�ڂ̊w�K�f�[�^�̍쐬
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());
	
		m_loader->get_netp2()->show_mse(middle_data, middle_data_test, "layer23");

		middle_data.release();
		middle_data_test.release();

	}
	void Learn23()
	{
		//2�K�w�ڂ̊w�K�f�[�^�̍쐬
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());

		//2�K�w�ڂ̊w�K
		m_loader->get_netp2()->learn( middle_data , middle_data_test , epoch );
		m_loader->get_netp2()->output_weight(weight_dir_name23);

		middle_data.release();
		middle_data_test.release();
	}

	void LoadALL()
	{

		//���ׂĂ̑w�̊w�K
		m_loader->create_netpa(
			layer_12_hidden_node,
			layer_23_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);

		//�S�w�̓ǂݍ���
		m_loader->get_netpa()->input_weight( weight_dir_nameall );
	}

	void LearnAll()
	{
		if (m_loader->get_netpa() == 0)
		{
			m_loader->create_netpa(
				learning_rate,
				lambda,
				momentum,
				info);
		}

		m_loader->get_netpa()->learn(*m_loader->get_input_data(), *m_loader->get_input_data_test(), epoch);
		m_loader->get_netpa()->output_weight(weight_dir_nameall);
	}

	void output_aam_all()
	{
		//���ԃf�[�^�̏��
		data_manager middle_data = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data_test());

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
