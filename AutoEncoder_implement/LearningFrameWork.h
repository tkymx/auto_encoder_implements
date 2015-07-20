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
			std::cout << "AutoEncoderの学習ができます" << std::endl;
			std::cout << "［引数１］学習率" << std::endl;
			std::cout << "［引数２］正則係数" << std::endl;
			std::cout << "［引数３］モーメンタム" << std::endl;
			std::cout << "［引数４］学習回数" << std::endl;
			std::cout << "［引数５］第一隠れ層のノード数" << std::endl;
			std::cout << "［引数６］第二隠れ層のノード数" << std::endl;
			std::cout << " [引数７］学習ファイルのフォルダ" << std::endl;
			std::cout << " [引数８］学習重みの保存読み込み場所(12層)" << std::endl;
			std::cout << " [引数９］学習重みの保存読み込み場所(23層)" << std::endl;
			std::cout << "［引数１０］学習重みの保存読み込み場所(全層)" << std::endl;
			std::cout << "［引数１１］option 学習の方法" << std::endl;
			std::cout << "［引数１２］option 学習の係数" << std::endl;
			std::cout << " [引数１３］学習する種類 shape texture aam" << std::endl;
			std::cout << " [引数１４］学習設定ファイル" << std::endl;
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

		std::cout << "AutoEncoderの学習開始" << std::endl;
		std::cout << "学習率：" << learning_rate << std::endl;
		std::cout << "正則化係数：" << lambda << std::endl;
		std::cout << "モーメンタム：" << momentum << std::endl;
		std::cout << "学習回数：" << epoch << std::endl; 
		std::cout << "学習フォルダ: " << input_dir << std::endl;
		std::cout << "12隠れ"  << layer_12_hidden_node << std::endl;
		std::cout << "23隠れ"  << layer_23_hidden_node << std::endl;
		std::cout << "学習方法" << argv[11] << " " << argv[12] << std::endl;
		std::cout << "学習データ構造" << data_structure << std::endl;
		std::cout << "学習設定ファイル" << argv[14] << std::endl;

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
		else if (data_structure == "shapetexture")
		{
			m_loader = new LoadShapeTexture();
		}

		m_loader->load_data(input_dir, learn_map);

		//normalize
		m_loader->get_input_data()->normalize( 1 );
		m_loader->get_input_data_test()->normalize(1, m_loader->get_input_data()->get_normal_param());

		//paramsの出力
		output_normalize_parameter( weight_dir_name+"//params.txt" , m_loader->get_input_data()->get_normal_param());
		output_normalize_parameter(weight_dir_name23 + "//params.txt", m_loader->get_input_data()->get_normal_param());
		output_normalize_parameter(weight_dir_nameall + "//params.txt", m_loader->get_input_data()->get_normal_param());

		//1階層目の学習
		m_loader->create_netp1(
			layer_12_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);
				

		//2階層目の学習
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
		//一層目のデータ読み込み
		m_loader->get_netp1()->input_weight( weight_dir_name );
		m_loader->get_netp1()->show_mse(*m_loader->get_input_data(), *m_loader->get_input_data_test(), "layer12");
	}

	void Learn12()
	{
		//一層目の学習
		m_loader->get_netp1()->learn(*m_loader->get_input_data(), *m_loader->get_input_data_test(), epoch);
		m_loader->get_netp1()->output_weight(weight_dir_name);
	}

	void Load23()
	{
		//ニ層目のデータ読み込み
		m_loader->get_netp2()->input_weight(weight_dir_name23);

		//2階層目の学習データの作成
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());
	
		m_loader->get_netp2()->show_mse(middle_data, middle_data_test, "layer23");

		middle_data.release();
		middle_data_test.release();

	}
	void Learn23()
	{
		//2階層目の学習データの作成
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());

		//2階層目の学習
		m_loader->get_netp2()->learn( middle_data , middle_data_test , epoch );
		m_loader->get_netp2()->output_weight(weight_dir_name23);

		middle_data.release();
		middle_data_test.release();
	}

	void LoadALL()
	{

		//すべての層の学習
		m_loader->create_netpa(
			layer_12_hidden_node,
			layer_23_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);

		//全層の読み込み
		m_loader->get_netpa()->input_weight( weight_dir_nameall );
		m_loader->get_netpa()->show_mse(*m_loader->get_input_data(), *m_loader->get_input_data_test(), "layerAll");
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

	void output_aam_12()
	{
		//中間データの情報
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());

		//出力(weightではないがデータ数、ノード数、データの順で格納できるため)
		output_weight(
			middle_data.get_input_data(),
			middle_data.get_data_count(),
			middle_data.get_input_node(),
			input_dir + "//train_" + data_structure + "_aam.dat"
			);

		std::cout << "output_aam : " << middle_data.get_data_count() << " : " << middle_data.get_input_node() << std::endl;

		output_weight(
			middle_data_test.get_input_data(),
			middle_data_test.get_data_count(),
			middle_data_test.get_input_node(),
			input_dir + "//test_" + data_structure + "_aam.dat"
			);

		std::cout << "output_aam_test : " << middle_data_test.get_data_count() << " : " << middle_data_test.get_input_node() << std::endl;
	}

	void output_aam_all()
	{
		//中間データの情報
		data_manager middle_data = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data_test());

		//出力(weightではないがデータ数、ノード数、データの順で格納できるため)
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

	void output_mse_each_12()
	{
		m_loader->get_netp1()->output_each_mse_test(*m_loader->get_input_data_test());
	}

	void output_mse_each()
	{
		m_loader->get_netpa()->output_each_mse_test(*m_loader->get_input_data_test());
	}

};


#endif
