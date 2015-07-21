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
			std::cout << "AutoEncoder‚ÌŠwK‚ª‚Å‚«‚Ü‚·" << std::endl;
			std::cout << "mˆø”‚PnŠwK—¦" << std::endl;
			std::cout << "mˆø”‚Qn³‘¥ŒW”" << std::endl;
			std::cout << "mˆø”‚Rnƒ‚[ƒƒ“ƒ^ƒ€" << std::endl;
			std::cout << "mˆø”‚SnŠwK‰ñ”" << std::endl;
			std::cout << "mˆø”‚Tn‘æˆê‰B‚ê‘w‚Ìƒm[ƒh”" << std::endl;
			std::cout << "mˆø”‚Un‘æ“ñ‰B‚ê‘w‚Ìƒm[ƒh”" << std::endl;
			std::cout << " [ˆø”‚VnŠwKƒtƒ@ƒCƒ‹‚ÌƒtƒHƒ‹ƒ_" << std::endl;
			std::cout << " [ˆø”‚WnŠwKd‚İ‚Ì•Û‘¶“Ç‚İ‚İêŠ(12‘w)" << std::endl;
			std::cout << " [ˆø”‚XnŠwKd‚İ‚Ì•Û‘¶“Ç‚İ‚İêŠ(23‘w)" << std::endl;
			std::cout << "mˆø”‚P‚OnŠwKd‚İ‚Ì•Û‘¶“Ç‚İ‚İêŠ(‘S‘w)" << std::endl;
			std::cout << "mˆø”‚P‚Pnoption ŠwK‚Ì•û–@" << std::endl;
			std::cout << "mˆø”‚P‚Qnoption ŠwK‚ÌŒW”" << std::endl;
			std::cout << " [ˆø”‚P‚RnŠwK‚·‚éí—Ş shape texture aam" << std::endl;
			std::cout << " [ˆø”‚P‚SnŠwKİ’èƒtƒ@ƒCƒ‹" << std::endl;
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

		std::cout << "AutoEncoder‚ÌŠwKŠJn" << std::endl;
		std::cout << "ŠwK—¦F" << learning_rate << std::endl;
		std::cout << "³‘¥‰»ŒW”F" << lambda << std::endl;
		std::cout << "ƒ‚[ƒƒ“ƒ^ƒ€F" << momentum << std::endl;
		std::cout << "ŠwK‰ñ”F" << epoch << std::endl; 
		std::cout << "ŠwKƒtƒHƒ‹ƒ_: " << input_dir << std::endl;
		std::cout << "12‰B‚ê"  << layer_12_hidden_node << std::endl;
		std::cout << "23‰B‚ê"  << layer_23_hidden_node << std::endl;
		std::cout << "ŠwK•û–@" << argv[11] << " " << argv[12] << std::endl;
		std::cout << "ŠwKƒf[ƒ^\‘¢" << data_structure << std::endl;
		std::cout << "ŠwKİ’èƒtƒ@ƒCƒ‹" << argv[14] << std::endl;

		start_time = clock();

		//declaration

		if( data_structure == "shape" )
		{
			m_loader = new LoadShape(learn_map);
		}
		else if( data_structure == "texture" )
		{
			m_loader = new LoadTexture(learn_map);
		}
		else if( data_structure == "aam" )
		{
			m_loader = new LoadAAM(learn_map);
		}
		else if (data_structure == "shapetexture")
		{
			m_loader = new LoadShapeTexture(learn_map);
		}

		m_loader->load_data(input_dir);

		//normalize
		m_loader->get_input_data()->normalize( 1 );
		m_loader->get_input_data_test()->normalize( m_loader->get_input_data()->get_normal_param());

		//params‚Ìo—Í
		output_normalize_parameter( weight_dir_name+"//params.txt" , m_loader->get_input_data()->get_normal_param());
		output_normalize_parameter(weight_dir_name23 + "//params.txt", m_loader->get_input_data()->get_normal_param());
		output_normalize_parameter(weight_dir_nameall + "//params.txt", m_loader->get_input_data()->get_normal_param());

		//1ŠK‘w–Ú‚ÌŠwK
		m_loader->create_netp1(
			layer_12_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);		

		//2ŠK‘w–Ú‚ÌŠwK
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
		//ˆê‘w–Ú‚Ìƒf[ƒ^“Ç‚İ‚İ
		m_loader->get_netp1()->input_weight( weight_dir_name );
		m_loader->get_netp1()->show_mse(*m_loader->get_input_data(), *m_loader->get_input_data_test(), "layer12");
	}

	void Learn12()
	{
		//ˆê‘w–Ú‚ÌŠwK
		m_loader->get_netp1()->learn(*m_loader->get_input_data(), *m_loader->get_input_data_test(), epoch);
		m_loader->get_netp1()->output_weight(weight_dir_name);
	}

	void Load23()
	{
		//ƒj‘w–Ú‚Ìƒf[ƒ^“Ç‚İ‚İ
		m_loader->get_netp2()->input_weight(weight_dir_name23);

		//2ŠK‘w–Ú‚ÌŠwKƒf[ƒ^‚Ìì¬
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());
	
		m_loader->get_netp2()->show_mse(middle_data, middle_data_test, "layer23");

		middle_data.release();
		middle_data_test.release();

	}
	void Learn23()
	{
		//2ŠK‘w–Ú‚ÌŠwKƒf[ƒ^‚Ìì¬
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());

		//2ŠK‘w–Ú‚ÌŠwK
		m_loader->get_netp2()->learn( middle_data , middle_data_test , epoch );
		m_loader->get_netp2()->output_weight(weight_dir_name23);

		middle_data.release();
		middle_data_test.release();
	}

	void LoadALL()
	{

		//‚·‚×‚Ä‚Ì‘w‚ÌŠwK
		m_loader->create_netpa(
			layer_12_hidden_node,
			layer_23_hidden_node,
			learning_rate,
			lambda,
			momentum,
			info);

		//‘S‘w‚Ì“Ç‚İ‚İ
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
		//’†ŠÔƒf[ƒ^‚Ìî•ñ
		data_manager middle_data = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netp1()->foward_all_data(*m_loader->get_input_data_test());

		//o—Í(weight‚Å‚Í‚È‚¢‚ªƒf[ƒ^”Aƒm[ƒh”Aƒf[ƒ^‚Ì‡‚ÅŠi”[‚Å‚«‚é‚½‚ß)
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
		//’†ŠÔƒf[ƒ^‚Ìî•ñ
		data_manager middle_data = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data());
		data_manager middle_data_test = m_loader->get_netpa()->foward_all_data(*m_loader->get_input_data_test());

		//o—Í(weight‚Å‚Í‚È‚¢‚ªƒf[ƒ^”Aƒm[ƒh”Aƒf[ƒ^‚Ì‡‚ÅŠi”[‚Å‚«‚é‚½‚ß)
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
