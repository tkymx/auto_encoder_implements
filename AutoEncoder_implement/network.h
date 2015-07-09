#include"utility.h"
#include"data.h"

#ifndef NETWORK
#define NETWORK

/**
 *	データのシャッフル
 */
void data_shuffle( float** input_data , int data_count )
{
	for( int i = 0 ; i < data_count ; i++ )
	{
		int s1 = static_cast<int>(rand_range(0, static_cast<float>(data_count)));
		int s2 = static_cast<int>(rand_range(0, static_cast<float>(data_count)));
//		std::cout << i << ":" << s1 << ":" << s2 << std::endl;
		std::swap( s1 , s2 );
	}
}

/**
 *	ネットワークツール
 */

enum learn_mode{ learn_mse=0 , learn_cross_entropy=1 };

enum learn_tool{ noise_normal_tool = 0, corrupt_tool = 1, none_tool = 2 };

/*
*	学習時のインターフェース
*	：事前処理など
*/
class network_tool
{
private:
	learn_mode cost_function;
public:
	network_tool(learn_mode cost)
		: cost_function(cost)
	{
	}
public:
	learn_mode get_learn_mode()
	{
		return cost_function;
	}
public:
	virtual void prepare_data(data_manager &data)
	{
	}
};

class network_tool_denoising_corrupt : public network_tool
{
private:
	float corrupt_rate;
public:
	network_tool_denoising_corrupt(learn_mode cost, float rate)
		: network_tool(cost)
		, corrupt_rate(rate)
	{
	}
public:
	virtual void prepare_data(data_manager &data)
	{
		corrupt(data.get_input_data(), data.get_data_count(), data.get_input_node(), corrupt_rate);
	}
};

class network_tool_denoising_noise_normal : public network_tool
{
private:
	float sigma;
public:
	network_tool_denoising_noise_normal(learn_mode cost, float s)
		: network_tool(cost)
		, sigma(s)
	{
	}
public:
	virtual void prepare_data(data_manager &data)
	{
		noise_normal(data.get_input_data(), data.get_data_count(), data.get_input_node(), sigma);
	}
};


/*
*	学習情報の保持を行う
*	：DEAを実現するための欠損情報
*	：学習のコスト関数など
*/
class learn_info
{
public:
	learn_mode mode;
	learn_tool tool;
	float info_value;
public:
	learn_info(std::string str, float value)
		: info_value(value)
		, mode(learn_mse)
		, tool(none_tool)
	{
		if (str.find("mse") != std::string::npos)
		{
			mode = learn_mse;
		}
		else if (str.find("MSE") != std::string::npos)
		{
			mode = learn_mse;
		}
		else if (str.find("ce") != std::string::npos)
		{
			mode = learn_cross_entropy;
		}
		else if (str.find("crossentropy") != std::string::npos)
		{
			mode = learn_cross_entropy;
		}
		else if (str.find("CrossEntropy") != std::string::npos)
		{
			mode = learn_cross_entropy;
		}
		else if (str.find("CROSSENTROPY") != std::string::npos)
		{
			mode = learn_cross_entropy;
		}

		if (str.find("noise_normal") != std::string::npos)
		{
			tool = noise_normal_tool;
		}
		else if (str.find("nn") != std::string::npos)
		{
			tool = noise_normal_tool;
		}
		else if (str.find("corrupt") != std::string::npos)
		{
			tool = corrupt_tool;
		}
	}

	network_tool* get_network_tool()
	{

		if (tool == noise_normal_tool)
		{
			return new network_tool_denoising_noise_normal(mode, info_value);
		}
		if (tool == corrupt_tool)
		{
			return new network_tool_denoising_corrupt(mode, info_value);
		}
		return new network_tool(mode);
	}
};

/*
*	ネットワークの学習情報をテキストで持っておいて読み込みを行う
*	(e.f.) shape_model : /home/data/model/shape 
*/
std::map<std::string, std::string> get_learn_dat(std::string filename)
{
	std::map<std::string, std::string> learn_map;

	std::ifstream ifs(filename.c_str());
	if (!ifs.is_open())
	{
		std::cout << filename << "がありませんでした。" << std::endl;
	}

	while (!ifs.eof())
	{
		std::string str;
		std::getline(ifs, str);

		int split = str.find_first_of(":");

		std::string tag = DeleteSpace(str.substr(0, split));
		std::string text = DeleteSpace(str.substr(split + 1));

		learn_map.insert(std::make_pair(tag, text));

	}
	return learn_map;
}

template< class T>
T get_map_param(std::map<std::string, std::string> learn_map, std::string tag)
{
	return learn_map[tag];
}
template<>
int get_map_param(std::map<std::string, std::string> learn_map, std::string tag)
{
	return atoi(learn_map[tag].c_str());
}
template<>
float get_map_param(std::map<std::string, std::string> learn_map, std::string tag)
{
	return static_cast<float>(atof(learn_map[tag].c_str()));
}

/*
*　harving クラス
*/
class network_parameter;
class harving_implements
{
protected:
	int m_average_count;
	int m_current_count;
	float m_current_mse;
	float m_before_mse;
	network_parameter *m_network;
public:
	harving_implements(int count , network_parameter* net)
		: m_average_count(count)
		, m_current_count(0)
		, m_current_mse(0)
		, m_before_mse(10000)
		, m_network(net)
	{
	}
	virtual ~harving_implements()
	{
	}
public:
	bool _exec(float mse)
	{
		if (mse > m_before_mse || abs(mse-m_before_mse) < 0.0001 )
		{
			//ウェイトを戻す
			load_weight();
			
			//学習率の現象
			less_learning_rate();

			return false;
		}
		m_before_mse = mse;
		store_weight();
		return true;
	}
	bool exec(float mse)
	{
		//一定区間ごとの平均を求める
		m_current_mse += mse / m_average_count;
		m_current_count++;

		bool result = true;

		//一定期間ごとにハービングを行う。
		if (m_current_count >= m_average_count)
		{
			result = _exec(m_current_mse);

			m_current_mse = 0;
			m_current_count = 0;
		}

		return result;
	}

public:
	virtual void load_weight() = 0;
	virtual void store_weight() = 0;
	virtual void less_learning_rate() = 0;
};

/**
 *	ネットワークパラメータ	
 */	

class network_parameter
{
public:

	int input_node;

	tester *train_tester;
	tester *test_tester;

	network_tool *m_network_tool;

	clock_t current_time;

	float learning_rate;
	float learning_rate_th;
	float lambda;
	float momentum;

	harving_implements *m_harving;

public:

	network_parameter(int _input_node, float lr, float lam, float mo)
		: train_tester(new tester())
		, test_tester(new tester())
		, m_network_tool(NULL)
		, learning_rate(lr)
		, learning_rate_th(static_cast<float>(lr/pow(2,1)))
		, lambda(lam)
		, momentum(mo)
		, m_harving(NULL)
	{
		input_node = _input_node;
	}
	virtual ~network_parameter()
	{
		if( train_tester != 0 )delete train_tester;
		if( test_tester != 0 )delete test_tester;
		if (m_harving != NULL)delete m_harving;
	}

public:

	void set_network_tool( network_tool *_tool )
	{
		m_network_tool = _tool;
	}

	int get_input_node()
	{
		return input_node;
	}

	void set_tester( tester *_train_tester , tester *_test_tester )
	{
		if( train_tester != 0 )delete train_tester;
		if( test_tester != 0 )delete test_tester;
		train_tester = _train_tester;
		test_tester = _test_tester;
	}

	void epoch( data_manager &input_data , bool isSingle = false)
	{
		float** output_data = new_array( input_data.get_data_count() , input_node );

#ifdef TIME_TEST
		clock_t foward_time = 0;
		clock_t back_time = 0;
		vb_time = 0;
		hb_time = 0;
		w_time = 0;
		copy_time = 0;
		copy_weight_time = 0;
		copy_weight1_time = 0;
		copy_weight2_time = 0;
		copy_bias_time = 0;
#endif

		data_manager corrupt_data = input_data.copy();

		//データの事前処理
		prepare_data( corrupt_data );

		for( int data = 0; data < input_data.get_data_count() ; data++ )
		{
#ifdef TIME_TEST	
			//std::cout << "(" << data << "/" << input_data.get_data_count() << ")\r" << std::flush; 	
			current_time = clock();
#endif

			//foward
			foward( corrupt_data.get_input_data()[data] , output_data[data] , isSingle );

#ifdef TIME_TEST	
			foward_time += ( clock() - current_time );
			current_time = clock();

#endif
			//backpropagate
			backpropagate( input_data.get_input_data()[data] , output_data[data] , isSingle );

#ifdef TIME_TEST	
			back_time += ( clock() - current_time );
			current_time = clock();
#endif
		}

#ifdef TIME_TEST
		std::cout << "time foward: " << foward_time << std::endl;
		std::cout << "time backpropagate_vb: " << vb_time << std::endl;
		std::cout << "time backpropagate_hb: " << hb_time << std::endl;
		std::cout << "time backpropagate_w: " << w_time << std::endl;
		std::cout << "time backpropagate_copy_weight: " << copy_weight_time << std::endl;
		std::cout << "time backpropagate_copy_weight1: " << copy_weight1_time << std::endl;
		std::cout << "time backpropagate_copy_weight2: " << copy_weight2_time << std::endl;
		std::cout << "time backpropagate_copy_bias: " << copy_bias_time << std::endl;
		std::cout << "time backpropagate_copy: " << copy_time << std::endl;
		std::cout << "time backpropagate: " << back_time << std::endl;
		current_time = clock();
#endif

		delete_array( output_data , input_data.get_data_count() );

		corrupt_data.release();

#ifdef TIME_TEST	
		std::cout << "time delete output data: " << clock() - current_time << std::endl;
		current_time = clock();
#endif

	}
	void learn( data_manager &data , data_manager &data_test, int _epoch , bool isSingle = false )
	{
		//learning
		for( int e = 0; e < _epoch ; e++ )
		{
		
#ifdef TIME_TEST			
			std::cout << "time: start" << std::endl;
			current_time = clock();
#endif
			//学習順番のシャッフル
			data_shuffle( data.get_input_data() , data.get_data_count() );

#ifdef TIME_TEST			
			std::cout << "time shuffle: " << clock() - current_time << std::endl;
			current_time = clock();
#endif

			//learn
			epoch( data , isSingle );
			
			//test
			std::cout << "epoch" << e ;
			std::cout << ":test:" << show_calc_mse( data_test , test_tester );
			std::cout << ":train:" << show_calc_mse( data , train_tester ) << ":lr:" << learning_rate << std::endl;

#ifdef TIME_TEST			
			std::cout << "time test : " << clock() - current_time << std::endl;
			current_time = clock();
#endif

			//ハービング動作
			if (m_harving != NULL)
			{
				if (!m_harving->exec(get_calc_mse(data_test, test_tester)))
				{
					e -= 10;
				}
			}

			//終了処理
			if (learning_rate <= learning_rate_th)
			{
				break;
			}
		}
	}

	void show_mse( data_manager &data , data_manager &data_test,std::string str)
	{
		//test
		std::cout << str << ":test:" << show_calc_mse( data_test , test_tester );
		std::cout << ":train:" << show_calc_mse( data , train_tester ) << std::endl;
	
	}

	data_manager foward_all_data( data_manager data , bool isSingle = false )
	{
		data_manager output_data = data_manager::create_data( data.get_data_count() , get_hidden_node_count() );
		for( int i = 0 ;i < data.get_data_count(); i++ )
		{
			this->encode( data.get_input_data()[i] , output_data.get_input_data()[i] , isSingle );
		}
		return output_data;
	}
	data_manager decode_all_data(data_manager data, bool isSingle = false)
	{
		data_manager output_data = data_manager::create_data(data.get_data_count(), input_node);
		for (int i = 0; i < data.get_data_count(); i++)
		{
			this->decode(data.get_input_data()[i], output_data.get_input_data()[i], isSingle);
		}
		return output_data;
	}

	std::string show_calc_mse(data_manager &input, tester *_tester, bool isSingle = false)
	{
		data_manager output = input.copy();
	
		for( int i = 0; i< input.get_data_count() ; i++ )
		{
			foward( input.get_input_data()[i] , output.get_input_data()[i] , isSingle );
		}
		
		std::string mse = _tester->show_mse( input , output );

		output.release();

		return mse;
	}

	float get_calc_mse(data_manager &input, tester *_tester, bool isSingle = false)
	{
		data_manager output = input.copy();

		for (int i = 0; i< input.get_data_count(); i++)
		{
			foward(input.get_input_data()[i], output.get_input_data()[i], isSingle);
		}

		float mse = _tester->get_mse(input, output);

		output.release();

		return mse;
	}

	
public:
	virtual void foward( float* input_data, float* output_data , bool isSingle )=0;
	virtual void encode( float* input_data, float* hidden_data , bool isSingle )=0;
	virtual void decode( float* hidden_data, float* output_data , bool isSingle )=0;
	virtual void backpropagate( float* input_data , float* output_data , bool isSingle )=0;
	virtual void output_weight( std::string dir_name )=0;
	virtual void input_weight( std::string dir_name )=0;

public:
	void prepare_data( data_manager &data )
	{
		if( m_network_tool != NULL )
		{
			m_network_tool->prepare_data( data );
		}
	}

public:
	virtual int get_hidden_node_count() = 0;	

};

class network_parameter3 : public network_parameter
{
public:

	int hidden2_node;
	int output_node;

	float* hidden2_data;
	
	float** weight12;
	float** weight23;
	
	float** weight12_store;
	float** weight23_store;	
	
	float** weight12_d;
	float** weight23_d;
	
	float* dlhb;
	float* dlvb;

	float* ppde12;
	float* ppde23;

	class harving_implements_3 : public harving_implements
	{
	protected:
		float** weight12_before;
		float** weight23_before;

		int hidden;
		int input;
		int output;

	public:
		harving_implements_3(int count, network_parameter3* net3)
			: harving_implements(count, net3)
		{
			hidden = net3->hidden2_node;
			input = net3->input_node;
			output = net3->output_node;

			weight12_before = new_array(hidden,input+1);
			weight23_before = new_array(output, hidden+1);
		}
		virtual ~harving_implements_3()
		{
			delete_array(weight12_before, hidden);
			delete_array(weight23_before, output);
		}
	public:
		void load_weight()
		{
			network_parameter3* net3 = dynamic_cast<network_parameter3*>(m_network);
			copy_array(weight12_before, net3->weight12, net3->get_hidden_node_count(), net3->input_node + 1);
			copy_array(weight23_before, net3->weight23, net3->output_node, net3->get_hidden_node_count() + 1 );
		}
		void store_weight()
		{
			network_parameter3* net3 = dynamic_cast<network_parameter3*>(m_network);
			copy_array(net3->weight12, weight12_before, net3->get_hidden_node_count(), net3->input_node + 1);
			copy_array(net3->weight23, weight23_before, net3->output_node, net3->get_hidden_node_count() + 1);
		}
		void less_learning_rate()
		{
			m_network->learning_rate /= 2;
		}
	};

public:

	network_parameter3( int _input_node , int _middle_node , int _output_node , float lr , float lam , float mo)
		: network_parameter(_input_node,lr,lam,mo)
	{
		hidden2_node = _middle_node;
		output_node = _output_node;	

		hidden2_data = new float[hidden2_node];

		weight12 = new_array( hidden2_node , input_node+1 );
		weight23 = new_array( output_node , hidden2_node+1 );

		weight12_store = new_array( hidden2_node , input_node+1);
		weight23_store = new_array( output_node , hidden2_node+1); 

		weight12_d = new_array( hidden2_node , input_node+1 );
		weight23_d = new_array( output_node , hidden2_node+1 );

		dlhb = new float[hidden2_node];
		dlvb = new float[output_node];

		ppde12 = new float[hidden2_node];
		ppde23 = new float[output_node];

		//initialize	
		init_weight_data( weight12 , hidden2_node , input_node);
		init_weight_data( weight23 , output_node , hidden2_node );
		init_data( weight12_store , hidden2_node , output_node + 1 );
		init_data( weight23_store , output_node , hidden2_node + 1 );
		init_data( weight12_d , hidden2_node , input_node + 1 );
		init_data( weight23_d , output_node , hidden2_node + 1 );

		//値のコピー
		//copy_array( weight12 , weight12_store , hidden2_node , input_node );
		//copy_array( weight23 , weight23_store , output_node , hidden2_node );

		//こっちのほうが性能が良い
		update_weight_store( weight12 , weight23 , weight12_store , weight23_store , input_node , hidden2_node );

		//ハービングの設定
		m_harving = new harving_implements_3( 20 , this );

		std::cout << "network_parameter3:"  << input_node << " " << hidden2_node  << " " << output_node << std::endl;
		
	}
	~network_parameter3()
	{

		delete_value( hidden2_data );

		delete_array( weight12 , hidden2_node );
		delete_array( weight23 , output_node );
		delete_array( weight12_store , hidden2_node );
		delete_array( weight23_store , output_node );
		delete_array( weight12_d , hidden2_node );
		delete_array( weight23_d , output_node );

		delete_value( dlhb );
		delete_value( dlvb );

		delete_value( ppde12 );
		delete_value( ppde23 );	
	}

public:


	virtual void foward( float* input_data, float* output_data , bool isSingle = false)
	{
		::foward( input_data , hidden2_data , weight12 , input_node , hidden2_node , isSingle );
		::foward( hidden2_data , output_data , weight23 , hidden2_node , output_node , isSingle );
	}
	virtual void encode( float* input_data, float* hidden_data , bool isSingle = false )
	{
		::foward( input_data , hidden_data , weight12 , input_node , hidden2_node , isSingle );
	}	
	virtual void decode( float* hidden_data, float* output_data , bool isSingle = false)
	{
		::foward( hidden_data , output_data , weight23 , hidden2_node , output_node , isSingle );
	}

	virtual void backpropagate( float* input_data , float* output_data , bool isSingle = false )
	{
		if( m_network_tool->get_learn_mode() == learn_cross_entropy )
		{
			::backpropagate_cross_entropy( 
				input_data , hidden2_data , output_data , 
				weight12 , weight23 , weight12_store , weight23_store  ,weight12_d , weight23_d ,
				learning_rate , lambda , momentum ,
				dlvb , dlhb , input_node , hidden2_node , output_node ,
			        isSingle );
		}
		else if( m_network_tool->get_learn_mode() == learn_mse )
		{
			::backpropagate_mse( 
				input_data , hidden2_data , output_data,
				weight12 , weight23 , weight12_store , weight23_store , weight12_d , weight23_d,
				learning_rate , lambda , momentum ,
				ppde12 , ppde23,
				input_node , hidden2_node , output_node);
		}
	}
	virtual int get_hidden_node_count()
	{
		return hidden2_node;
	}

	virtual void output_weight( std::string dir_name )
	{
		::output_weight( weight12 , hidden2_node , input_node , dir_name + "//weight_0.bin"  );
		::output_weight( weight23 , output_node , hidden2_node , dir_name + "//primeweight_0.bin"  );
		::output_bias( weight12 , hidden2_node , input_node , dir_name + "//bias_0.bin" );
		::output_bias( weight23 , output_node , hidden2_node , dir_name + "//vbias_0.bin" );
	}
	
	virtual void input_weight( std::string dir_name )
	{
		::input_weight( weight12 , hidden2_node , input_node , dir_name + "//weight_0.bin"  );
		::input_weight( weight23 , input_node , hidden2_node , dir_name + "//primeweight_0.bin"  );
		::input_bias( weight12 , hidden2_node , input_node , dir_name + "//bias_0.bin" );
		::input_bias( weight23 , output_node , hidden2_node , dir_name + "//vbias_0.bin" );
	}

	friend class network_parameter5;

};

class network_parameter5 : public network_parameter
{
private:

	int hidden2_node;
	int hidden3_node;
	int hidden4_node;
	int output_node;

	float* hidden2_data;
	float* hidden3_data;
	float* hidden4_data;

	float** weight12;
	float** weight23;
	float** weight34;
	float** weight45;

	float** weight12_store;
	float** weight23_store;
	float** weight34_store;
	float** weight45_store;

	float** weight12_d;
	float** weight23_d;
	float** weight34_d;
	float** weight45_d;

	float* ppde12;
	float* ppde23;
	float* ppde34;
	float* ppde45;

	class harving_implements_5 : public harving_implements
	{
	protected:
		float** weight12_before;
		float** weight23_before;
		float** weight34_before;
		float** weight45_before;

		int input;
		int hidden2;
		int hidden3;
		int hidden4;
		int output;

	public:
		harving_implements_5(int count, network_parameter5* net5)
			: harving_implements(count, net5)
		{
			weight12_before = new_array(net5->hidden2_node, net5->input_node + 1);
			weight23_before = new_array(net5->hidden3_node, net5->hidden2_node + 1);
			weight34_before = new_array(net5->hidden4_node, net5->hidden3_node + 1);
			weight45_before = new_array(net5->output_node, net5->hidden4_node + 1);

			input = net5->input_node;
			hidden2 = net5->hidden2_node;
			hidden3 = net5->hidden3_node;
			hidden4 = net5->hidden4_node;
			output = net5->output_node;
		}
		virtual ~harving_implements_5()
		{
			delete_array(weight12_before, hidden2);
			delete_array(weight23_before, hidden3);
			delete_array(weight34_before, hidden4);
			delete_array(weight45_before, output);
		}
	public:
		void load_weight()
		{
			network_parameter5* net5 = dynamic_cast<network_parameter5*>(m_network);
			copy_array(weight12_before, net5->weight12, net5->hidden2_node, net5->input_node);
			copy_array(weight23_before, net5->weight23, net5->hidden3_node, net5->hidden2_node);
			copy_array(weight34_before, net5->weight34, net5->hidden4_node, net5->hidden3_node);
			copy_array(weight45_before, net5->weight45, net5->output_node, net5->hidden4_node);
		}
		void store_weight()
		{
			network_parameter5* net5 = dynamic_cast<network_parameter5*>(m_network);
			copy_array(net5->weight12, weight12_before, net5->hidden2_node, net5->input_node);
			copy_array(net5->weight23, weight23_before, net5->hidden3_node, net5->hidden2_node);
			copy_array(net5->weight34, weight34_before, net5->hidden4_node, net5->hidden3_node);
			copy_array(net5->weight45, weight45_before, net5->output_node, net5->hidden4_node);
		}
		void less_learning_rate()
		{
			m_network->learning_rate /= 2;
		}
	};

public:

	void initialize_network( int _input_node , int _middle2_node , int _middle3_node , int _middle4_node , int _output_node)
	{
		hidden2_node = _middle2_node;
		hidden3_node = _middle3_node;
		hidden4_node = _middle4_node;
		output_node = _output_node;

		hidden2_data = new float[hidden2_node];
		hidden3_data = new float[hidden3_node];
		hidden4_data = new float[hidden4_node];

		weight12 = new_array( hidden2_node , input_node+1 );
		weight23 = new_array( hidden3_node , hidden2_node+1 );
		weight34 = new_array( hidden4_node , hidden3_node+1 );
		weight45 = new_array( output_node , hidden4_node+1 );

		weight12_store = new_array( hidden2_node , input_node+1);
		weight23_store = new_array( hidden3_node , hidden2_node+1); 
		weight34_store = new_array( hidden4_node , hidden3_node+1); 
		weight45_store = new_array( output_node , hidden4_node+1); 

		weight12_d = new_array( hidden2_node , input_node+1 );
		weight23_d = new_array( hidden3_node , hidden2_node+1 );
		weight34_d = new_array( hidden4_node , hidden3_node+1 );
		weight45_d = new_array( output_node , hidden4_node+1 );

		ppde12 = new float[hidden2_node];
		ppde23 = new float[hidden3_node];
		ppde34 = new float[hidden4_node];
		ppde45 = new float[output_node];

		//initialize	
		init_weight_data( weight12 , hidden2_node , input_node);
		init_weight_data( weight23 , hidden3_node , hidden2_node );
		init_weight_data( weight34 , hidden4_node , hidden3_node );
		init_weight_data( weight45 , output_node , hidden4_node );
		
		init_data( weight12_store , hidden2_node , output_node + 1 );
		init_data( weight23_store , hidden3_node , hidden2_node + 1 );
		init_data( weight34_store , hidden4_node , hidden3_node + 1 );
		init_data( weight45_store , output_node , hidden4_node + 1 );
		
		init_data( weight12_d , hidden2_node , input_node + 1 );
		init_data( weight23_d , hidden3_node , hidden2_node + 1 );
		init_data( weight34_d , hidden4_node , hidden3_node + 1 );
		init_data( weight45_d , output_node , hidden4_node + 1 );		

		//ハービングの設定
		m_harving = new harving_implements_5(20, this);

	}


	network_parameter5(  int _input_node , int _middle2_node , int _middle3_node , int _middle4_node , int _output_node , float lr , float lam , float mo)
		: network_parameter(_input_node, lr, lam, mo)
	{

		initialize_network( 
				_input_node , 
				_middle2_node , 
				_middle3_node , 
				_middle4_node , 
				_output_node );

		//値のコピー
		//copy_array( weight12 , weight12_store , hidden2_node , input_node );
		//copy_array( weight23 , weight23_store , output_node , hidden2_node );

		//こっちのほうが性能が良い
		update_weight_store( weight12 , weight45 , weight12_store , weight45_store , input_node , hidden2_node );
		update_weight_store( weight23 , weight34 , weight23_store , weight34_store , hidden2_node , hidden3_node );

		std::cout << "network_parameter5:" << input_node << " " << hidden2_node  << " " << output_node << std::endl;

	}
	
	network_parameter5( network_parameter3 *netp1 , network_parameter3 *netp2, float lr , float lam , float mo )
		: network_parameter(netp1->input_node , lr, lam, mo)
	{

		initialize_network( 
				netp1->input_node , 
				netp1->hidden2_node ,
			       	netp2->hidden2_node , 
				netp1->hidden2_node , 
				netp1->input_node );
		
		copy_array( netp1->weight12 , weight12 , hidden2_node , input_node + 1);
		copy_array( netp2->weight12 , weight23 , hidden3_node , hidden2_node + 1);
		copy_array( netp2->weight23 , weight34 , hidden4_node , hidden3_node + 1);
		copy_array( netp1->weight23 , weight45 , output_node , hidden4_node + 1);

		//値のコピー
		//copy_array( weight12 , weight12_store , hidden2_node , input_node );
		//copy_array( weight23 , weight23_store , output_node , hidden2_node );

		//こっちのほうが性能が良い
		update_weight_store( weight12 , weight45 , weight12_store , weight45_store , input_node , hidden2_node );
		update_weight_store( weight23 , weight34 , weight23_store , weight34_store , hidden2_node , hidden3_node );

		std::cout << "network_parameter5:" << input_node << " " << hidden2_node  << " " << output_node << std::endl;
	}

	~network_parameter5()
	{
		delete_value( hidden2_data );
		delete_value( hidden3_data );
		delete_value( hidden4_data );

		delete_array( weight12 , hidden2_node );
		delete_array( weight23 , hidden3_node );
		delete_array( weight34 , hidden4_node );
		delete_array( weight45 , output_node );
		
		delete_array( weight12_store , hidden2_node );
		delete_array( weight23_store , hidden3_node );
		delete_array( weight34_store , hidden4_node );
		delete_array( weight45_store , output_node );

		delete_array( weight12_d , hidden2_node );
		delete_array( weight23_d , hidden3_node );
		delete_array( weight34_d , hidden4_node );
		delete_array( weight45_d , output_node );

		delete_value( ppde12 );
		delete_value( ppde23 );			
		delete_value( ppde34 );
		delete_value( ppde45 );
	}

public:

	virtual void foward( float* input_data, float* output_data , bool isSingle = false )
	{
		::foward( input_data , hidden2_data , weight12 , input_node , hidden2_node , isSingle);
		::foward( hidden2_data , hidden3_data , weight23 , hidden2_node , hidden3_node , isSingle);
		::foward( hidden3_data , hidden4_data , weight34 , hidden3_node , hidden4_node , isSingle);
		::foward( hidden4_data , output_data , weight45 , hidden4_node , output_node , isSingle);
	}
	
	virtual void encode( float* input_data, float* hidden_data , bool isSingle = false )
	{
		::foward( input_data , hidden2_data , weight12 , input_node , hidden2_node , isSingle );
		::foward( hidden2_data , hidden_data , weight23 , hidden2_node , hidden3_node , isSingle );
	}
	
	virtual void decode( float* hidden_data, float* output_data , bool isSingle = false )
	{
		::foward( hidden_data , hidden4_data , weight34 , hidden3_node , hidden4_node , isSingle);
		::foward( hidden4_data , output_data , weight45 , hidden4_node , output_node , isSingle);
	}

	virtual void backpropagate( float* input_data , float* output_data , bool isSingle = false )
	{

		::backpropagate_mse_5( 
				input_data , hidden2_data , hidden3_data , hidden4_data , output_data,
				weight12 , weight23 , weight34 , weight45 , 
				weight12_store , weight23_store , weight34_store , weight45_store,
			       	weight12_d , weight23_d, weight34_d , weight45_d ,
				learning_rate , lambda , momentum ,
				ppde12 , ppde23, ppde34 , ppde45 ,
				input_node , hidden2_node , hidden3_node , hidden4_node , output_node);
	}
	virtual int get_hidden_node_count()
	{
		return hidden3_node;
	}

	friend class network_parameter3;

	virtual void output_weight( std::string dir_name )
	{
		::output_weight( weight12 , hidden2_node , input_node , dir_name + "//weight_0.bin"  );
		::output_weight( weight23 , hidden3_node , hidden2_node , dir_name + "//weight_1.bin"  );
		::output_weight( weight34 , hidden4_node , hidden3_node , dir_name + "//primeweight_1.bin"  );
		::output_weight( weight45 , output_node , hidden4_node , dir_name + "//primeweight_0.bin"  );

		::output_bias( weight12 , hidden2_node , input_node , dir_name + "//bias_0.bin" );
		::output_bias( weight23 , hidden3_node , hidden2_node , dir_name + "//bias_1.bin" );
		::output_bias( weight34 , hidden4_node , hidden3_node , dir_name + "//vbias_1.bin" );
		::output_bias( weight45 , output_node , hidden4_node , dir_name + "//vbias_0.bin" );
	}
	
	virtual void input_weight( std::string dir_name )
	{
		::input_weight( weight12 , hidden2_node , input_node , dir_name + "//weight_0.bin"  );
		::input_weight( weight23 , hidden3_node , hidden2_node , dir_name + "//weight_1.bin"  );
		::input_weight( weight34 , hidden4_node , hidden3_node , dir_name + "//primeweight_1.bin"  );
		::input_weight( weight45 , output_node , hidden4_node , dir_name + "//primeweight_0.bin"  );

		::input_bias( weight12 , hidden2_node , input_node , dir_name + "//bias_0.bin" );
		::input_bias( weight23 , hidden3_node , hidden2_node , dir_name + "//bias_1.bin" );
		::input_bias( weight34 , hidden4_node , hidden3_node , dir_name + "//vbias_1.bin" );
		::input_bias( weight45 , output_node , hidden4_node , dir_name + "//vbias_0.bin" );
	}



};

/*
* ネットワークを用いたtester
*/

class tester_decode_procrutes : public tester_procrutes
{
protected:
	network_parameter* net_param;
public:
	tester_decode_procrutes( network_parameter* _net_param , procrustes_parameter *_pparameter , normal_param _np )
		: tester_procrutes( _pparameter , _np )
		, net_param( _net_param )
	{
	}

	virtual float get_mse( data_manager &input , data_manager &output )
	{
		if( input.get_input_node() != output.get_input_node() )
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "データの個数が正しくありません" << std::endl;
		}

		data_manager data = output.copy();
		data_manager d_data = data_manager::create_data( data.get_data_count() , net_param->get_input_node() );

		//ネットワークの伝搬
		FORI( data.get_data_count() )
		{
			net_param->decode( data.get_input_data()[i] , d_data.get_input_data()[i] , false );
		}

		//逆正規化
		d_data.denormalize( 1 , np);

		//プロクラテスブンセキ
		pparameter->denormalize( d_data );

		float mse = ::get_mse( pparameter->get_data().get_input_data() , d_data.get_input_data() , data.get_data_count() , d_data.get_input_node() );

		data.release();
		d_data.release();

		return mse;
	}

};

class tester_procrutes_aam : public tester
{
protected:
	procrustes_parameter_shape *pparameter_shape;
	procrustes_parameter_texture *pparameter_texture;
	network_parameter* shape_model;
	network_parameter* texture_model;
	normal_param np_shape_model;
	normal_param np_texture_model;
	normal_param np;
public:
	tester_procrutes_aam(
		procrustes_parameter_shape *_pparameter_shape,
		procrustes_parameter_texture *_pparameter_texture,
		network_parameter* _shape_model,
		network_parameter* _texture_model,
		normal_param _np_shape_model,
		normal_param _np_texture_model,
		normal_param _np)
		: pparameter_shape(_pparameter_shape)
		, pparameter_texture(_pparameter_texture)
		, shape_model(_shape_model)
		, texture_model(_texture_model)
		, np_shape_model(_np_shape_model)
		, np_texture_model(_np_texture_model)
		, np(_np)
	{
	}
	virtual ~tester_procrutes_aam()
	{
	}

	//フォワード後のデータ元のデータからMSEの算出と表示を行う
	virtual std::string show_mse(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "データ数が正しくありません" << std::endl;
		}

		data_manager data_o = output.copy();

		//aamパラメータの逆正規化
		data_o.denormalize(1, np);

		//aamパラメータの分割
		data_manager shape_param;
		data_manager texture_param;
		data_o.split(shape_model->get_hidden_node_count(), shape_param, texture_param);

		//shape,textureの復元
		data_manager shape = shape_model->decode_all_data(shape_param);
		data_manager texture = texture_model->decode_all_data(texture_param);

		//shape,textureの逆正規化
		shape.denormalize(1, np_shape_model);
		texture.denormalize(1, np_texture_model);

		//プロクラステク分析の逆正規化
		pparameter_shape->denormalize(shape);
		pparameter_texture->denormalize(texture);

		//mseの算出
		float mse_shape = ::get_mse(
			pparameter_shape->get_data().get_input_data(),
			shape.get_input_data(),
			shape.get_data_count(),
			shape.get_input_node());

		float mse_texture = ::get_mse(
			pparameter_texture->get_data().get_input_data(),
			texture.get_input_data(),
			texture.get_data_count(),
			texture.get_input_node());

		//解放
		data_o.release();
		shape_param.release();
		texture_param.release();
		shape.release();
		texture.release();

		char c[256];

#ifdef _WIN32
		sprintf_s(c,"%lf %lf",mse_shape , mse_texture);
#else
		sprintf(c,"%lf %lf",mse_shape , mse_texture);
#endif

		return c;
	}
};

class tester_procrutes_aam_decode : public tester_procrutes_aam
{
protected:
	network_parameter* net_param;
public:
	tester_procrutes_aam_decode(
		procrustes_parameter_shape *_pparameter_shape,
		procrustes_parameter_texture *_pparameter_texture,
		network_parameter* _shape_model,
		network_parameter* _texture_model,
		normal_param _np_shape_model,
		normal_param _np_texture_model,
		normal_param _np,
		network_parameter* _net_param
		)
		: tester_procrutes_aam(
			_pparameter_shape,
			_pparameter_texture,
			_shape_model,
			_texture_model,
			_np_shape_model,
			_np_texture_model,
			_np
		)
		, net_param(_net_param)
	{
	}
	virtual ~tester_procrutes_aam_decode()
	{
	}

	//フォワード後のデータ元のデータからMSEの算出と表示を行う
	virtual std::string show_mse(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "データ数が正しくありません" << std::endl;
		}

		data_manager data = output.copy();
		data_manager data_o = net_param->decode_all_data(data);

		//aamパラメータの逆正規化
		data_o.denormalize(1, np);

		//aamパラメータの分割
		data_manager shape_param;
		data_manager texture_param;
		data_o.split(shape_model->get_hidden_node_count(), shape_param, texture_param);

		//shape,textureの復元
		data_manager shape = shape_model->decode_all_data(shape_param);
		data_manager texture = texture_model->decode_all_data(texture_param);

		//shape,textureの逆正規化
		shape.denormalize(1, np_shape_model);
		texture.denormalize(1, np_texture_model);

		//プロクラステク分析の逆正規化
		pparameter_shape->denormalize(shape);
		pparameter_texture->denormalize(texture);

		//mseの算出
		float mse_shape = ::get_mse(
			pparameter_shape->get_data().get_input_data(),
			shape.get_input_data(),
			shape.get_data_count(),
			shape.get_input_node());

		float mse_texture = ::get_mse(
			pparameter_texture->get_data().get_input_data(),
			texture.get_input_data(),
			texture.get_data_count(),
			texture.get_input_node());

		//解放
		data.release();
		data_o.release();
		shape_param.release();
		texture_param.release();
		shape.release();
		texture.release();

		char c[256];

#ifdef _WIN32
		sprintf_s(c,"%lf %lf",mse_shape , mse_texture);
#else
		sprintf(c, "%lf %lf", mse_shape, mse_texture);
#endif

		return c;
	}
};


//ネットワークの読み込み
//dir_name：ネットワークのフォルダ名
//dir_name+_layer : ネットワークのレイヤー数
//dir_name+_hidden1 : 一つ目の隠れ層
//dir_name+_hidden2 : 二つ目の隠れ層
network_parameter* get_network_parameter(
	std::string dir_name,
	std::map<std::string, std::string> learn_map,
	int input_node,
	float lr = 0,
	float mo = 0,
	float lm = 0)
{
	network_parameter* param = NULL;

	if (get_map_param<int>(learn_map, dir_name + "_layer") == 5)
	{
		param = new network_parameter5(
			input_node,
			get_map_param<int>(learn_map, dir_name + "_hidden1"),
			get_map_param<int>(learn_map, dir_name + "_hidden2"),
			get_map_param<int>(learn_map, dir_name + "_hidden1"),
			input_node,
			lr, mo, lm);
		param->input_weight(get_map_param<std::string>(learn_map, dir_name));

	}
	if (get_map_param<int>(learn_map, dir_name + "_layer") == 3)
	{
		param = new network_parameter3(
			input_node,
			get_map_param<int>(learn_map, dir_name + "_hidden1"),
			input_node,
			lr, mo, lm);
		param->input_weight(get_map_param<std::string>(learn_map, dir_name));

	}

	return param;
}


#endif
