#include "utility.h"
#include <fstream>
#include <assert.h>

#ifndef DATA_DEFINE
#define DATA_DEFINE



class procrustes_parameter;
class data_manager
{
protected:
	//データ
	float** input_data;
	int data_count;
	int input_node;
	normal_param np;

public:
	data_manager()
		: input_data()
		, input_node()
		, data_count()
		, np()
	{
	}
	~data_manager()
	{
	}
public:
	bool is_open()
	{
		return input_data != 0;
	}
	float** get_input_data()
	{
		return input_data;
	}
	int get_data_count()
	{
		return data_count;
	}
	int get_input_node()
	{
		return input_node;
	}
	normal_param get_normal_param()
	{
		return np;
	}
public:
	void release()
	{
		delete_array( input_data , data_count );
		delete_normal_param( np );
	}
public:
	static data_manager create_data( int dc , int in )
	{
		data_manager dm;
		dm.data_count = dc;
		dm.input_node = in;
		dm.input_data = new_array( dc , in );
		return dm;
	}
public:
	data_manager copy()
	{
		data_manager data = data_manager::create_data( data_count , input_node );
		copy_array( input_data , data.get_input_data() , data_count , input_node );
		return data;
	}
	void split(int split, data_manager &data1, data_manager &data2)
	{
		assert(input_node > split);

		data1 = create_data(data_count, split);
		data2 = create_data(data_count, input_node - split);

		FORI(data_count)
		{
			FORJ(split)
			{
				data1.get_input_data()[i][j] = input_data[i][j];
			}
			FORJ(input_node-split)
			{
				data2.get_input_data()[i][j] = input_data[i][split+j];
			}
		}
	}

	void normalize( int mode )
	{
		if( !np.is_active() )
		{
			np = ::create_normal_param( input_data , data_count , input_node );
		}

		::normalize( input_data , data_count , input_node , np , mode );		
	}
	void normalize( int mode , normal_param _np )
	{
		::normalize( input_data , data_count , input_node , _np , mode );
	}
	void denormalize( int mode )
	{
		::denormalize( input_data , data_count , input_node , np , mode );
	}
	void denormalize( int mode, normal_param _np )
	{
		::denormalize( input_data , data_count , input_node , _np , mode );
	}
};

//AAMパラメータの読み込み
//　名前はtrain_texture.dat ならtrainまでの指定で良い
data_manager load_data_aam( std::string filename , int &boundary_count )
{
	std::string filename_shape = filename + "_shape_aam.dat";
	std::string filename_texture = filename + "_texture_aam.dat";

	std::ifstream shape_file;
	std::ifstream texture_file;
	shape_file.open( filename_shape.c_str() , std::ios::in | std::ios::binary );
	texture_file.open( filename_texture.c_str() , std::ios::in | std::ios::binary );
	if (!shape_file.is_open())
	{
		std::cout << filename_shape << "がつかりません" << std::endl;
		exit(1);
	}
	if (!shape_file.is_open())
	{
		std::cout << filename_shape << "がつかりません" << std::endl;
		exit(1);
	}

	int count_shape = get_stream4<int>( shape_file );
	int number_shape = get_stream4<int>( shape_file );
	
	int count_texture = get_stream4<int>( texture_file );
	int number_texture = get_stream4<int>( texture_file );

	boundary_count = number_shape;

	if( count_shape != count_texture )
	{
		std::cout << "データの総数が異なります" << count_shape << " " << count_texture << std::endl;
	}

	data_manager input_data = data_manager::create_data( count_shape , number_shape + number_texture );

	FORI( count_shape )
	{
		std::cout << "(" << i << "/" << count_shape << ")\r";

		FORJ( number_shape )
		{
			input_data.get_input_data()[i][j] = get_stream4<float>( shape_file );
		}

		FORJ( number_texture )
		{
			input_data.get_input_data()[i][number_shape+j] = get_stream4<float>( texture_file );
		}
	}

	shape_file.close();
	texture_file.close();

	return input_data;
}

data_manager load_data_texture( std::string filename  )
{
	std::cout << "load:" << filename << std::endl;

	std::ifstream ifs;
	ifs.open( filename.c_str() , std::ios::in | std::ios::binary );
	if( !ifs.is_open() )
	{
		return data_manager();
	}

	// shape 無視

	int count = get_stream4<int>( ifs );
	int num = get_stream4<int>( ifs );

	std::cout << count << " " << num << std::endl;

	for( int i = 0; i < count ; i++ )
	{
		for( int j = 0; j < num ; j++ )
		{
			float value = get_stream4<float>( ifs );
		}
	}

	// texture

	count = get_stream4<int>( ifs );
	num = get_stream4<int>( ifs );

	std::cout << count << " " << num << std::endl;

	data_manager input_data = data_manager::create_data( count , num );

	for( int i = 0; i < count ; i++ )
	{
		std::cout << "(" << i << "/" << count << ")\r";

		for( int j = 0; j < num ; j++ )
		{
			float value = get_stream4<float>( ifs );	
			input_data.get_input_data()[i][j] = value;
		}
	}
	std::cout << std::endl;

	ifs.close();

	return input_data;
}

data_manager load_data_shape( std::string filename  )
{
	std::cout << "load:" << filename << std::endl;

	std::ifstream ifs;
	ifs.open( filename.c_str() , std::ios::in | std::ios::binary );
	if( !ifs.is_open() )
	{
		return data_manager();
	}

	// shapeのみ

	int count = get_stream4<int>( ifs );
	int num = get_stream4<int>( ifs );

	std::cout << count << " " << num << std::endl;

	data_manager input_data = data_manager::create_data( count , num );

	for( int i = 0; i < count ; i++ )
	{
		for( int j = 0; j < num ; j++ )
		{
			float value = get_stream4<float>( ifs );
			input_data.get_input_data()[i][j] = value;
		}
	}

	ifs.close();

	return input_data;
}

//PCAの主成分パラメータの読み込み（次元数，データ数）と格納されているので反対に読み込む
data_manager load_data_pca_parameter( std::string filename  )
{
	std::cout << "load:" << filename << std::endl;

	std::ifstream ifs;
	ifs.open( filename.c_str() , std::ios::in | std::ios::binary );
	if( !ifs.is_open() )
	{
		return data_manager();
	}

	int num = get_stream4<int>( ifs );
	int count = get_stream4<int>( ifs );

	std::cout << "pca:" << count << " " << num << std::endl;

	data_manager input_data = data_manager::create_data( count , num );

	for( int i = 0; i < num ; i++ )
	{
		for( int j = 0; j < count ; j++ )
		{
			float value = get_stream4<float>( ifs );
			input_data.get_input_data()[j][i] = value;
		}
	}

	ifs.close();

	return input_data;
}


/*
 *	PCAのパラメータによる正規化みたいなもの
 */
class pca_parameter
{
protected:
	float** eigen_matrix;	//固有ベクトル(restore_data_count,parameter_count)
	float*	mean_vector;	//平均値(parameter_count)
	int restore_data_count;	//PCAで復元後のデータの次元数
	int parameter_count;	//主成分パラメータの次元数
public:
	pca_parameter()
		: eigen_matrix()
		, mean_vector()
	{
	}
	~pca_parameter()
	{
	}
public:
	bool load_pca_parameter( std::string eigen_file_path , std::string mean_file_path )
	{
		//固有ベクトルの読み込み
		eigen_matrix = input_matrix( restore_data_count , parameter_count , eigen_file_path );
		if(eigen_matrix==NULL)
		{
			std::cout << eigen_file_path << "が存在していません" << std::endl;
		}

		//平均値の読み込み
		int dim = 0;
		mean_vector = input_vector( dim , mean_file_path );
		if(mean_vector==NULL)
		{
			std::cout << mean_file_path << "が存在していません" << std::endl;
		}

		//読み込んだデータの次元数のテスト
		if( dim != restore_data_count )
		{
			std::cout << "PCAファイルの対応が取れていません " << restore_data_count << " " << dim << std::endl;
		}	

		return true;
	}
	data_manager deconvert( data_manager &data )
	{
		float** restore_data = new_array( restore_data_count , data.get_data_count() );
		float** learned_data = new_array( data.get_input_node() , data.get_data_count() );

		//フォワード後のデータを転置
		t_matrix( data.get_input_data() , data.get_data_count() , data.get_input_node() , learned_data );
		
		//固有ベクトルと主成分パラメタータの掛け算
		mul_matrix( 
				eigen_matrix , restore_data_count , parameter_count,
				learned_data , data.get_input_node() , data.get_data_count(),
				restore_data );

		//平均形状の足し算
		plus_horizontal( 
				restore_data , restore_data_count , data.get_data_count(),
				mean_vector , restore_data_count);

		//転地していれる
		data_manager deconvert_data = data_manager::create_data( data.get_data_count() , restore_data_count );
		t_matrix( 
				restore_data , restore_data_count , data.get_data_count() , 
				deconvert_data.get_input_data() );

		//解放
		delete_array( restore_data , restore_data_count );
		delete_array( learned_data , data.get_input_node() );
				
		return deconvert_data;
	}
};

/*
 *	プロクラステス分析によって得ることができた正規化パラメータ
 */
class procrustes_parameter
{
protected:
	float* sx;
	float* sy;
	float* tx;
	float* ty;
	float* alpha;
	float* beta;
	data_manager data_shape;	//正解データ
	data_manager data_texture;	//正解データ
public:
	procrustes_parameter()
		: sx(0)
		, sy(0)
		, tx(0)
		, ty(0)
		, alpha(0)
		, beta(0)
	{
	}
	~procrustes_parameter()
	{
		release();
	}
public:

	data_manager get_origin_shape()
	{
		return data_shape;
	}
	data_manager get_origin_texture()
	{
		return data_texture;
	}

	//データが読み込まれているか
	bool isActive()
	{
		if( sx == 0 )return false;
		if( sy == 0 )return false;
		if( tx == 0 )return false;
		if( ty == 0 )return false;
		if( alpha == 0 )return false;
		if( beta == 0 )return false;
		return true;
	}
	
	bool load_data( std::string filename )
	{
		std::cout << "load_procrustes_parameter:" << filename << std::endl;

		std::ifstream ifs;
		ifs.open( filename.c_str() , std::ios::in | std::ios::binary );
		if( !ifs.is_open() )
		{
			return false;
		}

		int count = get_stream4<int>( ifs );
		int snum = get_stream4<int>( ifs );
		int tnum = get_stream4<int>( ifs );

		sx = new float[count];
		sy = new float[count];
		tx = new float[count];
		ty = new float[count];
		alpha = new float[count];
		beta = new float[count];

		std::cout << count << " " << snum << " " << tnum << std::endl;

		data_shape = data_manager::create_data( count , snum );
		data_texture = data_manager::create_data( count , tnum );

		for( int i = 0; i < count ; i++ )
		{
			for( int j = 0; j < snum/2 ; j++ )
			{
				float valuex = static_cast<float>( get_stream4<int>( ifs ) );
				float valuey = static_cast<float>( get_stream4<int>( ifs ) );
				data_shape.get_input_data()[i][j] = valuex;
				data_shape.get_input_data()[i][j+snum/2] = valuey;
			}
			
			for( int j = 0; j < tnum ; j++ )
			{
				float value = get_stream4<float>( ifs );
				data_texture.get_input_data()[i][j] = value;
			}

			tx[i] = get_stream4<float>( ifs );
			ty[i] = get_stream4<float>( ifs );
			sx[i] = get_stream4<float>( ifs );
			sy[i] = get_stream4<float>( ifs );
			alpha[i] = get_stream4<float>( ifs );
			beta[i] = get_stream4<float>( ifs );
		}
		
		ifs.close();

		return true;
	}

	virtual void denormalize( data_manager ) = 0;
	virtual data_manager get_data() = 0;

	void release()
	{
		data_shape.release();
		data_texture.release();

		delete_value( sx );
		delete_value( sy );
		delete_value( tx );
		delete_value( ty );
		delete_value( alpha );
		delete_value( beta );
	}
};

class procrustes_parameter_shape : public procrustes_parameter
{
public:
	procrustes_parameter_shape()
		: procrustes_parameter()
	{
	}
	procrustes_parameter_shape(std::string str)
		: procrustes_parameter()
	{
		load_data(str);
	}
public:
	virtual data_manager get_data()
	{
		return data_shape;
	}
	virtual void denormalize( data_manager data )
	{
		//変換
		for( int i = 0 ;i < data_shape.get_data_count() ; i++ )
		{
			for( int j = 0 , k = data_shape.get_input_node()/2 ; j < data_shape.get_input_node()/2 ; j++, k++ )
			{
				float ox = data.get_input_data()[i][j];
				float oy = data.get_input_data()[i][k];
				data.get_input_data()[i][j] = sx[i] * ox + sy[i] * oy + tx[i];
				data.get_input_data()[i][k] = -sy[i] * ox + sx[i] * oy + ty[i];
			}
		}
	}
};

class procrustes_parameter_texture : public procrustes_parameter
{
public:
	procrustes_parameter_texture()
		: procrustes_parameter()
	{
	}
	procrustes_parameter_texture(std::string str)
		: procrustes_parameter()
	{
		load_data(str);
	}
public:
	virtual data_manager get_data()
	{
		return data_texture;
	}
	virtual void denormalize( data_manager data )
	{
		//変換
		for( int i = 0 ;i < data_texture.get_data_count() ; i++ )
		{
			for( int j = 0 ; j < data_texture.get_input_node() ; j++ )
			{
				data.get_input_data()[i][j] = data.get_input_data()[i][j] * alpha[i] + beta[i];
			}
		}
	}
};

class tester
{
public:
	virtual float get_mse( data_manager &input , data_manager &output )
	{															
		if( input.get_input_node() != output.get_input_node() )
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "データ数が正しくありません" << std::endl;
		}

		return ::get_mse( input.get_input_data() , output.get_input_data() , input.get_data_count() , input.get_input_node() );	
	}

	//フォワード後のデータ元のデータからMSEの算出と表示を行う
	virtual std::string show_mse(data_manager &input, data_manager &output)
	{
		char c[256];

#ifdef _WIN32
		sprintf_s(c,"%f",get_mse(input,output));
#else
		sprintf(c,"%f",get_mse(input,output));
#endif

		return c;
	}
};

class tester_procrutes : public tester
{
protected:
	procrustes_parameter *pparameter;
	normal_param np;
public:
	tester_procrutes( procrustes_parameter *_pparameter , normal_param _np )
		: pparameter( _pparameter )
		, np(_np)
	{
	}
	virtual ~tester_procrutes()
	{
		delete pparameter;
	}

	virtual float get_mse( data_manager &input , data_manager &output )
	{
		if( input.get_input_node() != output.get_input_node() )
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "データ数が正しくありません" << std::endl;
		}

		data_manager data_o = output.copy();

		data_o.denormalize(1,np);

		pparameter->denormalize( data_o );

		float mse = ::get_mse( pparameter->get_data().get_input_data() , data_o.get_input_data() , data_o.get_data_count() , data_o.get_input_node() );

		data_o.release();

		return mse;
	}

};

class tester_pca_procrutes : public tester_procrutes
{
protected:
	pca_parameter *m_pca_parameter;
public:
	tester_pca_procrutes( pca_parameter* _pca_parameter , procrustes_parameter *_pparameter , normal_param _np )
		: tester_procrutes( _pparameter , _np )
		, m_pca_parameter( _pca_parameter )
	{
	}
	virtual ~tester_pca_procrutes()
	{
		delete m_pca_parameter;
	}

	virtual float get_mse( data_manager &input , data_manager &output )
	{
		if( input.get_input_node() != output.get_input_node() )
		{
			std::cout << "入力次元数が正しくありません" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "データ数が正しくありません" << std::endl;
		}

		data_manager data = output.copy();

		//0.1-0.9 逆正規化
		data.denormalize(1,np);

		//PCAによる変換
		data_manager pca_data = m_pca_parameter->deconvert( data );

		//プロクラステス分析による正規化
		pparameter->denormalize( pca_data );

		//誤差の計算
		float mse = ::get_mse( pca_data.get_input_data() , pparameter->get_data().get_input_data(), pca_data.get_data_count() , pca_data.get_input_node() );

		//解放
		data.release();
		pca_data.release();

		return mse;
	}

};

#endif
