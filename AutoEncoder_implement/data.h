#include "utility.h"
#include <fstream>
#include <assert.h>
#include <vector>

#ifndef DATA_DEFINE
#define DATA_DEFINE



class procrustes_parameter;
class data_manager
{
protected:
	//�f�[�^
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
			np = ::create_normal_param( input_data , data_count , input_node , mode );
		}

		::normalize( input_data , data_count , input_node , np );		
	}
	void normalize( normal_param _np )
	{
		::normalize( input_data , data_count , input_node , _np );
	}
	void denormalize( )
	{
		::denormalize( input_data , data_count , input_node , np );
	}
	void denormalize( normal_param _np )
	{
		::denormalize( input_data , data_count , input_node , _np  );
	}

	data_manager operator+(data_manager& _data)
	{
		data_manager data = data_manager::create_data(data_count, input_node + _data.input_node);

		FORI(data_count)
		{
			FORJ(input_node)
			{
				data.input_data[i][j] = input_data[i][j];
			}
			FORJ(_data.input_node)
			{
				int jj = input_node + j;

				data.input_data[i][jj] = _data.input_data[i][j];
			}
		}

		return data;
	}
};

//AAM�p�����[�^�̓ǂݍ���
//�@���O��train_texture.dat �Ȃ�train�܂ł̎w��ŗǂ�
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
		std::cout << filename_shape << "������܂���" << std::endl;
		exit(1);
	}
	if (!shape_file.is_open())
	{
		std::cout << filename_shape << "������܂���" << std::endl;
		exit(1);
	}

	int count_shape = get_stream4<int>( shape_file );
	int number_shape = get_stream4<int>( shape_file );
	
	int count_texture = get_stream4<int>( texture_file );
	int number_texture = get_stream4<int>( texture_file );

	boundary_count = number_shape;

	if( count_shape != count_texture )
	{
		std::cout << "�f�[�^�̑������قȂ�܂�" << count_shape << " " << count_texture << std::endl;
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

	// shape ����

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

	// shape�̂�

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

data_manager load_data_shape_texture(std::string filename,int& shape_node_count)
{
	data_manager shape_data = load_data_shape(filename);
	data_manager texture_data = load_data_texture(filename);
	if (!shape_data.is_open() || !texture_data.is_open())
	{
		return data_manager();
	}

	shape_node_count = shape_data.get_input_node();

	data_manager output = shape_data + texture_data;

	shape_data.release();
	texture_data.release();

	return output;

}


//PCA�̎听���p�����[�^�̓ǂݍ��݁i�������C�f�[�^���j�Ɗi�[����Ă���̂Ŕ��΂ɓǂݍ���
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
 *	PCA�̃p�����[�^�ɂ�鐳�K���݂����Ȃ���
 */
class pca_parameter
{
protected:
	float** eigen_matrix;	//�ŗL�x�N�g��(restore_data_count,parameter_count)
	float*	mean_vector;	//���ϒl(parameter_count)
	int restore_data_count;	//PCA�ŕ�����̃f�[�^�̎�����
	int parameter_count;	//�听���p�����[�^�̎�����
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
		//�ŗL�x�N�g���̓ǂݍ���
		eigen_matrix = input_matrix( restore_data_count , parameter_count , eigen_file_path );
		if(eigen_matrix==NULL)
		{
			std::cout << eigen_file_path << "�����݂��Ă��܂���" << std::endl;
		}

		//���ϒl�̓ǂݍ���
		int dim = 0;
		mean_vector = input_vector( dim , mean_file_path );
		if(mean_vector==NULL)
		{
			std::cout << mean_file_path << "�����݂��Ă��܂���" << std::endl;
		}

		//�ǂݍ��񂾃f�[�^�̎������̃e�X�g
		if( dim != restore_data_count )
		{
			std::cout << "PCA�t�@�C���̑Ή������Ă��܂��� " << restore_data_count << " " << dim << std::endl;
		}	

		return true;
	}
	data_manager deconvert( data_manager &data )
	{
		float** restore_data = new_array( restore_data_count , data.get_data_count() );
		float** learned_data = new_array( data.get_input_node() , data.get_data_count() );

		//�t�H���[�h��̃f�[�^��]�u
		t_matrix( data.get_input_data() , data.get_data_count() , data.get_input_node() , learned_data );
		
		//�ŗL�x�N�g���Ǝ听���p�����^�[�^�̊|���Z
		mul_matrix( 
				eigen_matrix , restore_data_count , parameter_count,
				learned_data , data.get_input_node() , data.get_data_count(),
				restore_data );

		//���ό`��̑����Z
		plus_horizontal( 
				restore_data , restore_data_count , data.get_data_count(),
				mean_vector , restore_data_count);

		//�]�n���Ă����
		data_manager deconvert_data = data_manager::create_data( data.get_data_count() , restore_data_count );
		t_matrix( 
				restore_data , restore_data_count , data.get_data_count() , 
				deconvert_data.get_input_data() );

		//���
		delete_array( restore_data , restore_data_count );
		delete_array( learned_data , data.get_input_node() );
				
		return deconvert_data;
	}
};

/*
 *	�v���N���X�e�X���͂ɂ���ē��邱�Ƃ��ł������K���p�����[�^
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
	data_manager data_shape;	//�����f�[�^
	data_manager data_texture;	//�����f�[�^
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

	//�f�[�^���ǂݍ��܂�Ă��邩
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
		//�ϊ�
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
		//�ϊ�
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
			std::cout << "���͎�����������������܂���" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		return ::get_mse( input.get_input_data() , output.get_input_data() , input.get_data_count() , input.get_input_node() );	
	}

	virtual std::vector<float> get_mse_each(data_manager &input, data_manager &output) 
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "���͎�����������������܂���" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		return ::get_mse_each(input.get_input_data(), output.get_input_data(), input.get_data_count(), input.get_input_node());
	}

	//�t�H���[�h��̃f�[�^���̃f�[�^����MSE�̎Z�o�ƕ\�����s��
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

	//�t�H���[�h��̃f�[�^���̃f�[�^����MSE�̎Z�o�ƕ\�����s��
	virtual std::vector<std::string> show_mse_each(data_manager &input, data_manager &output )
	{

		std::vector<float> value = get_mse_each(input, output);
		std::vector<std::string> mse_texts;

		char c[256];

		for (size_t i = 0; i < value.size(); i++)
		{
#ifdef _WIN32
			sprintf_s(c, "%f", value.at(i) );
#else
			sprintf(c, "%f", value.at(i) );
#endif
			mse_texts.push_back(c);
		}

		return mse_texts;
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
			std::cout << "���͎�����������������܂���" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data_o = output.copy();

		data_o.denormalize(np);

		pparameter->denormalize( data_o );

		float mse = ::get_mse( pparameter->get_data().get_input_data() , data_o.get_input_data() , data_o.get_data_count() , data_o.get_input_node() );

		data_o.release();

		return mse;
	}

	virtual std::vector<float> get_mse_each(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "���͎�����������������܂���" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data_o = output.copy();

		data_o.denormalize(np);

		pparameter->denormalize(data_o);

		std::vector<float> mse = ::get_mse_each(pparameter->get_data().get_input_data(), data_o.get_input_data(), data_o.get_data_count(), data_o.get_input_node());

		data_o.release();

		return mse;
	}

};

class tester_procrutes_sep : public tester
{
protected:
	procrustes_parameter *pparameterS;
	procrustes_parameter *pparameterT;
	normal_param np;
	int m_sep_count;
public:
	tester_procrutes_sep(procrustes_parameter *_pparameterS, procrustes_parameter *_pparameterT,int sep_count, normal_param _np)
		: pparameterS(_pparameterS)
		, pparameterT(_pparameterT)
		, m_sep_count(sep_count)
		, np(_np)
	{
	}
	virtual ~tester_procrutes_sep()
	{
		delete pparameterS;
		delete pparameterT;
	}

	virtual std::string show_mse(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "���͎�����������������܂���" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data_o = output.copy();

		data_o.denormalize( np);

		data_manager shape_data;
		data_manager texture_data;

		data_o.split(m_sep_count, shape_data, texture_data);

		pparameterS->denormalize(shape_data);
		pparameterT->denormalize(texture_data);

		float mse_shape = ::get_mse(pparameterS->get_data().get_input_data(), shape_data.get_input_data(), shape_data.get_data_count(), shape_data.get_input_node());
		float mse_texture = ::get_mse(pparameterT->get_data().get_input_data(), texture_data.get_input_data(), texture_data.get_data_count(), texture_data.get_input_node());

		data_o.release();
		shape_data.release();
		texture_data.release();

		char c[256];

#ifdef _WIN32
		sprintf_s(c, "%lf %lf", mse_shape, mse_texture);
#else
		sprintf(c, "%lf %lf", mse_shape, mse_texture);
#endif

		return c;
	}

	virtual std::vector<std::string> show_mse_each(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "���͎�����������������܂���" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data_o = output.copy();

		data_o.denormalize( np);

		data_manager shape_data;
		data_manager texture_data;

		data_o.split(m_sep_count, shape_data, texture_data);

		pparameterS->denormalize(shape_data);
		pparameterT->denormalize(texture_data);

		std::vector<float> mse_shape = ::get_mse_each(pparameterS->get_data().get_input_data(), shape_data.get_input_data(), shape_data.get_data_count(), shape_data.get_input_node());
		std::vector<float> mse_texture = ::get_mse_each(pparameterT->get_data().get_input_data(), texture_data.get_input_data(), texture_data.get_data_count(), texture_data.get_input_node());

		data_o.release();
		shape_data.release();
		texture_data.release();

		std::vector<std::string> mse_text;

		char c[256];

		for (size_t i = 0; i < mse_shape.size(); i++)
		{
#ifdef _WIN32
			sprintf_s(c, "%lf %lf", mse_shape.at(i), mse_texture.at(i));
#else
			sprintf(c, "%lf %lf", mse_shape.at(i), mse_texture.at(i));
#endif
			mse_text.push_back(c);
		}

		return mse_text;
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
			std::cout << "���͎�����������������܂���" << std::endl;
		}
		
		if( input.get_data_count() != output.get_data_count() )
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data = output.copy();

		//0.1-0.9 �t���K��
		data.denormalize(np);

		//PCA�ɂ��ϊ�
		data_manager pca_data = m_pca_parameter->deconvert( data );

		//�v���N���X�e�X���͂ɂ�鐳�K��
		pparameter->denormalize( pca_data );

		//�덷�̌v�Z
		float mse = ::get_mse( pca_data.get_input_data() , pparameter->get_data().get_input_data(), pca_data.get_data_count() , pca_data.get_input_node() );

		//���
		data.release();
		pca_data.release();

		return mse;
	}

	virtual std::vector<float> get_mse_each(data_manager &input, data_manager &output)
	{
		if (input.get_input_node() != output.get_input_node())
		{
			std::cout << "���͎�����������������܂���" << std::endl;
		}

		if (input.get_data_count() != output.get_data_count())
		{
			std::cout << "�f�[�^��������������܂���" << std::endl;
		}

		data_manager data = output.copy();

		//0.1-0.9 �t���K��
		data.denormalize( np);

		//PCA�ɂ��ϊ�
		data_manager pca_data = m_pca_parameter->deconvert(data);

		//�v���N���X�e�X���͂ɂ�鐳�K��
		pparameter->denormalize(pca_data);

		//�덷�̌v�Z
		std::vector<float> mse = ::get_mse_each(pca_data.get_input_data(), pparameter->get_data().get_input_data(), pca_data.get_data_count(), pca_data.get_input_node());

		//���
		data.release();
		pca_data.release();

		return mse;
	}

};

#endif
