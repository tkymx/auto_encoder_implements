#define _USE_MATH_DEFINES
#include<math.h>
#include<string>
#include<fstream> 
#include<iostream>

#ifndef UTILITY
#define UTILITY

#define FORI(x) for( int i = 0;i < x ;i++ )
#define FORJ(x) for( int j=0;j<x;j++ )
#define FORK(x) for( int k =0;k < x ; k++ )
#define FOR(x,v) for( int v=0;v<x;v++)

#ifndef _WIN32
#define SWAP_ENDIAN(val) ((int) ( \
	(((val) & 0x000000ff) << 24) | \
	(((val) & 0x0000ff00) <<  8) | \
	(((val) & 0x00ff0000) >>  8) | \
	(((val) & 0xff000000) >> 24) ))
#else
#define SWAP_ENDIAN(val) ((int)(val))
#endif


/*!
* 空白(スペース，タブ)を削除
* @param[inout] buf 処理文字列
*/
inline std::string DeleteSpace(std::string buf)
{
	size_t pos;
	while ((pos = buf.find_first_of(" 　\t")) != std::string::npos){
		buf.erase(pos, 1);
	}

	return buf;
}

/**
 * 乱数の生成
 */
float rand_range( float min , float max )
{
	float range = max - min;
	float value = min + ((float)(rand()-1)/(float)RAND_MAX) * range;

//	if( value >= max )std::cout << "max" << std::endl;

	return value;
}

template< class T>
T get_stream4( std::ifstream& stream )
{
	int value;
	T rvalue;
	
	stream.read( (char*)&value , 4 );

	value = SWAP_ENDIAN(value);	
	memcpy( &rvalue , &value , sizeof(T) );

	return rvalue;
}

/**
 * 解放処理
 */
void delete_value( float* data )
{
	delete[] data;
}

/**
 * 解放処理（二次元）
 */
void delete_array( float** data , int first_count )
{
	for( int i = 0; i < first_count ; i++ )
	{
		delete[] data[i];
	}
	delete[] data;
}

/**
 * 配列の作成
 */
float* new_array( int first )
{
	float *value = new float[first];
	return value;
}

/**
 * 配列の作成
 */
float** new_array( int first , int second )
{
	float **value = new float*[first];
	for( int i = 0; i < first ; i++ )
	{
		value[i] = new float[second];
	}	
	return value;
}

/**
 * 	配列の表示(二次元)
 */
void show_array( float** data , int first , int second , std::string name )
{
	std::cout << name << std::endl;
	for( int i = 0; i < first ; i++ )
	{
		for( int j = 0 ; j < second ; j++ )
		{
			std::cout << data[i][j] << " ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
}

/**
 * 	配列の表示
 */
void show_array( float* data , int count , std::string name )
{
	std::cout << name << std::endl;
	for( int i = 0; i < count ; i++ )
	{
		std::cout << data[i] << " ";
	}
	std::cout << std::endl;
}

/**
 *	corrupt
 */
inline float corrupt( float value , float p )
{
	return (static_cast<float>(rand())/static_cast<float>(RAND_MAX)) < p ? 0 : value ; 
}

inline void corrupt( float** data , int data_count , int node , float p )
{
	int i,j;

	#ifdef _OPENMP
		#pragma omp parallel for private(j)
	#endif
	for( i = 0; i < data_count ; i++ )
	{
		for( j = 0;j < node ; j++ )
		{
			data[i][j] = corrupt( data[i][j] , p );
		}
	}
}

/**
 *	noise_normal
 */
inline float noise_normal( float value , float s )
{
	return static_cast<float>( value +  1.0f  / ( sqrt( 2.0f * M_PI * s * s ) ) * exp( -  pow( rand_range(-10,10) , 2 ) / ( 2 * s*s ) ) );
	//return value + ( 1.0 / ( sqrt( 2.0 * M_PI * s * s ) ) ) * exp(-pow( rand_range(-10,10) , 2 ) / ( 2 * s*s )) ;
}

inline void  noise_normal( float** data , int data_count , int node , float p )
{
	int i,j;

	#ifdef _OPENMP
		#pragma omp parallel for private(j)
	#endif
	for( i = 0; i < data_count ; i++ )
	{
		for( j = 0;j < node ; j++ )
		{
			data[i][j] = noise_normal( data[i][j] , p );
		}
	}
}


/**
 * 	sigmoid
 * 	シグモイド関数
 */
float sigmoid( float value )
{
	return static_cast<float>( 1.0 / ( 1.0 + exp(-value) ) );
}

/**
 *	foward
 *	ニューラルネットワークのエンコードを行う
 *	weight : [隠れ層のインデックス][入力層のインデックス+1]
 */
void fowarded( float* input ,float* output , float** weight , int input_node , int output_node , bool isSingle = false )
{
	int i,j;

	if( isSingle )
	{
		for( i = 0; i < output_node ; i++ )
		{
			output[i] = 0;
			for( j = 0 ; j < input_node ; j++ )
			{
				output[i] += input[j] * weight[i][j];
			}
			output[i] = output[i] + weight[i][input_node];
		}
	
	}
	else
	{
		#ifdef _OPENMP
			#pragma omp parallel for private(j)
		#endif
		for( i = 0; i < output_node ; i++ )
		{
			output[i] = 0;
			for( j = 0 ; j < input_node ; j++ )
			{
				output[i] += input[j] * weight[i][j];
			}
			output[i] = output[i] + weight[i][input_node];
		}
	}
}
void foward( float* input ,float* output , float** weight , int input_node , int output_node , bool isSingle = false )
{
	fowarded( input , output , weight , input_node , output_node , isSingle );

	int i;

	#ifdef _OPENMP
		#pragma omp parallel for
	#endif
	for( i = 0; i < output_node ; i++ )
	{
		output[i] =   sigmoid( output[i] );
	}
}

/**
 *	vb 出力側のバイアスの学習
 *	クロスエントロピーのバックプロパゲーションをする
 */
void backpropagate_vb( 
		float* input , float* output , 
		float** w23 , float** w23_store , float** w23_d , 
		float* dlvb , 
		float learning_rate , float lambda , float momentum ,
		int input_node , int middle_node , bool isSingle = false )
{
	//vb
	
	if( isSingle )
	{
		for( int i = 0; i < input_node ; i++ )
		{
			dlvb[i] = input[i] - output[i];
			w23_d[i][middle_node] = learning_rate * dlvb[i] - lambda * w23[i][middle_node] + momentum * w23_d[i][middle_node];
			w23_store[i][middle_node] += w23_d[i][middle_node];
		}	
	}
	else
	{
		#ifdef _OPENMP
			#pragma omp parallel for
		#endif
		for( int i = 0; i < input_node ; i++ )
		{
			dlvb[i] = input[i] - output[i];
			w23_d[i][middle_node] = learning_rate * dlvb[i] - lambda * w23[i][middle_node] + momentum * w23_d[i][middle_node];
			w23_store[i][middle_node] += w23_d[i][middle_node];
		}
	}
}


/**
 * 	hb 隠れ層側のバイアスの学習
 */
void backpropagate_hb(
		float* middle , 
		float** w12 , float** w12_store , float** w12_d , 
		float* dlvb , float* dlhb , 
		float learning_rate , float lambda , float momentum , 
		int input_node , int middle_node , 
	        bool isSingle = false	)
{
	int i,j;

	//hb
	
	if( isSingle )
	{
		for( i = 0; i < middle_node ; i++ )
		{
			dlhb[i] = 0;
			for (j = 0; j < input_node; j++)
			{
				dlhb[i] += w12[i][j] * dlvb[j];
			}
			dlhb[i] *= middle[i] * ( 1 - middle[i] );

			w12_d[i][input_node] = learning_rate * dlhb[i] - lambda * w12[i][input_node] + momentum * w12_d[i][input_node];
			w12_store[i][input_node] += w12_d[i][input_node];
		}
	
	}
	else
	{	
		#ifdef _OPENMP
			#pragma omp parallel for private(j)
		#endif
		for( i = 0; i < middle_node ; i++ )
		{
			dlhb[i] = 0;
			for (j = 0; j < input_node; j++)
			{
				dlhb[i] += w12[i][j] * dlvb[j];
			}
			dlhb[i] *= middle[i] * ( 1 - middle[i] );

			w12_d[i][input_node] = learning_rate * dlhb[i] - lambda * w12[i][input_node] + momentum * w12_d[i][input_node];
			w12_store[i][input_node] += w12_d[i][input_node];
		}
	}
}

/**
 * 	wの学習（w12のみ）
 */
void backpropagate_w( 
		float* input , float* middle , 
		float** w12 , float** w12_store , float** w12_d , 
		float learning_rate , float momentum ,
		float* dlhb , float* dlvb ,  
		float lambda , 
		int input_node , int middle_node , 
	        bool isSingle = false	)
{
	float dlhbi=0,middlei=0;
	int i,j;
	
	//w
	
	if( isSingle )
	{
		for( i = 0 ;i < middle_node ; i++ )
		{
			dlhbi = dlhb[i];
			middlei = middle[i];
			for( j = 0; j < input_node ; j++ )
			{
				w12_d[i][j] = learning_rate * ( dlhbi * input[j] + dlvb[j] * middlei ) - lambda * w12[i][j] + momentum * w12_d[i][j];
				w12_store[i][j] += w12_d[i][j];
			}
		}
	
	}
	else
	{	

		#ifdef _OPENMP
			#pragma omp parallel for private(dlhbi,middlei,j)
		#endif
		for( i = 0 ;i < middle_node ; i++ )
		{
			dlhbi = dlhb[i];
			middlei = middle[i];
			for( j = 0; j < input_node ; j++ )
			{
				w12_d[i][j] = learning_rate * ( dlhbi * input[j] + dlvb[j] * middlei ) - lambda * w12[i][j] + momentum * w12_d[i][j];
				w12_store[i][j] += w12_d[i][j];
			}
		}
	}
}

/**
 * コピー関連の統合
 */

static clock_t copy_weight_time = 0;
static clock_t copy_weight1_time = 0;
static clock_t copy_weight2_time = 0;
static clock_t copy_bias_time = 0;

void update_weight( 
		float** weight12 , float** weight23 ,
	       	float** weight12_store , float** weight23_store ,
		int input_node , int hidden_node , 
	        bool isSingle = false )
{
	int i,j;

	if( isSingle )
	{
		for( i = 0; i < hidden_node ;i++ )
		{
			for( j=0;j<input_node;j++ )
			{
				weight23_store[j][i] = weight12_store[i][j];
				weight12[i][j] = weight12_store[i][j];
				weight23[j][i] = weight12_store[i][j];
			}
			weight12[i][input_node] = weight12_store[i][input_node];
		}
		for( i = 0; i< input_node ;i++ )
		{
			weight23[i][hidden_node] = weight23_store[i][hidden_node];
		}
	}
	else
	{
#ifdef TIME_TEST		
		clock_t current_time = clock();
#endif
		#ifdef _OPENMP
			#pragma omp parallel for private(j)
		#endif
		for( i = 0; i < hidden_node ;i++ )
		{
			for( j=0;j<input_node;j++ )
			{
				//weight23_store[j][i] = weight12_store[i][j];	//高速化のために消した
				weight12[i][j] = weight12_store[i][j];
			}
		}
	
#ifdef TIME_TEST		
		copy_weight_time += ( clock() - current_time );
	        current_time = clock();	
#endif

		#ifdef _OPENMP
			#pragma omp parallel for private(j)
		#endif
		for( i = 0; i < input_node  ;i++ )
		{
			for( j=0;j<hidden_node;j++ )
			{
				weight23[i][j] = weight12_store[j][i];
			}
		}
	
#ifdef TIME_TEST		
		copy_weight1_time += ( clock() - current_time );
	        current_time = clock();	
#endif

		#ifdef _OPENMP
			#pragma omp parallel for 
		#endif
		for( i = 0; i < hidden_node ;i++ )
		{
			weight12[i][input_node] = weight12_store[i][input_node];
		}

#ifdef TIME_TEST		
		copy_weight2_time += ( clock() - current_time );
	        current_time = clock();	
#endif

		#ifdef _OPENMP
			#pragma omp parallel for
		#endif
		for( i = 0; i< input_node ;i++ )
		{
			weight23[i][hidden_node] = weight23_store[i][hidden_node];
		}

#ifdef TIME_TEST		
		copy_bias_time += ( clock() - current_time );
	        current_time = clock();	
#endif
	}
}

void update_weight_store( float** weight12 , float** weight23 , float** weight12_store , float** weight23_store , int input_node , int hidden_node )
{
	int i,j;

	#ifdef _OPENMP
		#pragma omp parallel for private(j)
	#endif
	for( i = 0; i < hidden_node ;i++ )
	{
		for( j=0;j<input_node;j++ )
		{
			weight23[j][i] = weight12[i][j];
			weight12_store[i][j] = weight12[i][j];
			weight23_store[j][i] = weight12[i][j];
		}
		weight12_store[i][input_node] = weight12[i][input_node];
	}

	#ifdef _OPENMP
		#pragma omp parallel for
	#endif
	for( i = 0; i< input_node ;i++ )
	{
		weight23_store[i][hidden_node] = weight23[i][hidden_node];
	}
}

static clock_t vb_time = 0;
static clock_t hb_time = 0;
static clock_t w_time = 0;
static clock_t copy_time = 0;

/**
 * 	全体の学習
 */
inline void backpropagate_cross_entropy( 
		float* input , float* middle , float* output , 
		float** w12 , float** w23 , float** w12_store , float** w23_store , float** w12_d ,float** w23_d, 
		float learning_rate , float lambda , float momentum , 
		float* dlvb , float* dlhb , 
		int input_node, int middle_node , int output_node ,
	       	bool isSingle = false)
{
#ifdef TIME_TEST
	clock_t current_time = clock();
#endif
	
	backpropagate_vb( 
			input , output , 
			w23 , w23_store, w23_d ,  
			dlvb , 
			learning_rate , lambda , momentum , 
			input_node , middle_node , isSingle );

#ifdef TIME_TEST
	vb_time += ( clock() - current_time );
	current_time = clock();
#endif
	
	backpropagate_hb( 
			middle , 
			w12 , w12_store , w12_d ,
			dlvb , dlhb , 
			learning_rate , lambda , momentum , 
			input_node , middle_node , isSingle );

#ifdef TIME_TEST
	hb_time += ( clock() - current_time );
	current_time = clock();
#endif

	backpropagate_w( 
			input , middle , 
			w12 , w12_store , w12_d , 
			learning_rate , momentum, 
			dlhb , dlvb , 
			lambda , 
			input_node , middle_node , isSingle );

#ifdef TIME_TEST
	w_time += ( clock() - current_time );
	current_time = clock();
#endif

	//copy data
	update_weight( w12 , w23 , w12_store , w23_store , input_node , middle_node , isSingle );
	
#ifdef TIME_TEST
	copy_time += ( clock() - current_time );
	current_time = clock();
#endif
}

void copy_array( float** src , float** dst , int first, int second )
{
	int i,j;

	#ifdef _OPENMP
		#pragma omp parallel for private(j)
	#endif
	for( i = 0; i < first ;i++ )
	{
		for( j=0;j<second;j++ )
		{
			dst[i][j] = src[i][j];
		}
	}
}

inline void backpropagate_mse_last( 
		float* input , float* middle , float* output , 
		float** w23 , float** w23_store ,float** w23_d, 
		float learning_rate , float lambda , float momentum , 
		float* ppde23 , 
		int input_node, int middle_node )
{
	int i,j;
	float value;
	
	//3層目
	//
	#ifdef _OPENMP
		#pragma omp parallel for private(i,value)
	#endif
	for (j = 0; j < input_node; j++)
	{
		value = (input[j] - output[j]); //MSE

		ppde23[j] = value * output[j] * (1 - output[j]);

		for (i = 0; i < middle_node; i++)
		{
			w23_d[j][i] = learning_rate * ppde23[j] * middle[i] - lambda * w23[j][i] + momentum * w23_d[j][i]; 
			w23_store[j][i] += w23_d[j][i];
		}

		w23_d[j][middle_node] = learning_rate * ppde23[j] * 1 - lambda * w23[j][middle_node] + momentum * w23_d[j][middle_node];
		w23_store[j][middle_node] += w23_d[j][middle_node];
	}
}


inline void backpropagate_mse_continue( 
		float* input , float* middle , 
		float** w12 , float** w23 , float** w12_store , float** w12_d , 
		float learning_rate , float lambda , float momentum , 
		float* ppde12 , float* ppde23 , 
		int input_node, int middle_node , int output_node  )
{
	int i,j,k;
	float value;

	//2層目
	
	#ifdef _OPENMP
		#pragma omp parallel for private(i,k,value)
	#endif
	for (j = 0; j < middle_node; j++)
	{
		ppde12[j] = middle[j] * (1 - middle[j]);

		value = 0;

		for (k = 0; k < output_node ; k++)
		{
			value += ppde23[k] * w23[k][j];
		}

		ppde12[j] *= value;

		for (i = 0; i < input_node; i++)
		{
			w12_d[j][i] = learning_rate * ppde12[j] * input[i] - lambda * w12[j][i] + momentum * w12_d[j][i];
			w12_store[j][i] += w12_d[j][i];
		}

		w12_d[j][input_node] = learning_rate * ppde12[j] * 1 - lambda * w12[j][input_node] + momentum * w12_d[j][input_node];
		w12_store[j][input_node] += w12_d[j][input_node];
	}
}

inline void backpropagate_mse( 
		float* input , float* middle , float* output , 
		float** w12 , float** w23 , float** w12_store , float** w23_store , float** w12_d ,float** w23_d, 
		float learning_rate , float lambda , float momentum , 
		float* ppde12 , float* ppde23 , 
		int input_node, int middle_node , int output_node  )
{
	
	backpropagate_mse_last( 
		input , middle , output , 
		w23 , w23_store ,w23_d, 
		learning_rate , lambda , momentum , 
		ppde23 , 
		input_node, middle_node );

	backpropagate_mse_continue( 
		input , middle , 
		w12 , w23 , w12_store , w12_d , 
		learning_rate , lambda , momentum , 
		ppde12 , ppde23 , 
		input_node, middle_node , output_node  );
	
	
	copy_array( w12_store , w12 , middle_node , input_node+1 );
	copy_array( w23_store , w23 , input_node , middle_node+1 );
}


inline void backpropagate_mse_5( 
		float* input , float* middle2 , float* middle3 ,float* middle4  , float* output , 
		float** w12 , float** w23 , float** w34 , float** w45,
	       	float** w12_store , float** w23_store , float** w34_store , float** w45_store,
	       	float** w12_d ,float** w23_d, float** w34_d ,float** w45_d, 
		float learning_rate , float lambda , float momentum , 
		float* ppde12 , float* ppde23 , float* ppde34 , float* ppde45,
		int input_node, int middle2_node , int middle3_node , int middle4_node , int output_node  )
{
	//45
	backpropagate_mse_last( 
		input , middle4 , output , 
		w45 , w45_store ,w45_d, 
		learning_rate , lambda , momentum , 
		ppde45 , 
		output_node, middle4_node );

	//34
	backpropagate_mse_continue( 
		middle3 , middle4 , 
		w34 , w45 , w34_store , w34_d , 
		learning_rate , lambda , momentum , 
		ppde34 , ppde45 , 
		middle3_node, middle4_node , output_node  );

	//23
	backpropagate_mse_continue( 
		middle2 , middle3 , 
		w23 , w34 , w23_store , w23_d , 
		learning_rate , lambda , momentum , 
		ppde23 , ppde34 , 
		middle2_node, middle3_node  , middle4_node );

	//12
	backpropagate_mse_continue( 
		input , middle2 , 
		w12 , w23 , w12_store , w12_d , 
		learning_rate , lambda , momentum , 
		ppde12 , ppde23 , 
		input_node, middle2_node  , middle3_node );

	copy_array( w12_store , w12 , middle2_node , input_node+1 );
	copy_array( w23_store , w23 , middle3_node , middle2_node+1 );
	copy_array( w34_store , w34 , middle4_node , middle3_node+1 );
	copy_array( w45_store , w45 , output_node , middle4_node+1 );
}

/**
 * データの初期化
 */
void init_data( float* data , int count )
{
	for( int i = 0 ;i < count ; i++ )
	{
		data[i] = 0;
	}
}

/**
 * データの初期化（二次元）
 */
void init_data( float** data , int first_count , int second_count )
{
	for( int i = 0 ;i < first_count ; i++ )
	{
		for( int j = 0; j < second_count ; j++ )
		{
			data[i][j] = 0;
		}
	}
}

/**
 * weightの初期化
 */
void init_weight_data( float** data , int first_count , int second_count )
{
	float a = 4 * sqrt(6.0f / (float)(first_count + second_count)) / (float)second_count;

	for( int i = 0 ;i < first_count ; i++ )
	{
		for( int j = 0; j < second_count ; j++ )
		{
			data[i][j] = rand_range( -a , a );
		}
		data[i][second_count] = 0.5;
	}
}  

/**
 * MSEの計算
 */
float get_mse( float** test , float** answer , int data_count , int node )
{
	float value = 0;
	for( int data = 0 ; data < data_count ; data++ )
	{
		for( int i = 0; i < node ; i++ )
		{
			value += ( answer[data][i] - test[data][i] ) * ( answer[data][i] - test[data][i] ); 
		}
	}

	return  sqrt( value / (float)node / (float)data_count );
}

/**
 *	平均を計算する，
 *	axis 軸の次元 0 1 ...
 */
float* create_mean_vector( float** input_data ,int first , int second, int axis  )
{
	float *mean_value = 0;

	if( axis == 0 )
	{
		// i軸を焦点に平均を計算

		mean_value = new_array( first );
		for( int i = 0; i < first ; i++ )
		{
			mean_value[i]=0;
			for( int j = 0;j < second ;j++ )
			{
				mean_value[i] += input_data[i][j];
			}
			mean_value[i] /= (float)second;
		}
	}
	else if( axis == 1 )
	{
		//j軸を焦点に平均を計算

		mean_value = new_array( second );
		for( int i = 0; i < second ; i++ )
		{
			mean_value[i]=0;
			for( int j = 0;j < first ;j++ )
			{
				mean_value[i] += input_data[j][i];
			}
			mean_value[i] /= (float)first;
		}
	}

	return mean_value;
}

/**
 *	標準偏差を計算する，
 *	axis 軸の次元 0 1 ...
 */
float* create_std_vector( float** input_data ,int first , int second, int axis  )
{
	float* std_value = 0;
	float* mean_value = create_mean_vector( input_data , first , second , axis );

	if( axis == 0 )
	{
		std_value = new_array(first);

		for( int i = 0; i < first ; i++ )
		{
			std_value[i] = 0;
			for( int j = 0; j < second ; j++ )
			{
				std_value[i] += ( input_data[i][j] - mean_value[i] ) *  ( input_data[i][j] - mean_value[i] );
			}
			std_value[i] = sqrt( std_value[i]/(float)second );
		}
	}
	else if( axis == 1 )
	{
		std_value = new_array(second);

		for( int i = 0; i < second ; i++ )
		{
			std_value[i] = 0;
			for( int j = 0; j < first ; j++ )
			{
				std_value[i] += ( input_data[j][i] - mean_value[i] ) * ( input_data[j][i] - mean_value[i] );
			}
			std_value[i] = sqrt( std_value[i]/(float)first );
		}
	}
	delete_value( mean_value );	

	return std_value;
}

/**
 *	最大値を求める
 *	axis 軸の次元 0 1 ...
 */
float* create_max_vector( float** input_data ,int first , int second, int axis  )
{
	float* max_value = 0;

	if( axis == 0 )
	{
		max_value = new_array(first);

		for( int i = 0; i < first ; i++ )
		{
			max_value[i] = input_data[i][0];
			for( int j = 1; j < second ; j++ )
			{
				max_value[i] =  max_value[i] < input_data[i][j] ? input_data[i][j] : max_value[i];
			}
		}
	}
	else if( axis == 1 )
	{
		max_value = new_array(second);

		for( int i = 0; i < second ; i++ )
		{
			max_value[i] = input_data[0][i];
			for( int j = 0; j < first ; j++ )
			{
				max_value[i] = max_value[i] < input_data[j][i] ? input_data[j][i] : max_value[i];
			}
		}
	}
	return max_value;
}

/**
 *	最小値を求める
 *	axis 軸の次元 0 1 ...
 */
float* create_min_vector( float** input_data ,int first , int second, int axis  )
{
	float* min_value = 0;

	if( axis == 0 )
	{
		min_value = new_array(first);

		for( int i = 0; i < first ; i++ )
		{
			min_value[i] = input_data[i][0];
			for( int j = 1; j < second ; j++ )
			{
				min_value[i] =  min_value[i] > input_data[i][j] ? input_data[i][j] : min_value[i];
			}
		}
	}
	else if( axis == 1 )
	{
		min_value = new_array(second);

		for( int i = 0; i < second ; i++ )
		{
			min_value[i] = input_data[0][i];
			for( int j = 0; j < first ; j++ )
			{
				min_value[i] = min_value[i] > input_data[j][i] ? input_data[j][i] : min_value[i];
			}
		}
	}
	return min_value;
}



/** 
 * 正規化用のパラメータ
 */
struct normal_param
{
	float *mean_vector;
	float *std_vector;
	int length;
	float *max_vector;
	float *min_vector;
public:
	bool is_active()
	{
		return mean_vector != 0;
	}
};

/**
 *　正規化用パラメータの作成
 */
normal_param create_normal_param(float** input_data , int data_count , int input_count )
{
	normal_param np;
	np.mean_vector = create_mean_vector( input_data , data_count , input_count , 1 );
	np.std_vector = create_std_vector( input_data , data_count , input_count , 1 );
	np.max_vector = create_max_vector( input_data , data_count , input_count , 1 );
	np.min_vector = create_min_vector( input_data , data_count , input_count , 1 );
	np.length = input_count;

	return np;
}

/**
 *　正規化用のパラメータの解放
 */
void delete_normal_param( normal_param &np )
{
	if( !np.is_active() )
	{
		return;
	}
	
	delete_value( np.mean_vector );
	delete_value( np.std_vector );
	delete_value( np.max_vector );
	delete_value( np.min_vector );
	np = normal_param();
}

/** 
 *　正規化を行う
 *  mode 0 平均０分散１
 *  mode 1 0.1～0.9
 */
void normalize( float** input_data , int data_count , int input_count , normal_param &np , int mode = 1)
{
	if( !np.is_active() )
	{
		std::cout << "normal_paramがアクティブではありません" << std::endl;	
	}

	if( mode == 0 )
	{
		for( int i = 0; i < data_count ; i++ )
		{
			for( int j = 0; j < input_count ; j++ )
			{
				input_data[i][j] = ( input_data[i][j] - np.mean_vector[j] ) / np.std_vector[j];
			}
		}
	}
	else
	{
		for( int j = 0; j < input_count ; j++ )
		{
			float vMax = np.max_vector[j] - np.mean_vector[j];
			float vMin = np.min_vector[j] - np.mean_vector[j];

			for( int i = 0; i < data_count ; i++ )
			{
				input_data[i][j] = 0.1f + 0.8f * ( ( input_data[i][j] - np.mean_vector[j] ) - vMin ) / ( vMax - vMin );
			}
		}
	}
}

/** 
 *　逆正規化を行う
 *  mode 0 平均０分散１
 *  mode 1 0.1～0.9
 */
void denormalize( float** input_data , int data_count , int input_count , normal_param &np , int mode  )
{
	if( !np.is_active() )
	{
		std::cout << "normal_paramがアクティブではありません" << std::endl;	
	}

	if( mode==0 )
	{
		for( int i = 0; i < data_count ; i++ )
		{
			for( int j = 0; j < input_count ; j++ )
			{
				input_data[i][j] = input_data[i][j] *  np.std_vector[j] + np.mean_vector[j];
			}
		}
	}
	else
	{
		for( int j = 0; j < input_count ; j++ )
		{
			float vMax = np.max_vector[j] - np.mean_vector[j];
			float vMin = np.min_vector[j] - np.mean_vector[j];

			for( int i = 0; i < data_count ; i++ )
			{			
				input_data[i][j] = (  ( vMax - vMin ) * ( input_data[i][j] - 0.1f ) ) / 0.8f + ( np.mean_vector[j] + vMin  );
			}
		}
	}
}

void output_weight( float** weight , int first , int second , std::string filename )
{
	std::ofstream fout;
	fout.open( filename.c_str() , std::ios::out | std::ios::binary | std::ios::trunc );

	//fout << first << " " << second << std::endl;
	
	int first_r = SWAP_ENDIAN( first );
	int second_r = SWAP_ENDIAN( second );

	fout.write( (char*)&first_r , sizeof(int) );
	fout.write( (char*)&second_r , sizeof(int) );

	FORI( first)
	{
		FORJ(second)
		{
			int* value = reinterpret_cast<int*>( &weight[i][j] );
			int value_r = SWAP_ENDIAN( *value );
			fout.write( (char*)&value_r , sizeof(int) );

//			fout << weight[i][j] << " ";
		}
//		fout << std::endl;
	}

	fout.close();
}

float** input_matrix( int &first , int &second , std::string file_path  )
{
	std::cout << "load:" << file_path << std::endl;
	std::ifstream ifs;
	ifs.open( file_path.c_str() , std::ios::in | std::ios::binary );
	if( !ifs.is_open() )
	{
		return NULL;
	}

	first = get_stream4<int>( ifs );
       	second	= get_stream4<int>( ifs );

	std::cout << "input_matrix:" << first << " " << second << std::endl;

	float** data = new_array( first , second );

	FORI( first )
	{
		FORJ( second )
		{
			data[i][j] = get_stream4<float>(ifs);
		}
	}

	ifs.close();

	return data;
}

float* input_vector( int &first , std::string file_path  )
{
	std::cout << "load:" << file_path << std::endl;
	std::ifstream ifs;
	ifs.open( file_path.c_str() , std::ios::in | std::ios::binary );
	if( !ifs.is_open() )
	{
		return NULL;
	}

	first = get_stream4<int>( ifs );
	float* data = new float[ first ];

	std::cout << "input_vector:" << first  << std::endl;

	
	FORI( first )
	{	
		data[i] = get_stream4<float>(ifs);
	}

	ifs.close();

	return data;
}


void input_weight( float** weight , int first , int second , std::string filename )
{
	std::ifstream fin;
	fin.open( filename.c_str() , std::ios::in | std::ios::binary );
	if( !fin.is_open() )
	{
		std::cout << filename << "が存在していません" << std::endl;
		return;
	}
	
	int first_ = get_stream4<int>(fin);
	int second_ = get_stream4<int>(fin);

	//std::cout << first << " " << second << std::endl;

	if( first_ != first || second_ != second )
	{
		std::cout << first << ":" << first_ << " " << second << ":" << second_ << "err" << std::endl;
		return;
	}

	FORI( first)
	{
		FORJ(second)
		{
			weight[i][j] = get_stream4<float>(fin);
		}
	}

	fin.close();
}



void output_bias( float** weight , int first , int second , std::string filename )
{
	std::ofstream fout;
	fout.open( filename.c_str() , std::ios::out | std::ios::binary | std::ios::trunc );

	int first_r = SWAP_ENDIAN( first );
	fout.write( (char*)&first_r , sizeof(int) );

	FORI( first)
	{
		int* value = reinterpret_cast<int*>( &weight[i][second] );
		int value_r = SWAP_ENDIAN( *value );
		fout.write( (char*)&value_r , sizeof(int) );	
	}

	fout.close();

}

void input_bias( float** weight , int first , int second , std::string filename )
{
	std::ifstream fin;
	fin.open( filename.c_str() , std::ios::in | std::ios::binary );
	if( !fin.is_open() )
	{
		std::cout << filename << "が存在していません" << std::endl;
		return;
	}
	
	int first_ = get_stream4<int>(fin);

	if( first_ != first )
	{
		std::cout << first << ":" << first_ << std::endl;
		return;
	}

	FORI( first)
	{
		weight[i][second] = get_stream4<float>(fin);
	}

	fin.close();
}

void mul_matrix( float** in , int ifirst , int isecond , float** mat , int mfirst , int msecond , float** out )
{
	if( isecond != mfirst )
	{
		std::cout << "mul:次元数が一致しません" << isecond << " " << mfirst << std::endl;
	}

	int i,j,k;

	#ifdef _OPENMP
		#pragma omp parallel for private(j,k)
	#endif
	for( i = 0 ;i < ifirst ; i++ )
	{
		for( j = 0; j < msecond ; j++ )
		{
			out[i][j] = 0;
			for( k = 0; k < isecond ;k++ )
			{
				out[i][j] += in[i][k] * mat[k][j];
			}
		}
	}

}

//水平方向に行列とベクトルの引き算をする
void minus_horizontal( float** inout , int ifirst , int isecond , float* vec , int mfirst )
{
	if( ifirst != mfirst )
	{
		std::cout << "minus:次元数が一致しません" << ifirst << " " << mfirst << std::endl;
		exit(1);
	}

	FORI( isecond )
	{
		FORJ( ifirst )
		{
			inout[j][i] = inout[j][i] - vec[j];
		}
	}
}

//水平方向に行列とベクトルの足し算をする
void plus_horizontal( float** inout , int ifirst , int isecond , float* vec , int mfirst )
{
	if( ifirst != mfirst )
	{
		std::cout << "plus:次元数が一致しません" << ifirst << " " << mfirst << std::endl;
		exit(1);
	}

	FORI( isecond )
	{
		FORJ( ifirst )
		{
			inout[j][i] = inout[j][i] + vec[j];
		}
	}
}


//転地していれる
void t_matrix( float** in , int ifirst , int isecond , float** out )
{
	FORI( ifirst )
	{
		FORJ(isecond)
		{
			out[j][i] = in[i][j];
		}
	}
}

//////////////////////////////////test class//////////////////////////////////

/*

class test_backpropagate_vb : public test_program
{
public:
	virtual void declaretion()
	{
	}
	
	virtual void init()
	{
	}
	
	virtual void change_init()
	{
	}
	
	virtual void release()
	{
	}
	
	virtual void show_param()
	{
	}
	
	virtual void process()
	{
	}
}
   
 */

#define INPUT_TEST_2 10
#define HIDDEN_TEST_2 20

class test_program
{
public:
	test_program()
	{
	}
	~test_program()
	{
	}
public:
	virtual void declaretion() = 0;
	virtual void init() = 0;
	virtual void change_init() = 0;
	virtual void release() = 0;
	virtual void show_param() = 0;
	virtual void process() = 0;
	virtual void process_single() = 0;

public:
	void test()
	{
		std::cout << "declaretion" << std::endl;

		declaretion();

		std::cout << "init" << std::endl;

		init();		

		std::cout << "pararel" << std::endl;

		for( int e = 0; e < 1 ; e++ )
		{
			std::cout << "//////////////////test:" << e << std::endl;

			change_init();

			show_param();

			process();

			show_param();
		
		}

		std::cout << "origin" << std::endl;

		change_init();

		show_param();

		process_single();

		show_param();

		std::cout << "release" << std::endl;

		release();	
	
	}
};



#endif

