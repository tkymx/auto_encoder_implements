#include<iostream>
#include"utility.h"

//#include"test_code/backpropagate_vb.h"
//#include"test_code/backpropagate_hb.h"
//#include"test_code/backpropagate_w.h"
//#include"test_code/backpropagate_cross_entropy.h"
//#include"test_code/backpropagate_foward_cross_entropy.h"
//#include"test_code/backpropagate_all.h"
//#include"test_code/procrustes_parameter.h"
//#include"test_code/input_weight.h"
//#include "test_code/own_math.h"
#include "test_code/network_test.h"

#define TEST_INPUT_COUNT 3
#define TEST_HIDDEN_COUNT 2

//void test1();

int main()
{
	try
	{
		test_network _test;
		_test.test();
	}
	catch( const char* str )
	{
		std::cout << str << std::endl;
	}
}

#if 0

void test1()
{	
	std::cout << "hello test" << std::endl;

	std::cout << "foward" << std::endl;

	float* input_data = new float[TEST_INPUT_COUNT];
	float* hidden_data = new float[TEST_HIDDEN_COUNT];
	float* output_data = new float[TEST_INPUT_COUNT];

	float** weight12 = new_array( TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 );
	float** weight23 = new_array( TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 );

	float** weight12_store = new_array( TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1);
	float** weight23_store = new_array( TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1); 

	float* dlhb = new float[TEST_HIDDEN_COUNT];
	float* dlvb = new float[TEST_INPUT_COUNT];

	input_data[0] = 1;
	input_data[1] = 2;
	input_data[2] = 3;

	weight12[0][0] = 1;
	weight12[0][1] = 2;
	weight12[0][2] = 3;
	weight12[0][3] = 4;
	weight12[1][0] = 5;
	weight12[1][1] = 6;
	weight12[1][2] = 7;
	weight12[1][3] = 8;

	init_data( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 );

	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23" );

	// 1 2 3
	show_array( input_data , TEST_INPUT_COUNT , "1:array" );

	//1 2 3 4
	//5 6 7 8
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "2:array" ); 

	fowarded( input_data , hidden_data , weight12 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );

	//18,46
	show_array( hidden_data , TEST_HIDDEN_COUNT , "3:array" );

	update_weight_store(weight12 , weight23 , weight12_store , weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT);

	weight23[0][2] = 1;
	weight23[1][2] = 2;
	weight23[2][2] = 3;

	//1 5 1
	//2 6 2
	//3 7 3
	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1  , "weight23" );

	fowarded( hidden_data , output_data , weight23 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT );

	//249 314 379
	show_array( output_data , TEST_INPUT_COUNT , "4:array" );

	// 1 2 3 4
	// 5 6 7 8
	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "w12_store_array" );

	//1 5 1
	//2 6 2
	//3 7 3
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "w23_store_array" );

	backpropagate_vb( input_data , output_data , weight23_store , dlvb , 10 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );

	// 1 5 -2479
	// 2 6 -3118
	// 3 7 -3757
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "w23_store_array" );

	//-248 -312 -376
	show_array( dlvb , TEST_INPUT_COUNT , "dlvb_array" );

	backpropagate_hb( hidden_data , weight12 , weight12_store ,  dlvb , dlhb , 10 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );
	
	//1 2 3 6.12e+06
	//5 6 7 1.18901e+08
	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "w12_store_array" );

	//612000 1.18901e+07
	show_array( dlhb , TEST_HIDDEN_COUNT , "dlhb_array" );

	float **weight12_d = new_array( TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 );
	init_data( weight12_d , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 );

	backpropagate_w( input_data , hidden_data , weight12 , weight12_store , weight12_d , 10 , 1000 , dlhb , dlvb , 100 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT  );

	show_array( weight12_d , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "weight12_d" );

	//6.07526e+06 1.21836e+07 1.8292e+07 6.12e+06
	//1.18786e+08 2.37657e+08 3.56529e+08 1.18901e+08
	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "weight12_store" );
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "weight23_store" );
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "weight12" );
	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "weight23" );

	//update_weight( weight12 , weight23 , weight12_store ,weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );

	update_weight_store( weight12 , weight23 , weight12_store ,weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );
	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "weight12_store" );
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "weight23_store" );
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT + 1 , "weight12" );
	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT + 1 , "weight23" );


	//平均，分散
	//2.5 6.5
	//3 4 5 6
	//1.11803 1.11803
	//2 2 2 2
	float* mean_value0 = create_mean_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 0 );
	float* mean_value1 = create_mean_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 1 );
	float* std_value0 = create_std_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 0 );
	float* std_value1 = create_std_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 1 );
	float* max_value0 = create_max_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 0 );
	float* max_value1 = create_max_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 1 );
	float* min_value0 = create_min_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 0 );
	float* min_value1 = create_min_vector( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , 1 );

	show_array( mean_value0 , TEST_HIDDEN_COUNT , "mean_value0" );
	show_array( mean_value1 , TEST_INPUT_COUNT+1 , "mean_value1" );
	show_array( std_value0 , TEST_HIDDEN_COUNT , "std_value0");
	show_array( std_value1 , TEST_INPUT_COUNT+1 , "std_value1" );
	show_array( max_value0 , TEST_HIDDEN_COUNT , "max_value0");
	show_array( max_value1 , TEST_INPUT_COUNT+1 , "max_value1" );
	show_array( min_value0 , TEST_HIDDEN_COUNT , "min_value0");
	show_array( min_value1 , TEST_INPUT_COUNT+1 , "min_value1" );

	normal_param np = create_normal_param( weight12 , TEST_HIDDEN_COUNT ,TEST_INPUT_COUNT+1 );

	show_array( np.mean_vector , np.length , "np.mean_vector" );
	show_array( np.std_vector , np.length , "np.std_vector" );
	show_array( np.min_vector , np.length , "np.min_vector" );
	show_array( np.max_vector , np.length , "np.max_vector" );

	normalize( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , np , 0);
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12 normalize" );

	denormalize( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , np , 0 );
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12 normalize" );
	
	normalize( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , np , 1);
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12 normalize" );

	denormalize( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , np , 1 );
	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12 normalize" );

	//mse

	float **weight23_d = new_array( TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 );
	init_data( weight12_d , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 );
	init_data( weight23_d , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 );

	float* ppde23 = new float[TEST_INPUT_COUNT];
	init_data( ppde23 , TEST_INPUT_COUNT );
	
	float* ppde12 = new float[TEST_HIDDEN_COUNT];
	init_data( ppde12 , TEST_HIDDEN_COUNT );

	

	show_array( input_data , TEST_INPUT_COUNT , "input" );
	show_array( hidden_data , TEST_HIDDEN_COUNT , "hidden" );
	show_array( output_data , TEST_INPUT_COUNT , "output" );

	show_array( weight12 , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12" );
	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23" );

	copy_array( weight12 , weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT );
	copy_array( weight23 , weight23_store , TEST_INPUT_COUNT  , TEST_HIDDEN_COUNT );

	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "weight12_store" );
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23_store" );

	backpropagate_mse_last( 
			input_data , hidden_data , output_data ,
			weight23 , weight23_store , weight23_d , 
			0.1 , 0 , 0 ,
			ppde23 , 
			TEST_INPUT_COUNT , TEST_HIDDEN_COUNT );		

	//weight23_store
	//2.75661e+07 7.04467e+07 1.53145e+06
	//5.51952e+07 1.41054e+08 3.0664e+06
	//9.69597e+07 2.47786e+08 5.38665e+06
	//weight23_d
	//2.75661e+07 7.04467e+07 1.53145e+06
	//5.51952e+07 1.41054e+08 3.0664e+06
	show_array( ppde23 , TEST_INPUT_COUNT , "ppde23" );
	show_array( weight23 , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23" );
	show_array( weight23_store , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23_store" );
	show_array( weight23_d , TEST_INPUT_COUNT , TEST_HIDDEN_COUNT+1 , "weight23_d" );

	for(int i = 0;i<TEST_INPUT_COUNT;i++)ppde23[i] = i+1;
	
	show_array( ppde23 , TEST_INPUT_COUNT  , "ppde23" );

	backpropagate_mse_continue( 
		input_data , hidden_data , 
		weight12 , weight23 , weight12_store , weight12_d , 
		0.1 , 0 , 0 , 
		ppde12 , ppde23 , 
		TEST_INPUT_COUNT, TEST_HIDDEN_COUNT , TEST_INPUT_COUNT  );

	//w12_store
	//-427.4 -854.8 -1282.2 -424.4
	//-7861 -15726 -23591 -7858

	//w12_d
	//-428.4 -856.8 -1285.2 -428.4
	//-7866 -15732 -23598 -7866
	show_array( ppde12 , TEST_HIDDEN_COUNT , "ppde12" );
	show_array( weight12_store , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT +1, "w12_store" );
	show_array( weight12_d , TEST_HIDDEN_COUNT , TEST_INPUT_COUNT+1 , "w12_d" );


	delete_normal_param( np );

	delete_value( mean_value0 );
	delete_value( mean_value1 );
	delete_value( std_value0 );
	delete_value( std_value1 );
	delete_value( max_value0 );
	delete_value( max_value1 );
	delete_value( min_value0 );
	delete_value( min_value1 );

	//delete
	delete_value( input_data );
	delete_value( hidden_data );
	delete_value( output_data );

	delete_array( weight12 , TEST_HIDDEN_COUNT );
	delete_array( weight12_store , TEST_HIDDEN_COUNT );
      	delete_array( weight23 , TEST_INPUT_COUNT );
	delete_array( weight23_store , TEST_INPUT_COUNT );

	delete_array( weight12_d , TEST_HIDDEN_COUNT );
	delete_array( weight23_d , TEST_INPUT_COUNT );

	delete_value( ppde12 );
	delete_value( ppde23 );

	delete_value( dlhb );
	delete_value( dlvb );
	

}

#endif
