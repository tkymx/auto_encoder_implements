#include "LearningFrameWork.h"



int main(int argc , char* argv[])
{
	
	LearningFrameWork lfw( argc , argv );

	//一層目の読み込み
	lfw.Load12();

	//二層目の読み込み
	lfw.Load23();

	//全層の学習
	lfw.LearnAll();

	return 0;
}
