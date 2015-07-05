#include "LearningFrameWork.h"



int main(int argc , char* argv[])
{
	
	LearningFrameWork lfw( argc , argv );

	//一層目の読み込み
	lfw.Load12();

	//二層目の学習
	lfw.Learn23();

	return 0;
}
