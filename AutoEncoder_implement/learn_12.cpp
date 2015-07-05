#include "LearningFrameWork.h"

int main(int argc , char* argv[])
{
	
	LearningFrameWork lfw( argc , argv );

	//一層目の学習
	lfw.Learn12();

	return 0;
}
