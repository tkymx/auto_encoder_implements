#include "LearningFrameWork.h"

int main(int argc, char* argv[])
{
	LearningFrameWork lfw(argc, argv);

	//��w�ڂ̊w�K
	lfw.Learn12();
	//��w�ڂ̊w�K
	lfw.Learn23();
	//��w�ڂ̊w�K
	lfw.LearnAll();

	return 0;
}
