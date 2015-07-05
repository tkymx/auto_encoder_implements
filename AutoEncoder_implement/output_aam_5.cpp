#include "LearningFrameWork.h"



int main(int argc , char* argv[])
{
	
	LearningFrameWork lfw( argc , argv );

	//全層の読み込み
	lfw.LoadALL();
	
	//AAMパラメータの出力
	lfw.output_aam_all();

	return 0;
}
