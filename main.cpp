#include "ClassifierStructure.h"

#include<cstdio>
#include<cstdlib>
#include<ctime>

int main(int argc, char* argv[])
{
	int folds = 5;
	int algo = atoi(argv[3]);
	int comb = atoi(argv[4]);

	char na[][100] = {
		"Single Sample Perceptron Learning",
		"Batch Perceptron Learning",
		"Single sample Relaxation",
		"Batch Relaxation Learning",
		"MSE using Psuedo Inverse"
	};

	char nc[][100] = {
		"One Vs Rest",
		"One Vs One",
	};

	printf("ALGORITHM = %s\n",na[algo-1]);
	printf("COMBINATION = %s\n\n",nc[comb-1]);

	Dataset obj;
	ConfusionMatrix cm;
	
	obj.readData(argv[1],atoi(argv[2]));
	obj.writeData("Data.txt");
	
	crossValidate(obj,folds,0, cm, algo, comb);

	return 0;
}
