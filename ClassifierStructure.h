#ifndef __CLASSIFIER_STRUCTURE_H_INCLUDED__
#define __CLASSIFIER_STRUCTURE_H_INCLUDED__

#include<vector>
#include<string>
#include<set>
#include<map>

using namespace std;

class DataItem
{
	public:
	string classLabel;
	vector<float> feature;
};

class ConfusionMatrix
{
	public:
	map< string, map<string,int> > cfm;
	void initialize(set<string> s);
	void printx();
};


// Container for data feature vectors, class labels for each
// feature vector and functions to read and write data.
// Use constructor and destructor to initialize and clear data.
class Dataset
{
	public:

	// Variables for data
	set<string> class_names;
	vector<DataItem*> data;
	//...
	// Assume that the data file is plain text with each row
	// containing the class label followed by the features, 
	// separated by blank spaces.
	// classcol is zero indexed
	bool readData(const char *filename, int classcol);

	// Write data in the above format.
	bool writeData(const char *filename);
};


// Partition 'complete' dataset randomly into 'folds' parts and 
// returns a pointer to an array of pointers to the partial datasets.
// seed is an argument to random number generator. The function can
// be used to divide data for training, testing and cross validation.
// This need not replicate the data.
Dataset** splitDataset(Dataset complete, int folds=2, int seed=0);

// Merge the datasets indexed by indicesToMerge in the toMerge list and return a
// single dataset. This need not replicate the data.
Dataset* mergeDatasets(Dataset** toMerge, int numDatasets, int* indicesToMerge);

class Model
{
	public:
	vector< vector<float> > model;
	vector< string > fclass;
	vector< string > sclass;
	int algorithm;
	int combination;
};

// Class that carries out training and classification as well as
// store and read the learned model in/from a file.
class LinearClassifier
{
	public:
	// Variables
	int numClasses;
	Model *m;
	// Other variables to hold classifier parameters.
	//...

	// Loads classifier model from a file
	bool loadModel(const char *modelfilename, Model *M);

	// Saves the learned model parameters into a file
	bool saveModel(const char *modelfilename, int algorithm, int combination);

	// learn the parameters of the classifier from possibly multiple training datasets
	// using a specific learning algorithm and combination strategy. 
	// The function should return the training error in [0,1].
	// Algorithms:
	//	1: Single Sample Perceptron Learning (fixed eta)
	//	2: Batch Perceptron Learning (variable eta)
	//	3: Single sample Relaxation (variable eta)
	//	4: Batch Relaxation Learning (variable eta)
	//	5: MSE using Pseudo Inverse
	//	6: MSE using LMS Procedure
	// Combination:
	//	1: 1 vs. Rest
	//	2: 1 vs. 1 with Majority voting
	//	3: 1 vs. 1 with DDAG
	//	4: BHDT.

	float learnModel(Dataset* trainData, int algorithm, int combination);
	
	float single_sample_perceptron(Dataset* trainData, string curr_class, vector<float> &a);
	
	float batch_perceptron(Dataset* trainData, string curr_class, vector<float> &a);

	float single_sample_relaxation(Dataset* trainData, string curr_class, vector<float> &a);

	float batch_relaxation(Dataset* trainData, string curr_class, vector<float> &a);

	float pseudo_inverse(Dataset *trainData, string curr_class, vector<float> &a);

	// Classifies a DataItem and returns the class-label
	string classifySample(DataItem *D, Model *M);

	// classify a set of testDataItems and return the error rate in [0,1].
	// Also fill the entries of the confusionmatrix.
	float classifyDataset(Dataset *testSet, Model *M, ConfusionMatrix &cm);
};

// Divide the dataset and performa an n-fold cross-validation. Compute the
// average error rate in [0,1]. Fill in the standard deviation and confusion matrix.
float crossValidate(Dataset &complete, int folds, float stdDev, ConfusionMatrix &cm, int algo, int comb);

#endif
