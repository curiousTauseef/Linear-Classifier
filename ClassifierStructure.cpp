#include "ClassifierStructure.h"
#include "armadillo"

#include<iostream>
#include<cstdio>
#include<cstdlib>
#include<cstring>
#include<cmath>
#include<string>
#include<vector>
#include<algorithm>
#include<map>

using namespace std;
using namespace arma;

bool Dataset::readData(const char *filename, int classcol)
{
	FILE *fp = fopen(filename, "r");
	char s[1000],c, *pch;
	int j;
	DataItem *r; 

	data.clear();
	while(fscanf(fp,"%[^\n] ",s) != EOF)
	{
		j = 0;
		r = new DataItem;

		r->feature.push_back(1);
		pch = strtok (s,", ");
		while(pch != NULL)
		{
			if(j++ != classcol)
				r->feature.push_back(atof(pch));
			else
			{
				r->classLabel = string(pch);
				class_names.insert(string(pch));
			}
			pch = strtok (NULL, ", ");
		}
		data.push_back(r);
	}
	fclose(fp);
}

bool Dataset::writeData(const char *filename)
{
	FILE *fp = fopen(filename, "w");
	int l1 = data.size(), l2 = data[0]->feature.size();

	for(int i=0; i<l1; i++)
	{
		fprintf(fp,"%s",data[i]->classLabel.c_str());
		for(int j=1; j<l2; j++)
			fprintf(fp," %f",data[i]->feature[j]);
		if(i != l1-1)
			fputc('\n',fp);
	}
	fclose(fp);
}

ptrdiff_t myrandom (ptrdiff_t i)
{
	return rand()%i;
}

ptrdiff_t (*p_myrandom)(ptrdiff_t) = myrandom;

Dataset** splitDataset(Dataset complete, int folds, int seed)
{
	srand(seed);
	random_shuffle(complete.data.begin(), complete.data.end(), p_myrandom);

	Dataset **partial = new Dataset *[folds];
	for(int i=0; i<folds; i++)
		partial[i] = new Dataset;

	int sz = complete.data.size();
	int foldsize = sz / folds;
	int eqsize = foldsize * folds;

	for(int i=0; i<eqsize; i++)
	{
		partial[i/foldsize]->data.push_back(complete.data[i]);
		partial[i/foldsize]->class_names.insert(complete.data[i]->classLabel);
	}

	for(int i=eqsize; i<sz; i++)
	{
		partial[i-eqsize]->data.push_back(complete.data[i]);
		partial[i-eqsize]->class_names.insert(complete.data[i]->classLabel);
	}

	return partial;
}

Dataset* mergeDatasets(Dataset** toMerge, int numDatasets, int* indicesToMerge)
{
	Dataset *merged  = new Dataset;

	for(int i=0; i<numDatasets; i++)
	{
		int currSet = indicesToMerge[i];
		int sz = toMerge[currSet]->data.size();
		for(int j=0; j<sz; j++)
		{
			merged->data.push_back(toMerge[currSet]->data[j]);
			merged->class_names.insert(toMerge[currSet]->data[j]->classLabel);
		}
	}

	return merged;
}

bool LinearClassifier::saveModel(const char *modelfilename, int algorithm, int combination)
{
	FILE *fp = fopen(modelfilename, "w");
	int n = this->m->model.size(), m = this->m->model[0].size();

	fprintf(fp,"%d %d %d %d\n",algorithm,combination,n,m);
	for(int i=0; i<n; i++)
	{
		fprintf(fp,"%s",this->m->fclass[i].c_str());
		fprintf(fp," %s",this->m->sclass[i].c_str());
		for(int j=0; j<m; j++)
			fprintf(fp," %f",this->m->model[i][j]);
		fputc('\n',fp);
	}
	fclose(fp);
}

bool LinearClassifier::loadModel(const char *modelfilename, Model *M)
{
	vector<float> v;
	char temp[1000];
	float num;
	int n,m;
	FILE *fp = fopen(modelfilename, "r");

	fscanf(fp,"%d %d %d %d ",&M->algorithm,&M->combination,&n,&m);
	for(int i=0; i<n; i++)
	{
		fscanf(fp,"%s ",temp);
		M->fclass.push_back(string(temp));
		fscanf(fp,"%s ",temp);
		M->sclass.push_back(string(temp));

		v.clear();
		for(int j=0; j<m; j++)
		{
			fscanf(fp,"%f ",&num);
			v.push_back(num);
		}
		M->model.push_back(v);
	}
	fclose(fp);
}

float LinearClassifier::learnModel(Dataset* trainData, int algorithm, int combination)
{
	int e;
	this->m = new Model;

	switch(combination)
	{
		case 1:
			switch(algorithm)
			{
				case 1:
					for(set<string>::iterator it=trainData->class_names.begin(); it!=trainData->class_names.end(); it++)
					{
						vector<float> a;
						single_sample_perceptron(trainData,*it,a);
						this->m->model.push_back(a);
						this->m->fclass.push_back(*it);
						this->m->sclass.push_back(*it);

					}
					break;
				case 2:
					for(set<string>::iterator it=trainData->class_names.begin(); it!=trainData->class_names.end(); it++)
					{
						vector<float> a;
						batch_perceptron(trainData,*it,a);
						this->m->model.push_back(a);
						this->m->fclass.push_back(*it);
						this->m->sclass.push_back(*it);
					}
					break;
				case 3:
					for(set<string>::iterator it=trainData->class_names.begin(); it!=trainData->class_names.end(); it++)
					{
						vector<float> a;
						single_sample_relaxation(trainData,*it,a);
						this->m->model.push_back(a);
						this->m->fclass.push_back(*it);
						this->m->sclass.push_back(*it);
					}
					break;
				case 4:
					for(set<string>::iterator it=trainData->class_names.begin(); it!=trainData->class_names.end(); it++)
					{
						vector<float> a;
						batch_relaxation(trainData,*it,a);
						this->m->model.push_back(a);
						this->m->fclass.push_back(*it);
						this->m->sclass.push_back(*it);
					}
					break;
				case 5:
					for(set<string>::iterator it=trainData->class_names.begin(); it!=trainData->class_names.end(); it++)
					{
						vector<float> a;
						pseudo_inverse(trainData,*it,a);
						this->m->model.push_back(a);
						this->m->fclass.push_back(*it);
						this->m->sclass.push_back(*it);
					}
					break;
			}
			break;

		case 2:
			switch(algorithm)
			{
				case 1:
					for(set<string>::iterator it1=trainData->class_names.begin(); it1!=trainData->class_names.end(); it1++)
					{	
						set<string>::iterator it2 = it1;
						it2++;
						for(; it2!=trainData->class_names.end(); it2++)
						{
							Dataset* only2 = new Dataset;
							int n = trainData->data.size();
							for(int i=0; i<n; i++)
							{
								string s = trainData->data[i]->classLabel;
								if(s==*it1 || s==*it2)
									only2->data.push_back(trainData->data[i]);
							}
							vector<float> a;
							single_sample_perceptron(only2,*it1,a);
							this->m->model.push_back(a);
							this->m->fclass.push_back(*it1);
							this->m->sclass.push_back(*it2);
							delete only2;
						}
					}
					break;
				case 2:
					for(set<string>::iterator it1=trainData->class_names.begin(); it1!=trainData->class_names.end(); it1++)
					{
						set<string>::iterator it2 = it1;
						it2++;
						for(; it2!=trainData->class_names.end(); it2++)
						{
							Dataset* only2 = new Dataset;
							int n = trainData->data.size();
							for(int i=0; i<n; i++)
							{
								string s = trainData->data[i]->classLabel;
								if(s==*it1 || s==*it2)
									only2->data.push_back(trainData->data[i]);
							}
							vector<float> a;
							batch_perceptron(only2,*it1,a);
							this->m->model.push_back(a);
							this->m->fclass.push_back(*it1);
							this->m->sclass.push_back(*it2);
							delete only2;
						}
					}
					break;
				case 3:
					for(set<string>::iterator it1=trainData->class_names.begin(); it1!=trainData->class_names.end(); it1++)
					{
						set<string>::iterator it2 = it1;
						it2++;
						for(; it2!=trainData->class_names.end(); it2++)
						{
							Dataset* only2 = new Dataset;
							int n = trainData->data.size();
							for(int i=0; i<n; i++)
							{
								string s = trainData->data[i]->classLabel;
								if(s==*it1 || s==*it2)
									only2->data.push_back(trainData->data[i]);
							}
							vector<float> a;
							single_sample_relaxation(only2,*it1,a);
							this->m->model.push_back(a);
							this->m->fclass.push_back(*it1);
							this->m->sclass.push_back(*it2);
							delete only2;
						}
					}
					break;
				case 4:
					for(set<string>::iterator it1=trainData->class_names.begin(); it1!=trainData->class_names.end(); it1++)
					{
						set<string>::iterator it2 = it1;
						it2++;
						for(; it2!=trainData->class_names.end(); it2++)
						{
							Dataset* only2 = new Dataset;
							int n = trainData->data.size();
							for(int i=0; i<n; i++)
							{
								string s = trainData->data[i]->classLabel;
								if(s==*it1 || s==*it2)
									only2->data.push_back(trainData->data[i]);
							}
							vector<float> a;
							batch_relaxation(only2,*it1,a);
							this->m->model.push_back(a);
							this->m->fclass.push_back(*it1);
							this->m->sclass.push_back(*it2);
							delete only2;
						}
					}
					break;
				case 5:
					for(set<string>::iterator it1=trainData->class_names.begin(); it1!=trainData->class_names.end(); it1++)
					{
						set<string>::iterator it2 = it1;
						it2++;
						for(; it2!=trainData->class_names.end(); it2++)
						{
							Dataset* only2 = new Dataset;
							int n = trainData->data.size();
							for(int i=0; i<n; i++)
							{
								string s = trainData->data[i]->classLabel;
								if(s==*it1 || s==*it2)
									only2->data.push_back(trainData->data[i]);
							}
							vector<float> a;
							pseudo_inverse(only2,*it1,a);
							this->m->model.push_back(a);
							this->m->fclass.push_back(*it1);
							this->m->sclass.push_back(*it2);
							delete only2;
						}
					}
					break;
			}
			break;
	}
}

void printthe_a(vector<float> &a)
{
	//int m = a.size();
	//for(int i=0; i<m; i++)
	//	printf("%f\n",a[i]);
	//printf("\n");
}

float dotP(vector<float> &a,vector<float> &y)
{
	int n = a.size();
	float ans = 0;
	for(int i=0; i<n; i++)
		ans += a[i]*y[i];
	return ans;
}

bool misclassified(vector<float> &a,vector<float> &y)
{
	return dotP(a,y)<0;
}

float LinearClassifier::single_sample_perceptron(Dataset* trainData,string curr_class, vector<float> &a)
{
	int n = trainData->data.size();
	int m = trainData->data[0]->feature.size();
	int k = -1, cnt = 0, div = 1;

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	a.clear();
	a = vector<float>(m,0);
	a[0] = 1;

	while(div<=7*n && cnt!=n)
	{
		k = (k+1)%n;
		if(misclassified(a,trainData->data[k]->feature))
		{
			for(int i=0; i<m; i++)
				a[i] += trainData->data[k]->feature[i];
			cnt = 0;
		}
		else
			cnt++;
		div++;
	}

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	printthe_a(a);
}

float LinearClassifier::batch_perceptron(Dataset* trainData,string curr_class, vector<float> &a)
{
	int n = trainData->data.size();
	int m = trainData->data[0]->feature.size();
	int loop = 1;
	float *Y, div = 1.0, eta = 1;
	Y = new float[m];

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	a.clear();
	a = vector<float>(m,0);
	a[0] = 1;

	while(loop == 1 && div<=7*n)
	{
		loop = 0;
		memset(Y,0,sizeof Y);
		for(int i=0; i<n; i++)
		{
			if(misclassified(a,trainData->data[i]->feature))
			{
				loop = 1;
				for(int j=0; j<m; j++)
					Y[j] += trainData->data[i]->feature[j];
			}
		}
		for(int i=0; i<m; i++)
			a[i] += (eta/div)*Y[i];
		div++;
	}

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	printthe_a(a);
	delete[] Y;
}

float LinearClassifier::single_sample_relaxation(Dataset* trainData, string curr_class, vector<float> &a)
{
	int n = trainData->data.size();
	int m = trainData->data[0]->feature.size();
	int k = -1, div = 1, loop = 1;
	float eta = 1.0, b = 0.5;

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	a.clear();
	a = vector<float>(m,0);
	a[0] = 1;

	while(loop == 1 && div<=7*n)
	{
		k = (k+1)%n;
		float temp, dotpro = dotP(a,trainData->data[k]->feature);
		if(dotpro-b < 0)
		{
			loop = 1;
			temp = (b-dotpro)/dotP(trainData->data[k]->feature,trainData->data[k]->feature);
		}
		for(int j=0; j<m; j++)
			a[j] += (eta/div)*temp*trainData->data[k]->feature[j];
		div++;
	}

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	printthe_a(a);
}

float LinearClassifier::batch_relaxation(Dataset* trainData, string curr_class, vector<float> &a)
{
	int n = trainData->data.size();
	int m = trainData->data[0]->feature.size();
	int div = 1, loop = 1;
	float *Y, eta = 1.0, b = 0.5;
	Y = new float[m];

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;

	a.clear();
	a = vector<float>(m,0);
	a[0] = 1;
	
	while(loop == 1 && div<=4*n)
	{
		loop = 0;
		memset(Y,0,sizeof Y);
		for(int i=0; i<n; i++)
		{
			float temp, dotpro = dotP(a,trainData->data[i]->feature);
			if(dotpro-b < 0)
			{
				loop = 1;
				temp = (b-dotpro)/dotP(trainData->data[i]->feature,trainData->data[i]->feature);
			}
			for(int j=0; j<m; j++)
				Y[j] += temp*trainData->data[i]->feature[j];
		}
		for(int i=0; i<m; i++)
		{
			a[i] += (eta/div)*Y[i];
		}
		div++;
	}

	for(int i=0; i<n; i++)
		if(trainData->data[i]->classLabel != curr_class)
			for(int j=0; j<m; j++)
				trainData->data[i]->feature[j] *= -1;
	
	delete[] Y;
	printthe_a(a);
}

float LinearClassifier::pseudo_inverse(Dataset *trainData, string curr_class, vector<float> &a) 
{
	int n = trainData->data.size();
	int m = trainData->data[0]->feature.size();

	mat Y = zeros<mat>(n,m), A = zeros<mat>(m,1), B = ones<mat>(n,1);

	for(int i=0; i<n; i++)
		for(int j=0; j<m; j++)
			Y(i,j) = trainData->data[i]->feature[j] * ((trainData->data[i]->classLabel != curr_class) ? -1 : 1);
	
	A = pinv(Y)*B;
	for(int i=0; i<m; i++)
		a.push_back(A(i,0));
	//printthe_a(a);
}

string LinearClassifier::classifySample(DataItem *D, Model *M)
{
	string ret = "no class";
	int com = M->combination;
	int n = M->model.size();
	int m = M->model[0].size();

	if(com == 1)
	{
		for(int i=0; i<n; i++)
		{
			if(!misclassified(D->feature, M->model[i]))
				return M->fclass[i];
		}
	}
	else if(com == 2)
	{
		map<string, int> m;
		for(int i=0; i<n; i++)
			misclassified(D->feature, M->model[i]) ? m[M->sclass[i]]++ : m[M->fclass[i]]++;

		int maxx = -1;
		for(map<string,int>::iterator it=m.begin(); it!=m.end(); it++)
		{
			if(it->second > maxx)
			{
				maxx = it->second;
				ret = it->first;
			}
		}
	}
	return ret;
}

float LinearClassifier::classifyDataset(Dataset *testSet, Model *M , ConfusionMatrix &cm)
{
	int ntest = testSet->data.size();
	string lab;

	int c = 0;
	for(int i=0; i<ntest; i++)
	{
		lab = classifySample(testSet->data[i], M);
		if(lab == testSet->data[i]->classLabel) c++;
		cm.cfm[testSet->data[i]->classLabel][lab] += 1;
	}
	printf("correct = %d, total = %d\n",c,ntest);
	return 1.0-c/(float)ntest;
}

void ConfusionMatrix::initialize(set<string> s)
{
	cfm.clear();
	s.insert("no class");
	set<string>::iterator it1, it2;
	for(it1=s.begin(); it1!=s.end(); it1++)
		for(it2=s.begin(); it2!=s.end(); it2++)
			cfm[*it1][*it2] = 0;
}

void ConfusionMatrix::printx()
{
	map< string, map<string,int> >::iterator it;
	map<string,int>::iterator it2;

	printf("CONFUSION MATRIX\n");
	for(it=cfm.begin(); it!=cfm.end(); it++)
		printf("\t%s",it->first.c_str());
	printf("\n");

	for(it=cfm.begin(); it!=cfm.end(); it++)
	{
		printf("%s\t",it->first.c_str());
		for(it2=it->second.begin(); it2!=it->second.end(); it2++)
			printf("%d\t",it2->second);
		printf("\n");
	}
}

float crossValidate(Dataset &complete, int folds, float stdDev, ConfusionMatrix &cm, int algo, int comb)
{
	Dataset **t = splitDataset(complete, folds, time(NULL));
	float errorsum = 0, e2, temp, stanD, avg;

	for(int i=0; i<folds; i++)
	{
		int ind[folds], k = 0;
		for(int j=0; j<folds; j++)
			if(j != i)
				ind[k++] = j;

		Dataset *m = mergeDatasets(t, k, ind);

		LinearClassifier LC;

		LC.learnModel(m, algo, comb);
		LC.saveModel("learnt_model.txt", algo, comb);

		Model *MOD = new Model;
		LC.loadModel("learnt_model.txt",MOD);

		cm.initialize(complete.class_names);
		temp = LC.classifyDataset(t[i], MOD, cm);
		cm.printx();
		printf("\n");

		errorsum += temp;
		e2 += temp*temp*t[i]->data.size();

		delete MOD;
	}
	avg = errorsum/folds;
	stanD = sqrt(e2/folds - avg*avg);
	printf("Average Error = %f\n",avg);
	printf("Standard Deviation = %f\n",stanD);
	return 0;
}
