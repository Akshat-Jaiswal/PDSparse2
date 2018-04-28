#include "multi.h"
#include <omp.h>
#include <cassert>

using namespace std;
StaticModel* readModel(char* file){
	
	StaticModel* model = new StaticModel();
	
	ifstream fin(file);
	char* tmp = new char[LINE_LEN];
	fin >> tmp >> (model->K);
	
	fin >> tmp;
	string name;
	for(int k=0;k<model->K;k++){
		fin >> name;
		model->label_name_list->push_back(name);
		model->label_index_map->insert(make_pair(name,k));
	}
	
	fin >> tmp >> (model->D);
	model->w = new SparseVec[model->D];
	
	vector<string> ind_val;
	int nnz_j;
	for(int j=0;j<model->D;j++){
		fin >> nnz_j;
		model->w[j].resize(nnz_j);
		for(int r=0;r<nnz_j;r++){
			fin >> tmp;
			ind_val = split(tmp,":");
			int k = atoi(ind_val[0].c_str());
			Float val = atof(ind_val[1].c_str());
			model->w[j][r].first = k;
			model->w[j][r].second = val;
		}
	}
	
	delete[] tmp;
	return model;
}
int main(int argc, char** argv){
	
	if( argc < 1+3 ){
		cerr << "multiPred [trainfile] [model] [constraintsFile]" << endl;
		exit(0);
	}
	// temporary vector to hold tranformed yi 
	vector<int> yi_tranformed;

	char* testFile = argv[1];
	char* modelFile = argv[2];
	char* constraintFile=argv[3];
	
	// load model
	StaticModel* model = readModel(modelFile);
	// load data
	Problem* prob = new Problem();
	readData( testFile, prob );
	cerr << "Ntest=" << prob->N << endl;	
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	vector<Labels>* labels = &(prob->labels);
	// read the constraints
	ifstream fin(constraintFile);
	int i=0;
	int tmp;
	int total=0;
	int satisfied=0;
	Float* prod = new Float[model->K];
	for(int i=0;i<prob->N;i++){
		memset(prod, 0.0, sizeof(Float)*model->K);
		// first calculate the scores for each label using our model
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
        for(SparseVec::iterator it=xi->begin(); it!=xi->end(); it++){
			
			int j= it->first;
			Float xij = it->second;
			if( j >= model->D )
				continue;
			SparseVec* wj = &(model->w[j]);
			for(SparseVec::iterator it2=wj->begin(); it2!=wj->end(); it2++){
				int k = it2->first;
				prod[k] += it2->second*xij;
			}
		}
		// transform yi -> yi in our model
		yi_tranformed.clear();
		for(int j=0;j<yi->size();++j){
			string  real_label=prob->label_name_list[yi->at(j)];
			// label in our model
			int y=model->label_index_map->at(real_label);
			yi_tranformed.push_back(y);
		}		
		// now load the constraints for this example 
		//now read until -1 is encounted
		do{
			fin>>tmp;
			//if this label is in ground truth do nothing else
			if(find(yi_tranformed.begin(),yi_tranformed.end(),tmp)==yi_tranformed.end()){
				// add constraints for each positive like we do while training
				for(auto pos:yi_tranformed){
					// check whether score of pos > neg
					if(prod[pos]>prod[tmp])
						satisfied++;
				}
				total+=yi_tranformed.size();
			}
		}while(tmp!=-1);
	}
	fin.close();
	cerr<<"satisfied constraints:"<<satisfied<<endl;
	cerr<<"total constraints:"<<total<<endl;
	cerr<<"\% satisfied constraints:"<<satisfied*100.0/total<<endl;
}
