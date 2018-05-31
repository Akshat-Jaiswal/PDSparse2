#include "multi.h"
#include <omp.h>
#include <cassert>

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
	// read embedding parameters
	fin >> tmp >> (model->ED);
	Float val;
	for(int i=0;i< model-> ED;++i){
		fin>>val;
		model->E.push_back(val);
	}
	delete[] tmp;
	return model;
}

int main(int argc, char** argv){
	
	if( argc < 1 ){
		cerr << "multiPred [model] ..." << endl;
        cerr << "\t-p S <output_file>: print top S <label>:<prediction score> pairs to <output_file>, one line for each instance. (default S=0 and no file is generated)" << endl;
        cerr << "\tcompute top k accuracy, default k=1" << endl;
		exit(0);
	}
	char* fname="model.final";
	// read the first model
	StaticModel* model = new StaticModel();
	Float **w;
	Float *E;
	cerr<<"Reading First Model "<<argv[1] << endl;
	// ------------------------------------ //
	ifstream fin(argv[1]);
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
	cerr<<model->D<<" "<<model->K<<endl;
	// initialize dense array
	w= new Float*[model->D];
	for(int i=0;i<model->D;++i){
		w[i]= new Float[model->K];
		memset(w[i], 0.0, sizeof(Float)* model->K);
	}
	
	vector<string> ind_val;
	int nnz_j;
	for(int j=0;j<model->D;j++){
		fin >> nnz_j;
		for(int r=0;r<nnz_j;r++){
			fin >> tmp;
			ind_val = split(tmp,":");
			int k = atoi(ind_val[0].c_str());
			Float val = atof(ind_val[1].c_str());
			w[j][k]+=val;
		}
	}
	// read embedding parameters
	fin >> tmp >> (model->ED);
	E= new Float[model->ED];
	memset(E, 0.0, sizeof(Float)*model->ED);
	Float val;
	for(int i=0;i< model-> ED;++i){
		fin>>val;
		E[i]+=val;
	}
	fin.close();
	cerr<<"Done initialization !!"<<endl;
	// ------------------------------------------------//
	// now read other models
	for(int i=2;i< argc;++i){
		cerr<<"Reading "<<argv[i]<<endl;
		ifstream fin(argv[i]);
		fin >> tmp >> (model->K);
		fin >> tmp;
		string name;
		for(int k=0;k<model->K;k++){
			fin >> name;
			// model->label_name_list->push_back(name);
			// model->label_index_map->insert(make_pair(name,k));
		}
		fin >> tmp >> (model->D);		
		vector<string> ind_val;
		int nnz_j;
		for(int j=0;j<model->D;j++){
			fin >> nnz_j;
			for(int r=0;r<nnz_j;r++){
				fin >> tmp;
				ind_val = split(tmp,":");
				int k = atoi(ind_val[0].c_str());
				Float val = atof(ind_val[1].c_str());
				w[j][k]+=val;
			}
		}
		// read embedding parameters
		fin >> tmp >> (model->ED);
		Float val;
		for(int i=0;i< model-> ED;++i){
			fin>>val;
			E[i]+=val;
		}
		fin.close();
	}
	int models_combined=argc-1;
	cerr<<"Combined all models"<<endl;
	// write down the final models
	ofstream fout(fname);
	fout << "nr_class " << model->K << endl;
	fout << "label ";
	for(vector<string>::iterator it=model->label_name_list->begin();
			it!=model->label_name_list->end(); it++){
		fout << *it << " ";
	}
	fout << endl;
	fout << "nr_feature " << model->D << endl;
	for(int i=0;i< model->D;++i){
		int nnz=0;
		// first count nnz entries in w[i]
		for(int j=0;j<model->K;++j){
			if(fabs(w[i][j]/models_combined) > EPS)
				nnz++;
		}
		fout << nnz << " ";
		// now repeat the above step
		for(int j=0;j<model->K;++j){
			if(fabs(w[i][j]/models_combined) > EPS){
				fout<< j <<":"<<w[i][j]/models_combined<<" ";
			}
		}
		fout<<endl;
	}
		// write down the embedding parameters
	fout << "embeddings_dimensions "<< model->ED <<endl;
	int i;
	for(i=0;i< model->ED-1;++i){
		fout<<E[i]/models_combined<<" ";
	}
	fout<<E[i]/models_combined<<endl;
	fout.close();		
	cerr<<"Final Model Written to Disk"<<endl;
//  clean up
	for(int i=0;i<model->D;++i)
		delete[] w[i];

	delete[] E;
	delete[] w;
	delete[] tmp;
}
