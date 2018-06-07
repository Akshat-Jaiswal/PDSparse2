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
	fin.close();
	delete[] tmp;
	return model;
}

int main(int argc, char** argv){
	
	if( argc < 1+3 ){
		cerr << "multiPred [embedding_file] [testfile] [model] (-p S <output_file>) (k)" << endl;
        cerr << "\t-p S <output_file>: print top S <label>:<prediction score> pairs to <output_file>, one line for each instance. (default S=0 and no file is generated)" << endl;
        cerr << "\tcompute top k accuracy, default k=1" << endl;
		exit(0);
	}

	char* embeddingFile = argv[1];
	char* testFile = argv[2]; 
	char* modelFile = argv[3];
    char* outFname;
    int S = 0, offset = 0;
    vector<Float> *embeddings;
    if (argc > 6 && strcmp(argv[4], "-p") == 0){
        S = atoi(argv[5]);
        outFname = argv[6];
        offset = 3;
    }
	int T = 1;
	if (argc > 4 + offset){
		T = atoi(argv[4 + offset]);
	}
	StaticModel* model = readModel(modelFile);
    // read embeddings
    embeddings= readEmbeddings(embeddingFile, *(model->label_index_map));

    if (T > model->K || S > model->K){
        cerr << "k or S is larger than domain size" << endl;
        exit(0);
    }
	Problem* prob = new Problem();
	readData( testFile, prob );
	
	cerr << "Ntest=" << prob->N << endl;
	
	double start = omp_get_wtime();
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	vector<Labels>* labels = &(prob->labels);
	Float hit=0.0;
	Float margin_hit = 0.0;
	Float* prod = new Float[model->K];
	int* max_indices = new int[model->K+1];
	for(int k = 0; k < model->K+1; k++){
		max_indices[k] = -1;
	}
    ofstream fout;
    if (S != 0){
        cerr << "Printing Top " << S << " <label>:<prediction score> pairs to " << outFname << ", one line per instance" << endl;
        fout.open(outFname);
    }
	for(int i=0;i<prob->N;i++){
		memset(prod, 0.0, sizeof(Float)*model->K);
		// initialize all the scores with embedding scores
		for(int k=0;k<model->K;++k){
			for(int j=0;j<model->ED;++j){
				prod[k]+= model->E[j] * embeddings[k][j]/ T;
			}
		}
		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
		int Ti = T;
		if (Ti <= 0)
			Ti = yi->size();
        int top = max(Ti, S);
		for(int ind = 0; ind < model->K; ind++){
			max_indices[ind] = ind;
		}
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
        nth_element(max_indices, max_indices+top, max_indices+model->K, ScoreComp(prod));
        sort(max_indices, max_indices+top, ScoreComp(prod));
		for(int k=0;k<Ti;k++){
			bool flag = false;
			for (int j = 0; j < yi->size(); j++){
				if (prob->label_name_list[yi->at(j)] == model->label_name_list->at(max_indices[k])){
					flag = true;
				}
			}
			if (flag)
				hit += 1.0/Ti;
		}
        if (S != 0){
            for (int k = 0; k < S; k++){
                if (k != 0){
                    fout << " ";
                }
                fout << model->label_name_list->at(max_indices[k]) << ":" << prod[max_indices[k]];
            }
            fout << endl;
        }
	}
    if (S != 0){
	    fout.close();
    }

	double end = omp_get_wtime();
	cerr << "Top " << T << " Acc=" << ((Float)hit/prob->N) << endl;
	cerr << "pred time=" << (end-start) << " s" << endl;
}
