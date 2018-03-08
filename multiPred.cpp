#include "multi.h"
#include <omp.h>
#include <cassert>

StaticModel* readModel(char* file) {

	StaticModel* model = new StaticModel();

	ifstream fin(file);
	char* tmp = new char[LINE_LEN];
	fin >> tmp >> (model->K);

	fin >> tmp;
	string name;
	for (int k = 0; k < model->K; k++) {
		fin >> name;
		model->label_name_list->push_back(name);
		model->label_index_map->insert(make_pair(name, k));
	}

	fin >> tmp >> (model->D);
	model->w = new SparseVec[model->D];

	vector<string> ind_val;
	int nnz_j;
	for (int j = 0; j < model->D; j++) {
		fin >> nnz_j;
		model->w[j].resize(nnz_j);
		for (int r = 0; r < nnz_j; r++) {
			fin >> tmp;
			ind_val = split(tmp, ":");
			int k = atoi(ind_val[0].c_str());
			Float val = atof(ind_val[1].c_str());
			model->w[j][r].first = k;
			model->w[j][r].second = val;
		}
	}
	fin >> tmp;
	int nnz;
	fin >> nnz;
	if(nnz!=0){
	model->E.resize(nnz);
		for (int r = 0; r < nnz; r++) {
			fin >> tmp;
			ind_val = split(tmp, ":");
			int k = atoi(ind_val[0].c_str());
			Float val = atof(ind_val[1].c_str());
			model->E[r].first = k;
			model->E[r].second = val;
		}
	}
	delete[] tmp;
	return model;
}

int main(int argc, char** argv) {

	if (argc < 1 + 2) {
		cerr << "multiPred [testfile] [model] (-p S <output_file>) (k)" << endl;
		cerr
				<< "\t-p S <output_file>: print top S <label>:<prediction score> pairs to <output_file>, one line for each instance. (default S=0 and no file is generated)"
				<< endl;
		cerr << "\tcompute top k accuracy, default k=1" << endl;
		exit(0);
	}

	char* testFile = argv[1];
	char* modelFile = argv[2];
	char* outFname;
	int S = 0, offset = 0;
	if (argc > 5 && strcmp(argv[3], "-p") == 0) {
		S = atoi(argv[4]);
		outFname = argv[5];
		offset = 3;
	}
	int T = 1;
	if (argc > 3 + offset) {
		T = atoi(argv[3 + offset]);
	}
	StaticModel* model = readModel(modelFile);

	if (T > model->K || S > model->K) {
		cerr << "k or S is larger than domain size" << endl;
		exit(0);
	}
	Problem* prob = new Problem();
	readData(testFile, prob);

	cerr << "Ntest=" << prob->N << endl;

	double start = omp_get_wtime();
	//compute accuracy
	vector<SparseVec*>* data = &(prob->data);
	vector<Labels>* labels = &(prob->labels);
	Float hit = 0.0;
	Float margin_hit = 0.0;
	Float* prod = new Float[model->K];
	int* max_indices = new int[model->K + 1];
	for (int k = 0; k < model->K + 1; k++) {
		max_indices[k] = -1;
	}
	ofstream fout;
	if (S != 0) {
		cerr << "Printing Top " << S << " <label>:<prediction score> pairs to "
				<< outFname << ", one line per instance" << endl;
		fout.open(outFname);
	}
	// now once the model is loaded create the same structures
	int K = model->K;
	// dense vector for now only
	Float* E = new Float[K * (K - 1) / 2];
	memset(E, 0.0, sizeof(Float) * K * (K - 1) / 2);
	vector<int> e_hash_nnz_index;

	for (SparseVec::iterator it = model->E.begin(); it != model->E.end();
			++it) {
		int ind = it->first;
		int val = it->second;
		e_hash_nnz_index.push_back(ind);
		E[ind] = val;
	}
	// initializing the inverted index
	pair<int,int> *inverted_index;
	inverted_index = new pair<int, int> [K * (K - 1) / 2];
	for (int u = 0, start = 0; u < K; ++u) {
		for (int v = 0; v < u; ++v) {
			inverted_index[start++] = make_pair(u, v);
		}
	}
	for (int i = 0; i < prob->N; i++) {
		memset(prod, 0.0, sizeof(Float) * model->K);

		SparseVec* xi = data->at(i);
		Labels* yi = &(labels->at(i));
		int Ti = T;
		if (Ti <= 0)
			Ti = yi->size();
		int top = max(Ti, S);
		for (int ind = 0; ind < model->K; ind++) {
			max_indices[ind] = ind;
		}
		for (SparseVec::iterator it = xi->begin(); it != xi->end(); it++) {

			int j = it->first;
			Float xij = it->second;
			if (j >= model->D)
				continue;
			SparseVec* wj = &(model->w[j]);
			for (SparseVec::iterator it2 = wj->begin(); it2 != wj->end();
					it2++) {
				int k = it2->first;
				prod[k] += it2->second * xij;
			}
		}
		// from here use the approx algorithm that uses edge weights
		// create a sparse vector for storing ybar
		Labels *ybar = new Labels();

		Labels *neg = new Labels();
		bool stop = false;
		int index;
		Float score;
		// now initially put all the labels with positive score into ybar
		for (int i = 0; i < model->K; ++i) {
			if (prod[i] > 0)
				ybar->push_back(i);
			else
				neg->push_back(i);
		}
		while (!stop) {
			stop = true;
			// iterate over all negatives to see if we can set them to positives
			for (Labels::iterator it = neg->begin(); it != neg->end(); ++it) {
				score = prod[*it];
				// now add edge scores if we make this active then
				for (Labels::iterator it2 = ybar->begin(); it2 != ybar->end();
						++it2) {
					index = locate(*it, *it2);
					score += E[index];
				}
				// if it is positive then move it to active and remove from negatives
				if (score >= 0) {
					// add it to actives
					ybar->push_back(*it);
					// remove it from negs
					*it = *(neg->end() - 1);
					neg->erase(neg->end() - 1);
					it--;
					// allow next iteration as configuration has changed
					stop = false;
					continue;
				}
			}
		}
		// now y_bar contains active labels that maximizes the sum
		delete neg;

		// now split the edge scores to prods
		//sort(ybar->begin(), ybar->end());
		if (top == ybar->size()) {
			// then no need to add edge weights simply output these
			// labels
			for(int j=0;j<top;++j)
				max_indices[j]=ybar->at(j);
		}
		// iterate over all non zero edges and add their potentials equally the nodes
		else if (top < ybar->size()) {
			vector<int>& eS = e_hash_nnz_index;
			Float ek = 0.0;
			int u, v, k;
			for (vector<int>::iterator it = eS.begin(); it != eS.end(); ++it) {
				k = *(it);
				ek = E[k];
				pair<int, int> uv = inverted_index[k];
				u = uv.first;
				v = uv.second;
				if (binary_search(ybar->begin(), ybar->end(), u)
						&& binary_search(ybar->begin(), ybar->end(), v)) {
					// that is if both are present in ybar then add to their node potentials
					prod[u] += ek / 2;
					prod[v] += ek / 2;
				}
			}
			for (int i = 0; i < ybar->size(); ++i) {
				max_indices[i] = ybar->at(i);
			}
			nth_element(max_indices, max_indices + top,
					max_indices + ybar->size(), ScoreComp(prod));
			sort(max_indices, max_indices + yi->size(), ScoreComp(prod));

		} else {
			// do what we were doing previously
			nth_element(max_indices, max_indices + top, max_indices + model->K,
					ScoreComp(prod));
			sort(max_indices, max_indices + top, ScoreComp(prod));

		}
		delete ybar;
		for (int k = 0; k < Ti; k++) {
			bool flag = false;
			for (int j = 0; j < yi->size(); j++) {
				if (prob->label_name_list[yi->at(j)]
						== model->label_name_list->at(max_indices[k])) {
					flag = true;
				}
			}
			if (flag)
				hit += 1.0 / Ti;
		}
		if (S != 0) {
			for (int k = 0; k < S; k++) {
				if (k != 0) {
					fout << " ";
				}
				fout << model->label_name_list->at(max_indices[k]) << ":"
						<< prod[max_indices[k]];
			}
			fout << endl;
		}
	}
	if (S != 0) {
		fout.close();
	}
	delete[] inverted_index;
	delete[] E;

	double end = omp_get_wtime();
	cerr << "Top " << T << " Acc=" << ((Float) hit / prob->N) << endl;
	cerr << "pred time=" << (end - start) << " s" << endl;
}
