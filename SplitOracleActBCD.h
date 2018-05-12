#include "util.h"
#include "multi.h"
#include "newHash.h"
#include <iomanip>
#include <cassert>
#include <map>
#include <unordered_set>
using namespace std;
#define loc(k) k*split_up_rate/K

extern double overall_time;
class constraints{
public:
// for storing sets of size k
	vector<pair<Labels*,Float>> act_k_pos_index;
	vector<pair<Labels*,Float>> act_k_neg_index;
};
class SplitOracleActBCD{

    public:
        SplitOracleActBCD(Param* param){
            train = param->train;
            heldoutEval = param->heldoutEval;	
            early_terminate = param->early_terminate;
            data = &(train->data);
            labels = &(train->labels);
            lambda = param->lambda;
            C = param->C;
            precision=param->precision;
            decay_rate=param->decay;
            solver= param->solver;
            N = train->N;
            D = train->D;
            K = train->K;
            dump_model = param->dump_model;
            if (param->dump_model){
                modelFname =string( param->modelFname);
            }

            //a random permutation
            hashfunc = new HashClass(K);
            hashindices = hashfunc->hashindices;

            //compute useful statistics and l_1 norm of every feature x_i
            nnz_a_i = 0.0; d = 0.0;
            cdf_sum = new vector<Float>();
            for(int i = 0; i < N; i++){
                SparseVec* xi = data->at(i);
                Float _cdf = 0.0;
                nnz_a_i += labels->at(i).size();
                d += xi->size();
                for (SparseVec::iterator it = xi->begin(); it < xi->end(); it++){
                    _cdf += fabs(it->second);
                }
                cdf_sum->push_back(_cdf);
            }
            nnz_a_i /= N; d /= N;
            max_iter = param->max_iter;

            //setting up for sampling oracle
            speed_up_rate = param->speed_up_rate;
            split_up_rate = param->split_up_rate;
            if( speed_up_rate==-1 )
                speed_up_rate = ceil( min(5.0*D*K/nnz(train->data)/C[0]/log((Float)K), d/10.0) );
            cerr << "lambda=" << lambda << ", C=";
            for(int p=0;p< precision;++p)
            	cerr  << C[p] << " ";
            cerr<<endl;
            using_importance_sampling = param->using_importance_sampling;
            if (using_importance_sampling){
                cerr << "using importance sampling" << ", speed up rate=" << speed_up_rate << endl;
            } else {
                cerr << "using uniform sampling" << ", speed up rate=" << speed_up_rate << endl;
            }

            //number of variables added to active set in each iteration.
            max_select = param->max_select;
            if (max_select == -1){
                int avg_label = 0;
                for (int i = 0; i < N; i++){
                    avg_label += labels->at(i).size();
                }
                avg_label /= N;
                if (avg_label < 1)
                    avg_label = 1;
                max_select = avg_label;
            }
            cerr<<" Max Combinations Selected Per iteration =" << max_select << endl;
            //global cache
            prod = new Float[K];	
            prod_cache = new Float[K];
            inside = new bool[K];
            inside_index = new bool[K];
            norms= new Float[K];	
            freq= new int[K];
            memset(prod, 0.0, sizeof(Float)*K);
            memset(prod_cache, 0.0, sizeof(Float)*K);
            memset(inside, false, sizeof(bool)*K);
            memset(inside_index, false, sizeof(bool)*K);
        }

        ~SplitOracleActBCD(){
            for(int j=0;j<D;j++)
                delete[] v[j];
            delete[] v;

            for(int j=0;j<D;j++)
                delete[] w[j];
            delete[] w;

            //delete global cache
            delete[] inside;
            delete[] inside_index;
            delete[] prod;
            delete[] prod_cache;
            delete[] freq;
            delete[] norms;

#ifdef USING_HASHVEC
            delete[] size_v;
            delete[] util_v;
            delete[] size_w;
#endif

    		for (int i = 0; i < N; ++i){
    			int prec=labels->at(i).size();
    			prec=min(prec,precision);
    			for (int p=0;p<prec;++p){
    				for(auto elem:cons[i][p].act_k_pos_index)
    					delete elem.first;
    				for(auto elem:cons[i][p].act_k_neg_index)
    					delete elem.first;
    			}
    		}
    		delete[] cons;

    		delete[] hashindices;
            delete[] non_split_index;	
            delete[] w_hash_nnz_index;
        }

        Model* solve(){
            //initialize alpha and v ( s.t. v = X^Talpha )

            //for storing best model
            non_split_index = new vector<int>[D];
#ifdef USING_HASHVEC
            v = new pair<int, pair<Float, Float>>*[D];
            size_v = new int[D];
            util_v = new int[D];
            memset(util_v, 0, D*sizeof(int));
            for (int j = 0; j < D; j++){
                size_v[j] = INIT_SIZE;
                v[j] = new pair<int, pair<Float,Float>>[size_v[j]];
                for(int k = 0; k < size_v[j]; k++){
                    v[j][k] = make_pair(-1, make_pair(0.0, 0.0));
                }
            }
            //for storing best model w
            size_w = new int[D];
            w = new pair<int, Float>*[D];
            for (int j = 0; j < D; j++){
                w[j] = new pair<int, Float>[1];
            }
#else
            v = new pair<Float, Float>*[D]; //w = prox(v);
            for(int j=0;j<D;j++){
                v[j] = new pair<Float, Float>[K];
                for(int k=0;k<K;k++){
                    v[j][k] = make_pair(0.0, 0.0);
                }
            }
            //for storing best model w
            w = new Float*[D];
            for (int j = 0; j < D; j++){
                w[j] = new Float[K];
                memset(w[j], 0.0, sizeof(Float)*K);
            }
#endif
            //initialize non-zero index array w
            w_hash_nnz_index = new vector<int>*[D];
            for(int j=0;j<D;j++){
                w_hash_nnz_index[j] = new vector<int>[split_up_rate];
                for(int S=0;S < split_up_rate; S++){
                    w_hash_nnz_index[j][S].clear();
                }
            }
            //initialize Q_diag (Q=X*X') for the diagonal Hessian of each i-th subproblem
            Q_diag = new Float[N];
            for(int i=0;i<N;i++){
                SparseVec* ins = data->at(i);
                Float sq_sum = 0.0;
                for(SparseVec::iterator it=ins->begin(); it!=ins->end(); it++)
                    sq_sum += it->second*it->second;
                Q_diag[i] = sq_sum;
            }
            //indexes for permutation of [N]
            int* index = new int[N];
            for(int i=0;i<N;i++)
                index[i] = i;
            //initialize active set out of [K] for each sample i
            //act_k_index = new vector<pair<int, Float>>[N];

            // initialize the constraint sets of each example
            cons= new constraints*[N];
            memset(freq,0,sizeof(int)*K);
            vector<int> tmp;
            for(int i=0,k;i<N;i++){
                Labels* yi = &(labels->at(i));
                for (Labels::iterator it = yi->begin(); it < yi->end(); it++){
                //    act_k_index[i].push_back(make_pair(*it, 0.0));
                    freq[*it]++;
                }
                // now create seperate constraint sets for individual examples
                // add all the combinations of size 1-k
                k=precision<yi->size()?precision:yi->size();
                cons[i]= new constraints[k];
                for(int p=1;p<=k;++p){
                	makeCombiUtil(cons[i][p-1].act_k_pos_index,tmp,0,p,yi);
                }
            }

            //for storing best model
            //best_act_k_index = NULL;

            //main loop
            ofstream f_dual("objectives.txt");
            int terminate_countdown = 0;
            double search_time=0.0, subsolve_time=0.0, maintain_time=0.0;
            double last_search_time = 0.0, last_subsolve_time = 0.0, last_maintain_time = 0.0;
            Float** alpha_i_new_diff ;
            Float** alpha_i_new_pos;
            // map for storing currently active alphas for example i updates
            map<int,Float> actives;

            iter = 0;
            best_heldout_acc = -1.0; best_model = NULL;
            Float dual=0.0;
            while( iter < max_iter ){

                random_shuffle( index, index+N );
                for(int r=0,prec;r<N;r++){

                    int i = index[r];
                    SparseVec* x_i = data->at(i);
                    Labels* yi = &(labels->at(i));
                    prec=yi->size()<precision? yi->size():precision;
#ifdef USING_HASHVEC
                    int index_alpha = 0, index_v = 0;
#endif
                    //search active variable
                    search_time -= omp_get_wtime();
                    search_active_i_graph(i,cons[i],prec);
                    search_time += omp_get_wtime();
                    //solve subproblem
                    subsolve_time -= omp_get_wtime();
                    alpha_i_new_diff= new Float*[prec];
                    alpha_i_new_pos= new Float*[prec];
                    //TODO:  gradient descent for positive part
                    // our methods for negative parts 
                    for(int p=0;p<prec;++p){
                    	// solve the posivite part which is common to all methods
                    	alpha_i_new_pos[p] = new Float[cons[i][p].act_k_pos_index.size()];
                    	// perform gradient descent (unconstrained) on positive part
                        subSolve7(i,			
                    		cons[i][p].act_k_pos_index,
			                alpha_i_new_pos[p],
							prec
			            );
                        // now solve the negative part
                    	alpha_i_new_diff[p]= new Float[cons[i][p].act_k_neg_index.size()];
        		        subSolve4(i,
    							cons[i][p].act_k_neg_index,
    						//	U[i][p],
                                alpha_i_new_diff[p],
    							p+1
                    	);
                    }
                    subsolve_time += omp_get_wtime();
                    //maintain v =  X^T\alpha;  w = prox_{l1}(v);
                    maintain_time -= omp_get_wtime();
    				int ind ;
    				actives.clear();
    				Float delta_alpha;
    				// identify active labels for each of sets of size k
    				for(int p=0;p<prec;++p){
						vector<pair<Labels*,Float>>& act_k_index_neg=
								cons[i][p].act_k_neg_index;
						ind=0;
						// identify active labels for update from negatives
						for (vector<pair<Labels*,Float>>::iterator it =
								act_k_index_neg.begin(); it != act_k_index_neg.end();
								it++) {
							delta_alpha = alpha_i_new_diff[p][ind++] - it->second;;
                            for (Labels::iterator it2 = it->first->begin();
									it2 != it->first->end(); ++it2) {
								if (actives.find(-(*it2)-1) == actives.end()) {
									actives[-(*it2)-1] = 0;
								}
								actives[-(*it2)-1] += delta_alpha;
							}
						}
						ind=0;
						// identify active labels for update from positives
						vector<pair<Labels*,Float>>& act_k_index_pos=
								cons[i][p].act_k_pos_index;
						for (vector<pair<Labels*,Float>>::iterator it =
								act_k_index_pos.begin(); it != act_k_index_pos.end();
								it++) {
							delta_alpha = alpha_i_new_pos[p][ind++] - it->second;;
	                        for (Labels::iterator it2 = it->first->begin();
									it2 != it->first->end(); ++it2) {
								if (actives.find((*it2)+1) == actives.end()) {
									actives[(*it2)+1] = 0;
								}
								actives[(*it2)+1] += delta_alpha;
							}
						}
    				}
                    for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
                        int J = it->first; 
                        Float f_val = it->second;
                        vector<int>* wJ = w_hash_nnz_index[J];
#ifdef USING_HASHVEC
                        pair<int, pair<Float, Float>>* vj = v[J];
                        int size_vj = size_v[J];
                        int util_vj = util_v[J];
                        int size_vj0 = size_vj - 1;
                        ind = 0;
    					for (std::map<int, Float>::iterator it2 = actives.begin();
    							it2 != actives.end(); it2++) {
    						int k = it2->first;
    						Float delta_alpha = it2->second;
    						if(k<0){
    							k=-k;
    							delta_alpha=-delta_alpha;
    						}
    						k--;
                            if( fabs(delta_alpha) < EPS )
                                continue;
                            //update v, w
                            find_index(vj, index_v, k, size_vj0, hashindices);
                            Float vjk = vj[index_v].second.first + f_val*delta_alpha;
                            Float wjk_old = vj[index_v].second.second;
                            Float wjk = prox_l1(vjk, lambda);
                            vj[index_v].second = make_pair(vjk, wjk);
                            if (vj[index_v].first == -1){
                                vj[index_v].first = k;
                                if ((++util_v[J]) > size_vj * UPPER_UTIL_RATE){
                                    resize(vj, v[J], size_v[J], size_vj, size_vj0, util_v[J], hashindices);
                                }
                            }
                            if ( wjk_old != wjk ){
                                if (fabs(wjk_old) < EPS){
                                    wJ[loc(k)].push_back(k);
                                }
                            }
                        }
#else
    					pair<Float, Float>* vj = v[J];
    					ind = 0;
    					int k;
    					for (std::map<int, Float>::iterator it2 = actives.begin();
    							it2 != actives.end(); it2++) {
    						k = it2->first;
    						Float delta_alpha = it2->second;
    						if(k<0){
    						    k=-k;
    						    delta_alpha=-delta_alpha;
    						}
    						k--;
    						if (fabs(delta_alpha) < EPS)
    							continue;
    						//update v, w
    						pair<Float, Float> vjk_wjk = vj[k];
    						Float vjk = vjk_wjk.first + f_val * delta_alpha;
    						Float wjk = prox_l1(vjk, lambda);

    						Float wjk_old = vjk_wjk.second;
    						vj[k] = make_pair(vjk, wjk);
                            if (wjk_old != wjk) {
    							if (fabs(wjk_old) < EPS) {
    								wJ[loc(k)].push_back(k);
    							}
    						}
    					}
#endif
                    }
    				//update alphas for each set
                    for(int p=0;p<prec;++p){
						bool has_zero = 0;
						vector<pair<Labels*,Float>>& act_k_index_neg=
								cons[i][p].act_k_neg_index;

						ind = 0;
						// update negatives
						for (vector<pair<Labels*,Float>>::iterator it =
								act_k_index_neg.begin(); it != act_k_index_neg.end();
								it++) {
							it->second = alpha_i_new_diff[p][ind++];
							has_zero |= (fabs(it->second) < EPS);
						}

						delete[] alpha_i_new_diff[p];
						//shrink act_k_neg_index
						if (has_zero) {
							vector<pair<Labels*,Float>> tmp_vec;
							tmp_vec.reserve(act_k_index_neg.size());
							for (vector<pair<Labels*,Float>>::iterator it =
									act_k_index_neg.begin(); it != act_k_index_neg.end();
									it++) {
								if ( it==act_k_index_neg.begin()||fabs(it->second) > EPS) {
									tmp_vec.push_back(*it);
								} else
									delete it->first;
							}
							act_k_index_neg = tmp_vec;
						}

						vector<pair<Labels*,Float>>& act_k_index_pos=
								cons[i][p].act_k_pos_index;
						ind = 0;
						// update positives
						for (vector<pair<Labels*,Float>>::iterator it =
								act_k_index_pos.begin(); it != act_k_index_pos.end();
								it++) {
							it->second = alpha_i_new_pos[p][ind++];
						}
                        delete[] alpha_i_new_pos[p];


                    }
                    // free up the pointers
                    delete[] alpha_i_new_pos;
                    delete[] alpha_i_new_diff;
    				maintain_time += omp_get_wtime();

                }

                cerr << "i=" << iter << "\t" ;
                nnz_a_i = 0.0;

                for(int i=0;i<N;i++){
                	int prec=precision<(labels->at(i)).size()?precision:(labels->at(i)).size();
                	for(int p=0;p<prec;++p)
	                    nnz_a_i += cons[i][p].act_k_pos_index.size()+ cons[i][p].act_k_neg_index.size();
                }

                nnz_a_i /= N;
                cerr << "nnz_a_i="<< (nnz_a_i) << "\t";
                nnz_w_j = 0.0;
                for(int j=0;j<D;j++){
                    for(int S=0;S < split_up_rate; S++){
                        nnz_w_j += w_hash_nnz_index[j][S].size(); //util_w[j][S];

                    }
                }
                nnz_w_j /= D;
                cerr << "nnz_w_j=" << (nnz_w_j) << "\t";
                cerr << "search=" << search_time-last_search_time << "\t";
                cerr << "subsolve=" << subsolve_time-last_subsolve_time << "\t";
                cerr << "maintain=" << maintain_time-last_maintain_time << "\t";
                if (search_time - last_search_time > (subsolve_time-last_subsolve_time + maintain_time - last_maintain_time)*2){
                    max_select *= 2;
                }
                if (max_select > 100){
                    max_select = 100;
                }
                last_search_time = search_time;
                last_maintain_time = maintain_time;
                last_subsolve_time = subsolve_time;
                overall_time += omp_get_wtime();
                Float dual_cur=dual_obj();
                f_dual<<nnz_a_i<<" "<<dual_cur<<endl;
                cerr << "dual_obj=" << dual_cur << "\t";
                //early terminate: if heldout_test_accuracy does not increase in last <early_terminate> iterations, stop!	
                if( heldoutEval != NULL){
#ifdef USING_HASHVEC
                    Float heldout_test_acc = heldoutEval->calcAcc(v, size_v, w_hash_nnz_index, hashindices, split_up_rate);
#else
                    Float heldout_test_acc = heldoutEval->calcAcc(v, w_hash_nnz_index, split_up_rate);
#endif

/*                  Float heldout_test_acc=evaluate_xi(); 
*/					 cerr << "heldout Acc=" << heldout_test_acc << " ";
                    if ( heldout_test_acc > best_heldout_acc){
                        best_heldout_acc = heldout_test_acc;
                        dual=dual_cur;
                        store_best_model();
                        if (dump_model){
                            string name = modelFname + "." + to_string(iter);
                            char* fname = new char[name.length()+1];
                            strcpy(fname, name.c_str());
                            cerr << ", dump_model_file=" << fname;
                            best_model->writeModel(fname);
                            delete fname;
                        }
                        terminate_countdown = 0;
                    } else {
                        cerr << "(" << (++terminate_countdown) << "/" << early_terminate << ")";
                        if (terminate_countdown == early_terminate){
                            overall_time -= omp_get_wtime();
                            break;
                        }
                    }
                }
                cerr << endl;

                overall_time -= omp_get_wtime();
                iter++;
            }
            f_dual.close();
            cerr << endl;

            //recover act_k_index to the best state so far
            //This is because act_k_index is not a part of model, but we might need to use act_k_index possibly in Post Solve
/*
            if (best_act_k_index != NULL){
                for (int i = 0; i < N; i++){
                    act_k_index_neg[i] = best_act_k_index_neg[i];
                    act_k_index_pos[i] = best_act_k_index_pos[i];

                }
            }
*/
            if (best_model == NULL){
                store_best_model();
            }

            //computing heldout accuracy 	
            cerr << "train time=" << (overall_time + omp_get_wtime()) << endl;
            cerr << "search time=" << search_time << endl;
            cerr << "subsolve time=" << subsolve_time << endl;
            cerr << "maintain time=" << maintain_time << endl;
    		cerr << "Dual Obj=" << dual << endl;

            //delete algorithm-specific variables
            // writing to a file for exploratory analysis
            ofstream fout("label_norms.txt");
            for(int i=0;i<K;++i){
            	fout<<freq[i]<<"\t"<<norms[i]<<endl;
            }
            fout.close();
            delete[] Q_diag;
            delete cdf_sum;
            delete[] index;
            return best_model;
        }

        //compute 1/2 \|w\|_2^2 + \sum_{i,k: k \not \in y_i} alpha_{i, k}
        Float evaluate_xi(){
        	int counts=0;
        	bool flag;
        	Float psi,min_score,max_score,scores;
        	int S=0;
        	for(int I=0;I<N;++I){
        	// compute the new updated scores for each label
//compute <xi,wk> for k=1...K
            		Labels* yi = &(labels->at(I));
            		SparseVec* xi = data->at(I);
            		// calculate the scores for each class
            		memset(prod_cache, 0.0, sizeof(Float) * K);
            		for (SparseVec::iterator current_index = xi->begin();
            				current_index < xi->end(); current_index++) {
            			Float xij = current_index->second;
            			int j = current_index->first;
            			vector<int>& wjS = w_hash_nnz_index[j][S];
            			if (wjS.size() == 0)
            				continue;
            			int k = 0, ind = 0;
            #ifdef USING_HASHVEC
            			int size_vj0 = size_v[j] - 1;
            #endif
            			Float wjk = 0.0;
            			auto vj = v[j];
            			for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end();
            					it2++) {
            				k = *(it2);
            #ifdef USING_HASHVEC
            				int index_v = 0;
            				find_index(vj, index_v, k, size_vj0, hashindices);
            				wjk = vj[index_v].second.second;
            #else
            				wjk = vj[k].second;
            #endif
                            if (wjk == 0.0 || inside[k]){
                                *it2=*(wjS.end()-1);
                                wjS.erase(wjS.end()-1);
                                it2--;
                                continue;
                            }
                            inside[k] = true;
            				prod_cache[k] += wjk * xij;
            			}
            			for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end(); it2++){
                            inside[*it2] = false;
                        }
            		}
            	// now check if all the constraints are satisfied
            		psi=0;
            		flag=true;
            		min_score=9999;
            		max_score=-9999;
            		for(auto elem:cons[I][0].act_k_pos_index){
            			Labels* vec=elem.first;
            			scores=0;
            			for(auto k:*vec){
            				scores+= prod_cache[k];
            			}
            			min_score=min_score<scores?min_score:scores;
            		}
            		for(auto elem:cons[I][0].act_k_neg_index){
            			Labels* vec=elem.first;
            			scores=0;
            			for(auto k:*vec){
            				scores+= prod_cache[k];
            			}
            			max_score=max_score>scores?max_score:scores;
            		}
 
            		if(min_score>max_score)
            			counts++;
        	}
        	return counts*1.0/N;
        }
        Float dual_obj(){
            Float dual_obj = 0.0;
            memset(inside, false, sizeof(bool)*K);
            for (int J = 0; J < D; J++){
                vector<int>* wJ = w_hash_nnz_index[J];
#ifdef USING_HASHVEC
                pair<int, pair<Float, Float>>* vj = v[J];
                int size_vj = size_v[J];
                int util_vj = util_v[J];
                int size_vj0 = size_vj - 1;
                int index_v = -1;
#else
                pair<Float, Float>* vj = v[J];
#endif

                for (int S = 0; S < split_up_rate; S++){
                    for (vector<int>::iterator it = wJ[S].begin(); it != wJ[S].end(); it++){
                        int k = *it;
                        if (inside[k]){
                            continue;
                        }
                        inside[k] = true;
#ifdef USING_HASHVEC
                        find_index(vj, index_v, k, size_vj0, hashindices);
                        Float wjk = vj[index_v].second.second;
#else
                        Float wjk = vj[k].second;
#endif
                        dual_obj += wjk*wjk;
                    }
                    for (vector<int>::iterator it = wJ[S].begin(); it != wJ[S].end(); it++){
                        int k = *it;
                        inside[k] = false;
                    }
                }
            }
            dual_obj /= 2.0;
            cerr<<dual_obj<<" ";
            Float delta;
            Float alpha_sum=0;
            Float lf;
            for (int i = 0; i < N; i++) {
    			int prec=labels->at(i).size();
                Labels* yi = &(labels->at(i));
    			prec=prec<precision?prec:precision;
    			for(int p=0;p<prec;++p){
					// substract dual variables corresponding to positive labels
    				vector<pair<Labels*,Float>>& act_index = cons[i][p].act_k_pos_index;
					for (vector<pair<Labels*,Float>>::iterator it = act_index.begin();
							it != act_index.end(); it++) {
						Float alpha_ik = it->second;
						dual_obj -= alpha_ik;
                        alpha_sum+= alpha_ik;
					}
                    // add dual variables corresponding to negative labels
                    lf= 1.0/(p+1);
                    vector<pair<Labels*,Float>>& act_index2 = cons[i][p].act_k_neg_index;
                    for (vector<pair<Labels*,Float>>::iterator it = act_index2.begin();
                            it != act_index2.end(); it++) {
                        Float alpha_ik = it->second;
                        delta=0;
                        for(auto label: *(it->first)){
                            if(find(yi->begin(),yi->end(),label)!= yi->end())
                                delta+=lf;
                        }
                        dual_obj += delta*alpha_ik;
                        alpha_sum+= delta*alpha_ik;
                    }

    			}
    		}
            cerr<<alpha_sum<<" ";
            return dual_obj;
        }
        /**
        * Function that uses conditional gradient decent with line search for step size to update alpha parameters
        **/
        void subSolve4(int I,
                vector<pair<Labels*,Float>>& act_k_neg_index,
                Float* alpha_i_new_neg, int prec) {

            Labels* yi = &(labels->at(I));
            SparseVec* xi = data->at(I);
            int n = act_k_neg_index.size();
            Float* d = new Float[n];
            Float* gradients= new Float[n];
            Float* margin= new Float[n];
            // because I have not explicitly added bias in negatives for xi >= 0 constraint
            int min_index=-1,min_value=0;
            // initialize with all zeros
            memset(d,0.0, sizeof(Float)*n);
            memset(gradients,0.0, sizeof(Float)*n);
            memset(margin,0.0, sizeof(Float)*n);
            // compute gradients
            int i = 0;
            Float lf=1.0/(prec);
            for (vector<pair<Labels*, Float>>::iterator it =
                    act_k_neg_index.begin(); it != act_k_neg_index.end(); it++) {
                Labels* k = it->first;
                Float alpha_ik = it->second;
                gradients[i] =0 ;
                for (Labels::iterator it = k->begin();
                        it != k->end(); ++it) {
                    if(find(yi->begin(),yi->end(),*it)!= yi->end()){
                        margin[i]+=lf;
                        gradients[i]+=lf;
                    }
                    gradients[i] -= prod_cache[*it] ;
                }
                ++i;
            }
            // find the min element and its index + compute inner product simultaneously
            Float inner_product=0.0;
            for(int j=0;j<n;++j){
                if(gradients[j]<min_value){
                    min_value=gradients[j];
                    min_index=j;
                }
                inner_product+= gradients[j]* (-1*act_k_neg_index[j].second);
            	d[j]=-act_k_neg_index[j].second;
            }
            if(min_index!=-1){
            // this means there is some alpha which has negative gradient
            // use seperate costs for different precisions
	            inner_product+= C[prec-1]*gradients[min_index];
	            d[min_index]+=C[prec-1];
			}
            //compute the new alpha values
            //need to compute the step size using line search
            Float step_size=1.0; // initial step size
            Float tau=0.5, c=0.5; // search control parameters
            Float m=0,t;
            m= inner_product;  // m= d^T\delta f
            t=-c*m;
            Float changes=0;
            Float delta_alpha;
            Float *alpha_new;
            // search for the step_size
            map<int,Float> actives;
            while(true){
                changes=0;
                actives.clear();
                // compute new alphas using current size
                    for (int i = 0; i < n; ++i)
                    {
                        alpha_i_new_neg[i]=step_size*d[i];
                        alpha_i_new_neg[i]+= act_k_neg_index[i].second;
                        delta_alpha=alpha_i_new_neg[i]-act_k_neg_index[i].second;
                        changes-=margin[i]*delta_alpha;
                        // push this info into a map
                        // iterate over labels corresponding to this alpha
                        // need to push -ve to tell that it should be substracted
                        for(auto it:*(act_k_neg_index[i].first)){
                            if(actives.find(-it-1)==actives.end())
                                actives[-it-1]=0.0;
                            actives[-it-1]+=delta_alpha;
                        }
                    }
                changes+=calculate_change(I,actives);
                if(changes>=step_size*t)
                    break;
                step_size=tau*step_size;
                if(step_size < 1e-8)
                	// early stop
                	break;
            }
            delete[] gradients;
            delete[] d;
            delete[] margin;
        }
        /**
        *	Function to perform gradient descent step using line search (unconstrained)
        **/
        void subSolve7(int I,
                vector<pair<Labels*,Float>>& act_k_neg_index,
                Float* alpha_i_new_neg, int prec) { 

            Labels* yi = &(labels->at(I));
            SparseVec* xi = data->at(I);
            int n = act_k_neg_index.size();
            Float* gradients= new Float[n];
            Float* margin= new Float[n];
            Float* d=new Float[n];
            memset(gradients,0.0, sizeof(Float)*n);
            memset(margin,0.0, sizeof(Float)*n);
            memset(d,0.0, sizeof(Float)*n);

            // compute gradients
            int i = 0;
            // scale the margin accordingly
            // margin= size of current set/ precision being optimized
            Float lf=act_k_neg_index[0].first->size()/(prec);
            for (vector<pair<Labels*, Float>>::iterator it =
                    act_k_neg_index.begin(); it != act_k_neg_index.end(); it++) {
                Labels* k = it->first;
                Float alpha_ik = it->second;
                gradients[i] =0 ;
                gradients[i]-=lf;
                margin[i]+=lf;
                for (Labels::iterator it = k->begin();
                        it != k->end(); ++it) {
                    gradients[i] += prod_cache[*it] ;
                }
                ++i;
            }
            // compute inner product
            Float inner_product=0.0;
            for(int j=0;j<n;++j){
                d[j]= - gradients[j]; 
                inner_product+= gradients[j]* (-gradients[j]);
            }
            //compute the new alpha values
            //need to compute the step size using line search
            Float step_size=1.0; // initial step size
            Float tau=0.5, c=0.5; // search control parameters
            Float m=0,t;
            // descent direction is opposite of gradient
            m= inner_product;  // m= d^T\delta f
            t=-c*m;
            Float changes=0;
            Float delta_alpha;
            Float *alpha_new;
            // search for the step_size
            map<int,Float> actives;
            while(true){
                changes=0;
                actives.clear();
                // compute new alphas using current size
                    for (int i = 0; i < n; ++i)
                    {
                        alpha_i_new_neg[i]=step_size*d[i];
                        alpha_i_new_neg[i]+= act_k_neg_index[i].second;
                        if(alpha_i_new_neg[i] < 0)  
                            alpha_i_new_neg[i]=0;
                        delta_alpha=alpha_i_new_neg[i]-act_k_neg_index[i].second;
                        changes+=margin[i]*delta_alpha;
                        // push this info into a map
                        // iterate over labels corresponding to this alpha
                        for(auto it:*(act_k_neg_index[i].first)){
                            if(actives.find(it+1)==actives.end())
                                actives[it+1]=0.0;
                            actives[it+1]+=delta_alpha;
                        }
                    }
                changes+=calculate_change(I,actives);
                if(changes>=step_size*t)
                    break;
                step_size=tau*step_size;
                if(step_size < 1e-8)
                	// early stop
                	break;
            }
            delete[] d;
            delete[] gradients;
            delete[] margin;
        }
        /**
        *	Function returns the change in dual objective without changing the parameters 
        *	@param map: storing the changes in the alpha values for labels
        *	@returns change in quadratic part of dual (old - previous)
        **/
        Float calculate_change(int I, map<int,Float>& actives){
        	SparseVec* x_i = data->at(I);
#ifdef USING_HASHVEC
                    int index_alpha = 0, index_v = 0;
#endif

        	Float changes=0;
        			for(SparseVec::iterator it=x_i->begin(); it!=x_i->end(); it++){
                        int J = it->first; 
                        Float f_val = it->second;
                        vector<int>* wJ = w_hash_nnz_index[J];
#ifdef USING_HASHVEC
                        pair<int, pair<Float, Float>>* vj = v[J];
                        int size_vj = size_v[J];
                        int util_vj = util_v[J];
                        int size_vj0 = size_vj - 1;
                        int ind = 0;
    					for (std::map<int, Float>::iterator it2 = actives.begin();
    							it2 != actives.end(); it2++) {
    						int k = it2->first;
    						Float delta_alpha = it2->second;
    						if(k<0){
    							k=-k;
    							delta_alpha=-delta_alpha;
    						}
    						k--;
                            if( fabs(delta_alpha) < EPS )
                                continue;
                            //update v, w
                            find_index(vj, index_v, k, size_vj0, hashindices);
                            Float vjk = vj[index_v].second.first + f_val*delta_alpha;
                            Float wjk_old = vj[index_v].second.second;
                            Float wjk = prox_l1(vjk, lambda);
                            Float dw= wjk-wjk_old;
    						changes+= dw*dw + 2* dw * wjk_old;
                        }
#else
    					pair<Float, Float>* vj = v[J];
    					int ind = 0;
    					int k;
    					for (std::map<int, Float>::iterator it2 = actives.begin();
    							it2 != actives.end(); it2++) {
    						k = it2->first;
    						Float delta_alpha = it2->second;
    						if(k<0){
    						    k=-k;
    						    delta_alpha=-delta_alpha;
    						}
                            k--;
    						if (fabs(delta_alpha) < EPS)
    							continue;
    						//update v, w
    						pair<Float, Float> vjk_wjk = vj[k];
    						Float vjk = vjk_wjk.first + f_val * delta_alpha;
    						Float wjk = prox_l1(vjk, lambda);

    						Float wjk_old = vjk_wjk.second;
    						Float dw= wjk-wjk_old;
    						changes+= dw*dw + 2* dw * wjk_old;
    					}
#endif
                    }
            return -changes;
        }
        void search_active_i_graph(int I, constraints *cons, int prec) {
            		//prod_cache should be all zero
            		//select one area from {0, ..., split_up_rate-1}
            		int S = rand() % split_up_rate;
            		//compute <xi,wk> for k=1...K
            		Labels* yi = &(labels->at(I));
            		SparseVec* xi = data->at(I);
            		// calculate the scores for each class
            		memset(prod_cache, 0.0, sizeof(Float) * K);
            		for (SparseVec::iterator current_index = xi->begin();
            				current_index < xi->end(); current_index++) {
            			Float xij = current_index->second;
            			int j = current_index->first;
            			vector<int>& wjS = w_hash_nnz_index[j][S];
            			if (wjS.size() == 0)
            				continue;
            			int k = 0, ind = 0;
            #ifdef USING_HASHVEC
            			int size_vj0 = size_v[j] - 1;
            #endif
            			Float wjk = 0.0;
            			auto vj = v[j];
            			for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end();
            					it2++) {
            				k = *(it2);
            #ifdef USING_HASHVEC
            				int index_v = 0;
            				find_index(vj, index_v, k, size_vj0, hashindices);
            				wjk = vj[index_v].second.second;
            #else
            				wjk = vj[k].second;
            #endif
                            if (wjk == 0.0 || inside[k]){
                                *it2=*(wjS.end()-1);
                                wjS.erase(wjS.end()-1);
                                it2--;
                                continue;
                            }
                            inside[k] = true;
            				prod_cache[k] += wjk * xij;
            			}
            			for (vector<int>::iterator it2 = wjS.begin(); it2 != wjS.end(); it2++){
                            inside[*it2] = false;
                        }

            		}
            		int k=precision<yi->size()?precision:yi->size();
            		int *max_indices;
                    // now find  k(precision@k) maximums from negative ones
                    max_indices=new int[K];
                    for(int i=0;i<K;++i){
                    	max_indices[i]=i;
                    }
                    // create a sets to hold the hash current hashes
                    unordered_set<long long> actives[prec];
                    //now compute hashes of all the sets in active set
                    hasher gethash;
                    int max_elements=-1;
                    for(int p=0;p<prec;++p){
						for (vector<pair<Labels*, Float>>::iterator it =
								cons[p].act_k_neg_index.begin();
								it != cons[p].act_k_neg_index.end(); ++it) {
							long long hash=gethash(*(it->first));
							actives[p].insert(hash);
						}
						int total_negs=(p+1)*cons[p].act_k_neg_index.size();
						max_elements=max(max_elements,total_negs);
						for (vector<pair<Labels*, Float>>::iterator it =
								cons[p].act_k_pos_index.begin();
								it != cons[p].act_k_pos_index.end(); ++it) {
							long long hash=gethash(*(it->first));
							actives[p].insert(hash);
						}
                    }
                    // now need to partial sort on the basis of scores
                    int partial_length=yi->size()+ max_elements +max_select;
                    partial_length= partial_length<K? partial_length:K;
            		nth_element(max_indices, max_indices+partial_length, max_indices+K, ScoreComp(prod_cache));
            		sort(max_indices, max_indices+partial_length, ScoreComp(prod_cache));

            		// need flags to indicate set has been added already
            		vector<int> need(prec,max_select);
                    // now declare vectors to store incremental sets
            		bool terminate=false;
                    vector<vector<int>> level[k];
                    for(int i=0;i<partial_length && ! terminate;++i){
                        // iterate from last level to second
                        for(int j=k-1;j>0;--j){
                            //pick elements to j-1 level and append to each j-1 level a[i]
                            for(auto elem:level[j-1]){
                                // create a new vector and add it to level j
                                vector<int> newconfig(elem.begin(),elem.end());
                                newconfig.push_back(max_indices[i]);
                                // run the last iteration of insertion sort
                                // required by hasher that input is sorted
                                int l= newconfig.size()-2;
                                while(l>=0 && newconfig[l]>newconfig[l+1]){
                                	std::swap(newconfig[l],newconfig[l+1]);
                                	l--;
                                }
                                // only add if its not already found
                                if(need[j]!=0){
                                    //check if this new configuration is already in set
                                    // if not then add and break
                                    long long hash= gethash(newconfig);
                                    if(actives[j].find(hash)==actives[j].end()){
                                        Labels* ybar=new vector<int>(newconfig.begin(),newconfig.end());
                                        // add this to active set
                                        cons[j].act_k_neg_index.push_back(make_pair(ybar,0.0));
                                        need[j]--;
                                        break;
                                    }
                                    // else do nothing
                                }
                                level[j].push_back(newconfig);
                            }
                        }
                        // add these to first level
                        vector<int> newconfig(1,max_indices[i]);
                        if(need[0]!=0){
                            long long hash= gethash(newconfig);
                                if(actives[0].find(hash)==actives[0].end()){
                                    Labels* ybar=new vector<int>(newconfig.begin(),newconfig.end());
                                        // add this to active set
                                    cons[0].act_k_neg_index.push_back(make_pair(ybar,0.0));
                                    need[0]--;
                                    break;
                                }
                        }
                        level[0].push_back(newconfig);
                        // check if all the sets of size 1-k have been found
                        terminate=true;
                        for(auto e:need)
                        	terminate &= e==0;
                    }
        }
        //store the best model as well as necessary indices
        void store_best_model(){
#ifdef USING_HASHVEC
            memset(inside, false, sizeof(bool)*K);
            for (int j = 0; j < D; j++){
                size_w[j] = 1;
                int total_size = 0;
                for (int S = 0; S < split_up_rate; S++){
                    total_size+=w_hash_nnz_index[j][S].size();
                }
                while (size_w[j] * UPPER_UTIL_RATE < total_size)
                    size_w[j] *= 2;
                delete[] w[j];
                w[j] = new pair<int, Float>[size_w[j]];
                non_split_index[j].clear();
                for(int it = 0; it < size_w[j]; it++)
                    w[j][it] = make_pair(-1, 0.0);
                memset(inside, false, sizeof(bool)*K);
                memset(norms, 0.0, sizeof(Float)*K);
 
                for(int S=0;S<split_up_rate;S++){
                    for (vector<int>::iterator it=w_hash_nnz_index[j][S].begin(); it!=w_hash_nnz_index[j][S].end(); it++){
                        int k = *it;
                        int index_v = 0;
                        find_index(v[j], index_v, k, size_v[j]-1, hashindices);
                        if (fabs(v[j][index_v].second.second) > 1e-12 && !inside[k]){
                            inside[k] = true;
                            int index_w = 0;
                            find_index(w[j], index_w, k, size_w[j]-1, hashindices);
                            w[j][index_w] = make_pair(k, v[j][index_v].second.second);
                            non_split_index[j].push_back(k);
                        }
                    }
                }
                //recover inside, avoid any complexity related to K
                for (vector<int>::iterator it=non_split_index[j].begin(); it != non_split_index[j].end(); it++){
                    int k = *it;
                    inside[k] = false;
                }
            }
            best_model = new Model(train, non_split_index, w, size_w, hashindices);
#else
            for (int j = 0; j < D; j++){
                for (vector<int>::iterator it = non_split_index[j].begin(); it != non_split_index[j].end(); it++){
                    w[j][*it] = 0.0;
                }
            }
            memset(inside, false, sizeof(bool)*K);
            memset(norms, 0.0, sizeof(Float)*K);
            for (int j = 0; j < D; j++){
                non_split_index[j].clear();
                pair<Float, Float>* vj = v[j];
                for(int S=0;S<split_up_rate;S++){
                    for (vector<int>::iterator it=w_hash_nnz_index[j][S].begin(); it!=w_hash_nnz_index[j][S].end(); it++){
                        int k = *it;
                        if (fabs(vj[k].second) > EPS && !inside[k]){
                            w[j][k] = vj[k].second;
                            norms[k]+= fabs(w[j][k]);
                            inside[k] = true;
                            non_split_index[j].push_back(k);
                        }
                    }
                }
                //recover inside, avoid any complexity related to K
                for (vector<int>::iterator it=non_split_index[j].begin(); it != non_split_index[j].end(); it++){
                    int k = *it;
                    inside[k] = false;
                }
            }
            best_model = new Model(train, non_split_index, w);
#endif		
/*
            if (best_act_k_index == NULL)
                best_act_k_index = new vector<pair<int, Float>>[N];
            for (int i = 0; i < N; i++)
                best_act_k_index[i] = act_k_index[i];
*/
            //store the constraints
            ofstream fout("constraints.txt");
            // Each line contains the labels for that particular example which are part of active set
            for(int i=0;i<N;++i){
            	// first write the positive labels then negative
            	for(auto elem:cons[i][0].act_k_pos_index){
            		for(auto label: *(elem.first))
            			fout<<label<<" ";
            	}
            	// now write the negative labels then negative
            	for(auto elem:cons[i][0].act_k_neg_index){
            		for(auto label: *(elem.first))
            			fout<<label<<" ";
            	}
            	fout<<"-1"<<endl; // to mark end
            }
            fout.close();
        }

    private:

        double best_heldout_acc = -1.0;
        Problem* train;
        HeldoutEval* heldoutEval;
        Float lambda;
        Float *C;
        vector<SparseVec*>* data;
        vector<Labels>* labels;
        int D; 
        int N;
        int K;
        // precision@k to optimize for
        int precision;
        int solver;
        Float* Q_diag;
        HashClass* hashfunc;
        vector<Float>* cdf_sum;
        HashVec** w_temp;
        vector<int>** w_hash_nnz_index;

        int max_iter;
        vector<int>* k_index;

        //sampling 
        bool* inside_index;
        Float* prod_cache;
        bool using_importance_sampling;
        int max_select;
        int speed_up_rate, split_up_rate;

        //global cache
        bool* inside;
        Float* prod;
        Float* norms;
        int* freq;
        //heldout options
        int early_terminate;

    public:

        //useful statistics
        Float nnz_w_j = 1.0;
        Float nnz_a_i = 1.0;
        Float d = 1.0;
        Float decay_rate;

        //(index, val) representation of alpha
        //vector<pair<int, Float>>* act_k_index;
        // separate for positive and negative part of constraints
        constraints **cons;
        //for storing best model
        //vector<pair<int, Float>>* best_act_k_index;
        vector<int>* non_split_index;
        Model* best_model = NULL;
        vector<int>** best_w_hash_nnz_index;

        //iterations used so far	
        int iter;
        //a random permutation stored in public
        int* hashindices;

        bool dump_model = false;
        string modelFname;


#ifdef USING_HASHVEC
        pair<int, Float>** w;
        int* size_w;
        pair<int, pair<Float, Float> >** v;
        pair<int, pair<Float, Float> >** best_v;  
        int* size_v;
        int* util_v;
#else
        Float** w;
        pair<Float, Float>** v;
        pair<Float, Float>** best_v;
#endif
};
