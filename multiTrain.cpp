#include "util.h"
#include "multi.h"
#include "SplitOracleActBCD.h"

double overall_time = 0.0;

void exit_with_help(){
	#ifdef USING_HASHVEC
	cerr << "Usage: ./multiTrainHash (options) [train_data] (model)" << endl;
	#else
	cerr << "Usage: ./multiTrain (options) [train_data] (model)" << endl;	
	#endif
	cerr << "options:" << endl;
	cerr << "-s solver: (default 2)" << endl;
	cerr << "	1 -- Stochastic-Active Block Coordinate Descent (Projected Gradient Descent)" << endl;
	cerr << "	2 -- Stochastic-Active Block Coordinate Descent (Frank Wolfe)" << endl;
	cerr << "	3 -- Stochastic-Active Block Coordinate Descent (Away Step Frank Wolfe)" << endl;
	cerr << "	4 -- Stochastic-Active Block Coordinate Descent (Pair Wise Frank Wolfe)" << endl;

	cerr << "-l lambda: L1 regularization weight (default 0.1)" << endl;
	cerr << "-D decay: decay rate for step size (default 0.01)" << endl;
	cerr << "-k precision@k: For Training  (default 1)" << endl;
	cerr << "-E weights for edges  (default 1.0)" << endl;


	cerr << "-c cost: cost of each sample (default 1.0)" << endl;
	cerr << "-r speed_up_rate: sample 1/r fraction of non-zero features to estimate gradient (default r = ceil(min( 5DK/(Clog(K)nnz(X)), nnz(X)/(5N) )) )" << endl;
	cerr << "-q split_up_rate: divide all classes into q disjoint subsets (default 1)" << endl;
	cerr << "-m max_iter: maximum number of iterations allowed if -h not used (default 50)" << endl;
	cerr << "-u uniform_sampling: use uniform sampling instead of importance sampling (default not)" << endl;
	cerr << "-g max_select: maximum number of dual variables selected during search (default: -1 (i.e. dynamically adjusted during iterations) )" << endl;
	cerr << "-p post_train_iter: #iter of post-training without L1R (default auto)" << endl;
	cerr << "-h <file>: using accuracy on heldout file '<file>' to terminate iterations" << endl;
	cerr << "-T <file>: Embeddings File" << endl;
	cerr << "-e early_terminate: how many iterations of non-increasing heldout accuracy required to earyly stop (default 3)" << endl;
	cerr << "-d : dump model file when better heldout accuracy is achieved, model files will have name (model).<iter>" << endl;
	exit(0);
}

void parse_cmd_line(int argc, char** argv, Param* param){

	int i;
	for(i=1;i<argc;i++){
		if( argv[i][0] != '-' )
			break;
		if( ++i >= argc )
			exit_with_help();

		switch(argv[i-1][1]){
			
			case 's': param->solver = atoi(argv[i]);
				  break;
			case 'l': param->lambda = atof(argv[i]);
				  break;
			case 'D': param->decay = atof(argv[i]);
				  break;
				  // special case read elements until the precison counts
			case 'c': delete[] param->C;
					  param->C = new Float[param->precision]; 
					  for(int p=0;p< param->precision;++p)
					  	param->C[p]=atof(argv[i++]);
					  i--;
				  break;
			case 'E': param->C2 = atof(argv[i]);
				  break;
			case 'r': param->speed_up_rate = atoi(argv[i]);
				  break;
			case 'k': param->precision = atoi(argv[i]);
				  break;
			case 'q': param->split_up_rate = atoi(argv[i]);
				  break;
			case 'm': param->max_iter = atoi(argv[i]);
				  break;
			case 'u': param->using_importance_sampling = false; --i;
				  break;
			case 'g': param->max_select = atoi(argv[i]);
				  break;
			case 'p': param->post_solve_iter = atoi(argv[i]);
				  break;
			case 'e': param->early_terminate = atoi(argv[i]);
				  break;
			case 'h': param->heldoutFname = argv[i];
				  param->max_iter=INF;
				  break;
			case 'T': param->embeddingFname = argv[i];
				  break;

			case 'd': param->dump_model = true; --i;
				  break;
			default:
				  cerr << "unknown option: -" << argv[i-1][1] << endl;
				  exit(0);
		}
	}

	if(i>=argc)
		exit_with_help();

	param->trainFname = argv[i];
	i++;

	if( i<argc )
		param->modelFname = argv[i];
	else{
		param->modelFname = new char[FNAME_LEN];
		strcpy(param->modelFname,"model");
	}
}


int main(int argc, char** argv){
	
	auto time_null = time(NULL);
	cerr << "random seed: " << time_null << endl;
	srand(time_null);
	
	Param* param = new Param();
	parse_cmd_line(argc, argv, param);
	Problem* train = new Problem();
	readData( param->trainFname, train);
	param->train = train;
	
	overall_time -= omp_get_wtime();

	if (param->heldoutFname != NULL){
		Problem* heldout = new Problem();
		readData( param->heldoutFname, heldout);
		cerr << "heldout N=" << heldout->data.size() << endl;
		param->heldoutEval = new HeldoutEval(heldout,1);
	}
	int D = train->D;
	int K = train->K;
	int N = train->data.size();
	cerr << "N=" << N << endl;
	cerr << "d=" << (Float)nnz(train->data)/N << endl;
	cerr << "D=" << D << endl; 
	cerr << "K=" << K << endl;	
	SplitOracleActBCD* solver = new SplitOracleActBCD(param);
	Model* model = solver->solve();
	model->writeModel(param->modelFname);
	
	
	overall_time += omp_get_wtime();
	cerr << "overall_time=" << overall_time << endl;
}
