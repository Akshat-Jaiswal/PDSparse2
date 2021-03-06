#ifndef UTIL
#define UTIL

#include<cmath>
#include<vector>
#include<map>
#include<string>
#include<cstring>
#include<stdlib.h>
#include<fstream>
#include<iostream>
#include<algorithm>
#include<omp.h>
#include<unordered_map>
#include<time.h>
#include<tuple>
#include<cassert>
#include<limits.h>
using namespace std;

typedef vector<pair<int,double> > SparseVec;
typedef unordered_map<int,double> HashVec;
typedef vector<int> Labels;
typedef double Float;
const int LINE_LEN = 100000000;
const int FNAME_LEN = 1000;
const int INF = INT_MAX;

#define EPS 1e-12
#define INFI 1e10
#define INIT_SIZE 16
#define PermutationHash HashClass
#define UPPER_UTIL_RATE 0.75
#define LOWER_UTIL_RATE 0.5

class ScoreComp{
	
	public:
	ScoreComp(Float* _score, bool _desc=true){
		score = _score;
		desc=_desc;
	}
	bool operator()(const int& ind1, const int& ind2){
		if(desc)
			return score[ind1] > score[ind2];
		else
			return score[ind1] < score[ind2];
	}
	private:
	Float* score;
	bool desc;
};

class PermutationHash{
	public:
	PermutationHash(){};
	PermutationHash(int _K){	
		K = _K;
		hashindices = new int[K];
		for (int i = 0; i < K; i++){
			hashindices[i] = i;
		}
		random_shuffle(hashindices, hashindices+K);
	}
	~PermutationHash(){
		delete [] hashindices;
	}
	int* hashindices;
	private:
	int K;
};

vector<string> split(string str, string pattern){

	vector<string> str_split;
	size_t i=0;
	size_t index=0;
	while( index != string::npos ){

		index = str.find(pattern,i);
		str_split.push_back(str.substr(i,index-i));

		i = index+1;
	}
	
	if( str_split.back()=="" )
		str_split.pop_back();

	return str_split;
}

double inner_prod(double* w, SparseVec* sv){

	double sum = 0.0;
	for(SparseVec::iterator it=sv->begin(); it!=sv->end(); it++)
		sum += w[it->first]*it->second;
	return sum;
}
// expects x not equal to y
int get_edge_index(int x, int y){
	// make sure that x always contains max of both
	if(x<=y)
		std::swap(x,y);
	return (x*(x-1)/2)+y;
}
double prox_l1_nneg( double v, double lambda ){
	
	if( v < lambda )
		return 0.0;

	return v-lambda;
}
inline Float prox_l1( Float v, Float lambda ){
	
	if( fabs(v) > lambda ){
		if( v>0.0 )
			return v - lambda;
		else 
			return v + lambda;
	}
	
	return 0.0;
}

void project_to_simplex(Float* x,Float* y, int D, Float C){
	int* index_u = new int[D];
	bool flag=true;
	Float cumsum=0;
	for (int i = 0; i < D; i++){
		index_u[i] = i;
		flag &= y[i]>=0;
		cumsum+=y[i];
		x[i]=y[i];
	}
	if(cumsum<=C && flag)
		return;
	sort(index_u, index_u+D, ScoreComp(y));
	int p=-1;
	Float score, max_score;
	Float cur_sum=0, lambda=0;
	for(int j=0;j<D;++j){
		cur_sum+=y[index_u[j]];
		score=y[index_u[j]]+(1.0/(j+1)*(C- cur_sum));
		if(score>0){
			p=j;
			lambda= score - y[index_u[p]];
		}
	}
	for(int i=0;i<D;++i){
		x[i]=y[i]+lambda;
		if(x[i]<0) x[i]=0;
	}
	delete[] index_u;
}
/*
 * Assuming y,ybar are sorted and indexing begins at 1
 */
Labels* diff_merge(Labels &y, Labels &ybar){
	Labels::iterator it1,it2;
	it1=y.begin();
	it2=ybar.begin();
	Labels *out=new Labels();
	while(it1!= y.end() && it2!=ybar.end()){
		if(*it1==*it2)
			++it1,++it2;
		else if(*it1<*it2){
			out->push_back((*it1)+1);
			it1++;
		}else{
			out->push_back((-*it2)-1);
			it2++;
		}
	}
	while(it1!=y.end()){
		out->push_back((*it1)+1);
		it1++;
	}
	while(it2!=ybar.end()){
			out->push_back(-(*it2)-1);
			it2++;
	}
	return out;
}
int sign(int &a){
	if (a<0)
		return -1;
	else
		return 1;
}
double norm_sq( double* v, int size ){

	double sum = 0.0;
	for(int i=0;i<size;i++){
		if( v[i] != 0.0 )
			sum += v[i]*v[i];
	}
	return sum;
}

/*
 * Function to compare if two combinations of labels are equal
 * Requires both of them to be sorted (in same order)
 * @param y: input vector<int>
 * @param ybar: input vector<int>
 * @return true, if y==ybar
 * 		   false, otherwise
 */
bool compare_combinations(Labels* y, Labels* ybar){
	if(y->size()!=ybar->size())
		return false;
	int it1=0;
	while(it1<y->size()){
		if(ybar->at(it1)!=y->at(it1))
			return false;
		it1++;
	}
	return true;
}
class hasher{
	public:
	std::size_t operator()(std::vector<int> &vec) const{
		  std::size_t seed = vec.size();
		  for(vector<int>::iterator i =vec.begin();i!=vec.end();++i) {
		    seed ^= *i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
		  }
		  return seed;
	}
};
void display(std::vector<int> const &vec){
	cout<<" (";
	for(auto elem: vec)
		cout<<elem<<" ";
	cout<<") ";
}
template <class RandomAccessIterator>
void insertion_sort (RandomAccessIterator first, RandomAccessIterator last){
   RandomAccessIterator i,j;
   for (i = first; i != last; i++)
   {
       j = i-1; 
       while (j >= first && *(j+1)<*j)
       {
			iter_swap(j,j+1);  
			j=j-1;
       }
   }
}
template <class RandomAccessIterator,class Compare>
void insertion_sort (RandomAccessIterator first, RandomAccessIterator last, Compare comp){
   RandomAccessIterator i,j;
   for (i = first; i != last; i++)
   {
       j = i-1; 
       while (j >= first && comp(*(j+1),*j))
       {
			iter_swap(j,j+1);  
			j=j-1;
       }
   }
}
/*
 * Generate all possible combinations of size k from n
 * @param ans: vector of vectors to store the subsets of size k
 * @param tmp: vector<int> to store the current subset (must be initially empty )
 * @param left: index to start from
 * @param k: size of the desired subset size
 * @param yi: vector<int>* set from which k size subsets are to be chosen
 * @return None
 */
void makeCombiUtil(vector<pair<Labels*,Float>>& ans,
    vector<int>& tmp, int left, int k, Labels* yi)
{
    // Pushing this vector to a vector of vector
    if (k == 0) {
    	Labels *ybar= new Labels(tmp.begin(),tmp.end());
        ans.push_back(make_pair(ybar,0.0));
        return;
    }

    // i iterates from left to n. First time
    // left will be 1
    for (int i = left; i < yi->size(); ++i)
    {
        tmp.push_back(yi->at(i));
        makeCombiUtil(ans, tmp, i + 1, k - 1, yi);

        // Popping out last inserted element
        // from the vector
        tmp.pop_back();
    }
}

int total_size( vector<int>* alpha, int size ){
	
	int sum = 0;
	for(int i=0;i<size;i++)
		sum += alpha[i].size();
	return sum;
}

int total_size( HashVec** w, int size ){
	
	int sum = 0;
	for(int j=0;j<size;j++)
		sum += w[j]->size();
	return sum;
}

long nnz( vector<SparseVec*>& data ){
	
	long sum =0;
	for(int i=0;i<data.size();i++){
		sum += data[i]->size();
	}
	return sum;
}

// maintain top tK indices, stored in max_indices, where indices are sorted by x[].
// Here the situation is x(i) has just been updated, where i may or may not exist in max_indices
inline bool update_max_indices(int* max_indices, Float* x, int i, int tK){
	//max_indices should have size tK+1
	int ind = 0;
	// entry ind is empty if max_indices[ind] == -1
	while (ind < tK && max_indices[ind] != -1 && max_indices[ind] != i){
		ind++;
	}
    bool adding_new_index = true;
	if (ind < tK && max_indices[ind] == i)
		adding_new_index = false;
	max_indices[ind] = i;
	int k = 0;
	//try move to right
	while (ind < tK-1 && max_indices[ind+1] != -1 && x[max_indices[ind+1]] > x[max_indices[ind]]){
                k = max_indices[ind];
                max_indices[ind] = max_indices[ind+1];
                max_indices[++ind] = k;
        }
	//try move to left
	while (ind > 0 && x[max_indices[ind]] > x[max_indices[ind-1]]){
		k = max_indices[ind];
		max_indices[ind] = max_indices[ind-1];
		max_indices[--ind] = k;
	}
	return adding_new_index;
}

//min_{x,y} \|x - b\|^2 + \|y - c\|^2
// s.t. x >= 0, y >= 0
//  \|x\|_1 = \|y\|_1 = t \in [0, C]
// x,b \in R^n; y,c \in R^m
// O( (m + n) log(m+n) ), but usually dominated by complexity of computing b, c
inline void solve_bi_simplex(int n, int m, Float* b, Float* c, Float C, Float* x, Float* y){
	int* index_b = new int[n];
	int* index_c = new int[m];
	for (int i = 0; i < n; i++)
		index_b[i] = i;
	for (int j = 0; j < m; j++)
		index_c[j] = j;
	sort(index_b, index_b+n, ScoreComp(b));
	sort(index_c, index_c+m, ScoreComp(c));
	Float* S_b = new Float[n];
	Float* S_c = new Float[m];
	Float* D_b = new Float[n+1];
	Float* D_c = new Float[m+1];
	Float r_b = 0.0, r_c = 0.0;
	for (int i = 0; i < n; i++){
		r_b += b[index_b[i]]*b[index_b[i]];
		if (i == 0)
			S_b[i] = b[index_b[i]];
		else
			S_b[i] = S_b[i-1] + b[index_b[i]];
		D_b[i] = S_b[i] - (i+1)*b[index_b[i]];
	}
	D_b[n] = C;
	for (int j = 0; j < m; j++){
		r_c += c[index_c[j]]*c[index_c[j]];
		if (j == 0)
			S_c[j] = c[index_c[j]];
		else
			S_c[j] = S_c[j-1] + c[index_c[j]];
		D_c[j] = S_c[j] - (j+1)*c[index_c[j]];
	}
	D_c[m] = C;
	int i = 0, j = 0;
	//update for b_{0..i-1} c_{0..j-1}
	//i,j is the indices of coordinate that we will going to include, but not now!
	Float t = 0.0;
	Float ans_t_star = 0;
	Float ans = INFI;
	int ansi = i, ansj = j;
	int lasti = 0, lastj = 0;
	do{
		lasti = i; lastj = j;
		Float l = t;
		t = min(D_b[i+1], D_c[j+1]);
		//now allowed to use 0..i, 0..j
		if (l >= C && t > C){
			break;
		}
		if (t > C) { 
			t = C;
		}
		Float t_star = ((i+1)*S_c[j] + (1+j)*S_b[i])/(i+j+2);
		//cerr << "getting t_star=" << t_star << endl;
		if (t_star < l){
			t_star = l;
		//	cerr << "truncating t_star=" << l << endl;
		}
		if (t_star > t){
			t_star = t;
		//	cerr << "truncating t_star=" << t << endl;
		}
		Float candidate = r_b + r_c + (S_b[i] - t_star)*(S_b[i] - t_star)/(i+1) + (S_c[j] - t_star)*(S_c[j] - t_star)/(j+1);
		//cerr << "candidate val=" << candidate << endl;
		if (candidate < ans){
			ans = candidate;
			ansi = i;
			ansj = j;
			ans_t_star = t_star;
		}
		while ((i + 1)< n && D_b[i+1] <= t){
			i++;
			r_b -= b[index_b[i]]*b[index_b[i]];
		}
		//cerr << "updating i to " << i << endl;
		while ((j+1) < m && D_c[j+1] <= t) {
			j++;
			r_c -= c[index_c[j]]*c[index_c[j]];
		}
		//cerr << "updating j to " << j << endl;
	} while (i != lasti || j != lastj);
	//cerr << "ansi=" << ansi << ", ansj=" << ansj << ", t_star=" << ans_t_star << endl;
	for(i = 0; i < n; i++){
		int ii = index_b[i];
		if (i <= ansi)
			x[ii] = (b[index_b[i]] + (ans_t_star - S_b[ansi])/(ansi+1));
		else
			x[ii] = 0.0;
	}
	for(j = 0; j < m; j++){
		int jj = index_c[j];
		if (j <= ansj)
			y[jj] = c[index_c[j]] + (ans_t_star - S_c[ansj])/(ansj+1);
		else
			y[jj] = 0.0;
	}

	delete[] S_b; delete[] S_c;
	delete[] index_b; delete[] index_c;
	delete[] D_b; delete[] D_c;
}

#endif
