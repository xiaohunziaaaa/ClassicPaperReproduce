#include "SVM.h"
#include <algorithm>
#include <cstring>
#include <stdlib.h>
#include <iomanip>

SVM::SVM(){
    
}

SVM::SVM(SVMopt* _svmopt){
    this->_C = _svmopt->_C;                    
    this->_epsilon = _svmopt->_epsilon;             
    this->_KTYPE = _svmopt->_KTYPE;              
    this->_ftrain = _svmopt->_ftrain;             
    this->_ftest = _svmopt->_ftest;            
    this->_fmodel = _svmopt->_fmodel;

    if(_ftrain!=""){
        istrainmodel = true;
        isloadmodel = false;
    }             
    else{
        istrainmodel = false;
        isloadmodel = true;
    }
}

SVM::~SVM(){

}


int SVM::train(){
    _w.clear();
    _b=0.0;
    _alpha.clear();
    _train_data.clear();

    //read samples into _train_data
    read_data(_ftrain, _train_data);
    _n = _train_data.size();

    _alpha.resize(_n, 0.0);
    _b = 0.0;

    _error_cache.resize(_n);
    //initial error_cache
    for(int k=0; k<_n; k++){
        _error_cache[k] = - _train_data[k].y;
    }


    //int examineAll = 0;
    //int numChanged = _n;                          //the number of alpha changed in each iteration; initialize it to _n in order to begin loop followed
    int examineAll = 1;
    int numChanged = 0;
    //find alpha1 and corresponding index
    //stop util no alpha changed in whole samples in one iteration
    /*
    while(numChanged>0 || examineAll==1){
        numChanged = 0

        //first search alpha1 in support vector 
        if (examineAll==0){
            for(int k=0; k<_n; k++){
                if(_alpha[k] != 0 && _alpha[k] != _C){
                    numChanged += one_step(k)
                }
        }
        else{
             for(int k=0; k<_n; k++){
                numChanged += one_step(k)
             }
             //search alpha1 in support vectors next iteration
             //and if no alpha1 found in whole samples, it will break this loop.
             examineAll = 0;
        }
        //if find no alpha1 in support vectors, then find it in whole samples
        if(examineAll==0 && numChanged==0 ){
            examineAll = 1;
        }
    }*/

    while(numChanged>0 || examineAll){
        numChanged = 0;

        //loop over all training examples to find alpha1
        if (examineAll){
            for(int k=0; k<_n; k++){
                numChanged += examineExample(k);
             }
        }
        //loop over examples where alpha is not 0 or _C
        else{
            for(int k=0; k<_n; k++){
                if(_alpha[k] != 0 && _alpha[k] != _C){
                    numChanged += examineExample(k);
                }
            }
        }
        //if find no alpha1 in support vectors, then find it in whole samples
        if(examineAll == 1 ){
            examineAll = 0;
        }
        else if(numChanged == 0){
            examineAll = 1;
        }
    }

    int dim = _train_data[0].input_dim;
    _w.resize(dim, 0.0);
    for(int i=0; i<dim; i++){
        for(int j=0; j<_n; j++){
            _w[i]+=_alpha[j]*_train_data[j].y*_train_data[j].x[i];
        }
    }

    //calculate object value
    float sum_alpha = 0.0;
    float s = 0.0;
    float t = 0.0;
    float obj = 0.0;
    for (int i = 0; i < _n; i++){
        s += _alpha[i];
    }
        
    for (int i = 0; i < _n; i++){
        for (int j = 0; j < _n; j++){
            t += _alpha[i] * _alpha[j] * _train_data[i].y * _train_data[j].y * kernel(_train_data[i].x, _train_data[j].x, this->_KTYPE);
        }
    }

    obj = sum_alpha - 0.5 * t;
        std::cout << "Objective func : " << obj << "\t\t\t";
    
    //fine tuning
    for (int i = 0; i < _n; i++){
        if (_alpha[i] < 1e-6){
            _alpha[i] = 0.0;
        }
    }

}

int SVM::predict(){
    //load x_data
    read_data(_ftest, _test_data);
    for(int i=0; i<_test_data.size(); i++){
        _test_data[i].y = g(_test_data[i])>0?1:-1;
    }
    write_data(_ftest, _test_data);

}

int SVM::load(){
    std::ifstream f(_fmodel);
    std::string w;
    getline(f, w);
    std::vector<std::string> weight =  split(w, " ");
    for(int i=0; i<weight.size(); i++){
        _w.push_back(atof(weight[i].c_str()));
    }
    return 0;
}

int SVM::dump(){
    std::ofstream f(_fmodel);
    for(int i=0;i<_w.size();i++){
        f<<_w[i]<<" ";
    }
    return 0;
}

float SVM::kernel(sample_x x1, sample_x x2, int type){
    if(x1.size()!=x2.size()){
        return -99999;
    }
    float result = 0.0;
    if(type==0){
        for(int i=0; i<x1.size(); i++){
            result += x1[i]*x2[i];
        }
        return result;
    }
    if(type==1){
        return 0.0;
    }
    if(type==2){
        return 0.0;
    }
}

int SVM::examineExample(int i1){

    float y1 = _train_data[i1].y;
    float _alpha1 = _alpha[i1];
    float E1 = 0.0;

    E1 = _error_cache[i1];

    //fisrt examine whether alpha1 satisfy KKT condition
    float r1 = y1*E1;

    // _alpha1  has three possible values
    // _alpha1 == 0, then check whether kkt >= 0
    // _alpha1 == _C, then check whether kkt <=0
    // 0< _alpha1 <_C, then check whether kkt == 0
    // this three situations can be combined int one
    if((r1 < -TOLERANCE && _alpha1 < _C) || (r1 > TOLERANCE && _alpha1 > 0)){
        //search second alpha to be optimized
        //firstly, search alpha2 which maximize |E1-E2|
       int i2 = -1;
       float temp = 0.0;
       float maxdis = 0.0;
        for(int k=0; k<_n; k++){
            if(_alpha[k] > 0 && _alpha[k] < _C){
                float E2 = _error_cache[k];
                temp = abs(E1-E2);
                if ( temp > maxdis){
                    maxdis = temp;
                    i2 = k;
                }
            }
        }
        if(i2 >= 0){
            if(takeStep(i1, i2)){
                return 1;
            }
        }
            
        //secondly, search alpha2 in support vectors
        for(int k0 = rand()%_n, k = k0; k < _n + k0; k++){
            i2 = k%_n;
            if(_alpha[i2] != 0 && _alpha[i2] != _C){
                if(takeStep(i1, i2)){
                    return 1;
                }
            }
        }

        //thirdly, search alpha2 in all alphas
        for(int k0 = rand()%_n, k = k0; k < _n + k0; k++){
            i2 = k%_n;
            if(takeStep(i1, i2)){
                return 1;
            }
        }
    }

    //if satisfy kkt condition, return 0, no alpha changed in this iteration
    return 0;





}

int SVM::takeStep(int i1, int i2){
    if(i1 == i2){
        return 0;
    }

    float _alpha1 = _alpha[i1];
    float _alpha2 = _alpha[i2];
    float y1 = _train_data[i1].y;
    float y2 = _train_data[i2].y;
    
    float L = 0.0;
    float H = 0.0;
    float eta = 0.0;

    float a2 = 0.0;
    float a1 = 0.0;
    float b1_new = 0.0;
    float b2_new = 0.0;
    float b_new = 0.0;

    float k11 = 0.0;
    float k12 = 0.0;
    float k22 = 0.0;

    float E1 = _error_cache[i1];
    float E2 = _error_cache[i2];


    //calculate Lower Bound and Higher Bound
    if(y1 == y2){
        float temp = _alpha1 + _alpha2;
        if(temp > _C){
            L = temp - _C;
            H = _C;
        }
        else{
            L = 0;
            H = temp;
        }
    }
    else{
        float temp = _alpha1 - _alpha2;
        if (temp > 0){
            L = 0;
            H = _C - temp;
        }
        else{
            L = -temp;
            H = _C;
        }
    }

    if(H-L < 1e-6){
        return 0;
    }
    //update alpha2 and alpha2 
    k11 = kernel(_train_data[i1].x, _train_data[i1].x, this->_KTYPE);
    k12 = kernel(_train_data[i1].x, _train_data[i2].x, this->_KTYPE);
    k22 = kernel(_train_data[i2].x, _train_data[i2].x, this->_KTYPE);
    eta = k11 + k22 - 2*k12;

    if(eta>0){
        a2 = _alpha2 + y2*(E1- E2)/eta;
        if(a2 < L) a2 = L;
        else if( a2 > H) a2 = H;
    }
    else{
        float c1 = eta / 2.0;
        float c2 = y2 * (E1 - E2) - eta * _alpha2;
        float Lobj = c1 * L * L + c2 * L;
        float Hobj = c1 * H * H + c2 * H;
        if (Lobj < Hobj - _epsilon){
            a2 = L;
        }
        else if (Lobj > Hobj + _epsilon){
            a2 = H;
        }
        else{
            a2 = _alpha2;
        }
    }


    // if descend slightly, break and search alpha1 or alpha2
    if(abs(a2 - _alpha2) < _epsilon*(a2 + _alpha2 + _epsilon)){
        return 0;
    }

    a1 = _alpha1 + y1*y2*(_alpha2 - a2);
    float b_old = _b;
    //update b 
    b1_new = E1 + y1*k11*(a1 - _alpha1) + y2*k12*(a2 - _alpha2) + _b;
    b2_new = E2 + y1*k12*(a1 - _alpha1) + y2*k22*(a2 - _alpha2) + _b;
    _b = (b1_new + b2_new)/2.0;
    //update Ei

    //faster than update Ei using g(x)-y
    for(int k=0; k<_n; k++){
        _error_cache[k] += y1*(a1 - _alpha[i1])*kernel(_train_data[k].x, _train_data[i1].x, this->_KTYPE); 
        _error_cache[k] += y2*(a2 - _alpha[i2])*kernel(_train_data[k].x, _train_data[i2].x, this->_KTYPE);
        _error_cache[k] +=  b_old - _b;
    }

    _alpha[i1] = a1;
    _alpha[i2] = a2;
    //for(int k=0; k<_n; k++){
    //    _error_cache[k] = g(_train_data[k]) - _train_data[k].y;
    //}
    return 1;
}

float SVM::g(SAMPLE sample){
    float s = 0.0;
    for (int i = 0; i < _n; i++){
        if (_alpha[i] > 0){
            s += _alpha[i] * _train_data[i].y * kernel(_train_data[i].x, sample.x, this->_KTYPE);
        }
    }
    s -= _b;                                    //Attention, minus here
    return s;
}

int SVM::read_data(std::string ftrain, Data &data){
    std::ifstream f(ftrain);
    std::string temp;
    while(getline(f, temp)){
        std::vector<std::string> oneline  = split(temp, " ");
        SAMPLE sample;
        sample.y = atof(oneline[0].c_str());
        for(int i=1; i<oneline.size(); i++){
            sample.x.push_back(atof(oneline[i].c_str()));
        }
        sample.input_dim = oneline.size() - 1;
        data.push_back(sample);
    }
    f.close();
    return 0;
}

int SVM::write_data(std::string ftest, Data &data){
    std::ofstream f(ftest);
    for(int i=0; i<_test_data.size(); i++){
        f<<_test_data[i].y << " ";
        for(int j=0; j<_test_data[i].x.size(); j++){
            f<<std::setprecision(5)<<_test_data[i].x[j]<<" ";
        }
        f<<std::endl;
    }
    f.close();
}
std::vector<std::string> SVM::split(const std::string& str, const std::string& delim){
	std::vector<std::string> res;
	if("" == str) return res;
	//先将要切割的字符串从string类型转换为char*类型
	char * strs = new char[str.length() + 1] ; //不要忘了
	strcpy(strs, str.c_str()); 
 
	char * d = new char[delim.length() + 1];
	strcpy(d, delim.c_str());
 
	char *p = strtok(strs, d);
	while(p) {
		std::string s = p; //分割得到的字符串转换为string类型
		res.push_back(s); //存入结果数组
		p = strtok(NULL, d);
	}
 
	return res;

}

