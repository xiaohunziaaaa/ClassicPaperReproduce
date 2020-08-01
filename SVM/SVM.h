#ifndef __SVM_H__
#define __SVM_H__
#include <vector>
#include <iostream>
#include <fstream>
#include "SVMopt.h"
#include <string>
#include <fstream>


typedef std::vector<float> sample_x;
typedef struct SAMPLE
{
    sample_x x;
    float y;
    int input_dim;
}SAMPLE;

typedef std::vector<SAMPLE> Data; 

class SVM{
    public:
        SVM();
        SVM(SVMopt* _svmopt);
        ~SVM();
        int train();                            //train model
        int predict();                          //using model to predict

        int load();                             //load existing model
        int dump();                             //dump trained model to file

        // mode flag
        bool istrainmodel = true;
        bool isloadmodel = false;

        //SVM options, which needed to be set when initializing SVM 
        float _C;                               // cost in SMO
        float _epsilon;                         // precision in SMO
        int _KTYPE;                           // kernel type     
        std::string _ftrain = "";                   // training data file
        std::string _ftest = "";                    // testing/predicting data file
        std::string _fmodel = "";                   // file to dump or load model

        // field to describe model
        std::vector<float> _w;
        std::vector<float> _alpha;
        float _b;
        int _n;

    
    private:
        float kernel(sample_x x_i, sample_x x_j, int type);     //kernel function
        int examineExample(int i1);
        int takeStep(int i1, int i2);
        float g(SAMPLE sample);

        int read_data(std::string _f, Data &data);
        int write_data(std::string _f, Data &data);
        std::vector<std::string> split(const std::string& str, const std::string& delim);

        //field used in SMO
        Data _train_data;                        //Data for training, also need when predict with kernel function
        Data _test_data;                         //Data for predicting         
        float TOLERANCE = 0.001;
        std::vector<float> _error_cache;         // store Ei                  

};
#endif
