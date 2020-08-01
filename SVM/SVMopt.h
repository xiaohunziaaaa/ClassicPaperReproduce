#ifndef __SVMOPT_H__
#define __SVMOPT_H__
#include <string>

//kernel type, only No_kernel, Gaussian Kernel, Polynomial Kernel supported
class SVMopt{
    public:
        float _C = 1.0;                         // cost in SMO
        float _epsilon = 0.001;                 // precision in SMO, used to decide whether descent of alpha2 is enough
        int _KTYPE = 0;                       // kernel type     
        std::string _ftrain="";                   // training data file
        std::string _ftest="";                    // testing/predicting data file
        std::string _fmodel="";                   // file to dump or load model
        SVMopt();
        ~SVMopt();
};

#endif

