#include "SVMopt.h"

SVMopt::SVMopt(){
     _C = 1.0;                         // cost in SMO
    _epsilon = 0.001;                 // precision in SMO, used to decide whether descent of alpha2 is enough
    _KTYPE = 0;                       // kernel type     
    _ftrain="";                   // training data file
     _ftest="";                    // testing/predicting data file
    _fmodel="";  
}

SVMopt::~SVMopt(){
    
}