#include <iostream>
#include <getopt.h>
#include <string>
#include "SVM.h"
#include "SVMopt.h"
#include <stdio.h>

int parse_command_line(int argc, char *argv[], SVMopt* svmopt);
void help();



//main -ftrain ./train.txt -ftest ./test.txt -C 1 -epsilon 0.001 
int main(int argc, char *argv[]){
    SVMopt* svmopt = new SVMopt();
    if(parse_command_line(argc, argv, svmopt)==-1){
        return -1;
    };
    SVM svm = SVM(svmopt);
    if(svm.istrainmodel){
        svm.train();
        if(svm._ftest!=""){
            svm.predict();
        }
    }
    else{
        svm.load();
        if(svm._ftest!=""){
            svm.predict();
        }
    }
}


int parse_command_line(int argc, char *argv[], SVMopt* svmopt){
    int option;
    const char *opt_string = "";
    struct option long_opts[] = {
        {"ftrain",          2, NULL, 0},
        {"ftest",           2, NULL, 1},
        {"fmodel",          2, NULL, 2},
        {"KTYPE",           2, NULL, 3},
        {"C",               1, NULL, 4},
        {"epsilon",         1, NULL, 5},
        {"help",            2, NULL, 6},
        {0, 0, 0, 0}
    };
    
    bool flag_train = false;
    bool flag_test = false;
    bool flag_model = false;
    while ((option = getopt_long(argc, argv, opt_string, long_opts, NULL)) != -1){
        switch (option) {
            case 0:
                svmopt->_ftrain = optarg;
                flag_train = true;
                break;
            case 1:
                svmopt->_ftest = optarg;
                flag_test = true;
                break;
            case 2:
                svmopt->_fmodel = optarg;
                flag_model = true;
                break;
            case 3:
                svmopt->_KTYPE = atoi(optarg);
                break;
            case 4:
                svmopt->_C = atof(optarg);
                break;
            case 5:
                svmopt->_epsilon = atof(optarg);
                break;
            case 6:
                help();
                return -1;
            default:
                printf("Unknown option: %c\n",(char)optopt);
                help();
                return -1;
        }
    }

    if (!(flag_test || flag_train || flag_model)){
        std::cout<<"You should indicate train or model to be loaded!";
        return -1;
    }

    if(!flag_train && !flag_model && flag_test){
        std::cout<<"You should indicate train or model to be loaded!";
        return -1;
    }

    if(flag_train && flag_model){
        std::cout<<"Either train a model or load a model, can not do both!";
        return -1;
    }
    return 0;
}

void help(){
    std::cout << "HOW-TO:" <<"--ftrain=${ftrain}"
                           <<"--fmodel=${fmodel} " 
                           <<"--ftest=${ftest} "
                           <<"[--KTYPE=${KYPE}]" 
                           <<"[--C=${cost}] " 
                           <<"[--epsilon=${eps}]"
                           <<"[--help]" 
                           << std::endl;
    std::cout << "-ftrain: training examples file"<<std::endl
              << "-fmodel: model path, if training, it means model path to dump model, otherwise, it means model path where model to be loaded"<<std::endl
              << "-fvalid: testing examples file"<<std::endl
              << "-KTYPE: kernel type, only No kernel(0), Gaussian kernel(1), polynomial kernel supported(2), default=No kernel"<<std::endl
              << "-epsilon: precision in SMO algorithm, default=0.001"<<std::endl
              << "-C: cost in SMO algorithm, default=1.0"<<std::endl
              << "-help: show this help";
}