


#include "matrix/util.h"
#include "matrix/rcxigemm.hpp"

using Eigen::ComputeThinU;
using Eigen::ComputeThinV;
using Eigen::Index;
using Eigen::JacobiSVD;
using Eigen::MatrixXf;

#define Iter_Num 1
template <int digit>
void precision_series_wapper(int num,float thread_f,char type){
    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,type);
    generate_matrix<float>(matrixB,K,N,type);




    //计算float和int矩阵乘法得到结果矩阵
    eigen_SGEMM(matrixA,matrixB,matrixC,M,K,K,N);

    float R0=0,R1=0,R5=0;
    for(int i=0;i<Iter_Num;i++){
        xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
        R0 += get_Ferror<float>(matrixC,matrixCQ,M,N);     



        xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
        R1 += get_Ferror<float>(matrixC,matrixCQ,M,N); 




        rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
        //Calculate the quantization matrix multiplication error after the repair error
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);   

        R5 += get_Ferror<float>(matrixC,matrixCQ,M,N); 

    }
    R0/=(float)Iter_Num;
    R1/=(float)Iter_Num;
    R5/=(float)Iter_Num;
    //std::cout<<";type = "<<type<<"; th = "<<thread_f<<"; precision="<<R5<<"; full="<<R1<<"; origin="<<R0<<"\n";
    printf("Int%d\t\t%d\t\t%c\t\t\t%f\t\t%f\t\t%f\t\t%f\n",digit,num,type,thread_f,R5,R1,R0);

    return ;
}

void precision_series_test(){

    printf("Digit\t\tSize\t\trandom type\t\tThreshold\t\tprecision(ours)\t\tprecision(full)\t\tprecision(origin)\n");
    std::vector<std::tuple<int, float, char>> inference_server_set = {
        std::make_tuple(1024, 1, 'k'),
        std::make_tuple(1024, 0.8, 'k'),
        std::make_tuple(1024, 0.6, 'k'),
        std::make_tuple(1024, 0.4, 'k'),
        std::make_tuple(1024, 0.2, 'k'),
        std::make_tuple(1024, 0.05, 'k'),

        std::make_tuple(1024, 1, 'e'),
        std::make_tuple(1024, 0.8, 'e'),
        std::make_tuple(1024, 0.5, 'e'),
        std::make_tuple(1024, 0.2, 'e'),
        std::make_tuple(1024, 0.1, 'e'),
        std::make_tuple(1024, 0.05, 'e'),


        std::make_tuple(1024, 1, 'u'),
        std::make_tuple(1024, 0.8, 'u'),
        std::make_tuple(1024, 0.5, 'u'),
        std::make_tuple(1024, 0.2, 'u'),
        std::make_tuple(1024, 0.1, 'u'),
        std::make_tuple(1024, 0.05, 'u'),

        std::make_tuple(1024, 1, 'n'),
        std::make_tuple(1024, 0.8, 'n'),
        std::make_tuple(1024, 0.5, 'n'),
        std::make_tuple(1024, 0.2, 'n'),
        std::make_tuple(1024, 0.1, 'n'),
        std::make_tuple(1024, 0.05, 'n'),

        std::make_tuple(1024, 1, 'p'),
        std::make_tuple(1024, 0.5, 'p'),
        std::make_tuple(1024, 0.5, 'p'),
        std::make_tuple(1024, 0.2, 'p'),
        std::make_tuple(1024, 0.1, 'p'),
        std::make_tuple(1024, 0.05, 'p'),

        std::make_tuple(4096, 1, 'k'),
        std::make_tuple(4096, 0.8, 'k'),
        std::make_tuple(4096, 0.6, 'k'),
        std::make_tuple(4096, 0.4, 'k'),
        std::make_tuple(4096, 0.2, 'k'),
        std::make_tuple(4096, 0.05, 'k'),

        std::make_tuple(4096, 1, 'e'),
        std::make_tuple(4096, 0.8, 'e'),
        std::make_tuple(4096, 0.5, 'e'),
        std::make_tuple(4096, 0.2, 'e'),
        std::make_tuple(4096, 0.1, 'e'),
        std::make_tuple(4096, 0.05, 'e'),


        std::make_tuple(4096, 1, 'u'),
        std::make_tuple(4096, 0.8, 'u'),
        std::make_tuple(4096, 0.5, 'u'),
        std::make_tuple(4096, 0.2, 'u'),
        std::make_tuple(4096, 0.1, 'u'),
        std::make_tuple(4096, 0.05, 'u'),

        std::make_tuple(4096, 1, 'n'),
        std::make_tuple(4096, 0.8, 'n'),
        std::make_tuple(4096, 0.5, 'n'),
        std::make_tuple(4096, 0.2, 'n'),
        std::make_tuple(4096, 0.1, 'n'),
        std::make_tuple(4096, 0.05, 'n'),

        std::make_tuple(4096, 1, 'p'),
        std::make_tuple(4096, 0.5, 'p'),
        std::make_tuple(4096, 0.5, 'p'),
        std::make_tuple(4096, 0.2, 'p'),
        std::make_tuple(4096, 0.1, 'p'),
        std::make_tuple(4096, 0.05, 'p'),
    };
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, th, type) = problem;
        precision_series_wapper<8>(num,th,type);
    }
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, th, type) = problem;
        precision_series_wapper<4>(num,th,type);
    }
}



template <int digit>
void speed_series_wapper(int num,float thread_f,char type){

    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,type);
    generate_matrix<float>(matrixB,K,N,type);


    auto start = std::chrono::high_resolution_clock::now();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff;
    double time;

    //Compute the multiplication of float and int matrices to get the resulting matrix
    eigen_SGEMM(matrixA,matrixB,matrixC,M,K,K,N);


    start = std::chrono::high_resolution_clock::now();
    
    for(int i=0;i<Iter_Num;i++) xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    double time0  = diff.count()/(double)Iter_Num;
  



    start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<Iter_Num;i++) xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    double time1  = diff.count()/(double)Iter_Num;
    




    start = std::chrono::high_resolution_clock::now();
    for(int i=0;i<Iter_Num;i++) rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    double time2  = diff.count()/(double)Iter_Num;    

    //std::cout<<";type = "<<type<<"; th = "<<thread_f<<"; precision="<<R5<<"; full="<<R1<<"; origin="<<R0<<"\n";
    printf("%f\t\t%d\t\t%lf\t\t%lf\t\t%lf\n",thread_f,num,time2,time1,time0);

    return ;
}

void speed_series_test(){

    printf("Threshoud\t\tSize\t\ttime(ours)\t\ttime(full)\t\ttime(origin)\n");
    std::vector<std::tuple<int, float, char>> inference_server_set = {
            std::make_tuple(2048, 0.0, 'u'),
            std::make_tuple(2048, 0.1, 'u'),
            std::make_tuple(2048, 0.2, 'u'),
            std::make_tuple(2048, 0.3, 'u'),
            std::make_tuple(2048, 0.4, 'u'),
            std::make_tuple(2048, 0.5, 'u'),
            std::make_tuple(2048, 0.6, 'u'),
            std::make_tuple(2048, 0.7, 'u'),
            std::make_tuple(2048, 0.8, 'u'),
            std::make_tuple(2048, 0.9, 'u'),
            std::make_tuple(2048, 1.0, 'u'),

            std::make_tuple(4096, 0.0, 'u'),
            std::make_tuple(4096, 0.1, 'u'),
            std::make_tuple(4096, 0.2, 'u'),
            std::make_tuple(4096, 0.3, 'u'),
            std::make_tuple(4096, 0.4, 'u'),
            std::make_tuple(4096, 0.5, 'u'),
            std::make_tuple(4096, 0.6, 'u'),
            std::make_tuple(4096, 0.7, 'u'),
            std::make_tuple(4096, 0.8, 'u'),
            std::make_tuple(4096, 0.9, 'u'),
            std::make_tuple(4096, 1.0, 'u'),

            std::make_tuple(8192, 0.0, 'u'),
            std::make_tuple(8192, 0.1, 'u'),
            std::make_tuple(8192, 0.2, 'u'),
            std::make_tuple(8192, 0.3, 'u'),
            std::make_tuple(8192, 0.4, 'u'),
            std::make_tuple(8192, 0.5, 'u'),
            std::make_tuple(8192, 0.6, 'u'),
            std::make_tuple(8192, 0.7, 'u'),
            std::make_tuple(8192, 0.8, 'u'),
            std::make_tuple(8192, 0.9, 'u'),
            std::make_tuple(8192, 1.0, 'u'),
    };
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, th, type) = problem;
        speed_series_wapper<8>(num,th,type);
    }

}

void density_series_wapper(int num,char type){
    int N=num,M=num,K=num;
    float thread_f=0;
    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,type);
    generate_matrix<float>(matrixB,K,N,type);


    for(int i=0;i<50;i++){
        thread_f= 0.02*i;

        printf("%d\t\t%c\t\t\t%lf\t\t",num,type,thread_f);
        const int digit=8;
        rcxigemm_print_sp<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
        //Calculate the quantization matrix multiplication error after the repair error
        float R5 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
       
         printf("\n");

    }

    return ;
}

void density_series_test(){
    printf("size\t\trandom type\t\tthreshoud\t\tdensity\n");
    std::vector<std::tuple<int, char>> inference_server_set = {


        std::make_tuple(1024,  'k'),

        std::make_tuple(1024,'e'),

        std::make_tuple(1024,  'u'),

        std::make_tuple(1024, 'n'),

        std::make_tuple(1024, 'p'),

        std::make_tuple(4096,  'k'),

        std::make_tuple(4096,'e'),

        std::make_tuple(4096,  'u'),

        std::make_tuple(4096, 'n'),

        std::make_tuple(4096, 'p'),

    };
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, type) = problem;
        density_series_wapper(num,type);
    }
}


int main(int argc, char* argv[]){
    std::string argument;
    if (argc != 1) {
        argument = argv[1];
    } else {
        printf("please type  correct parameter for test\n");
        return 0;
    }
    if(argument=="precision"){
        precision_series_test();
    }else if(argument=="speed"){
        speed_series_test();
    }else if(argument=="density"){
        density_series_test();
    } else {
        printf("unknown test parameter!\n");
    }
    return 0;

}