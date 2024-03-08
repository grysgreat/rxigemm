#include "math.h"
#include <assert.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <random>
//nvcc -arch=sm_75 -std=c++17 -Xcompiler -fopenmp rd_matrix.cu -o test -lcublas 


// 打印矩阵
void printMatrix(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 计算数组中元素的索引
            int index = i * cols + j;
            printf("%.4f   \t", matrix[index]);
        }
        std::cout << std::endl;
    }
}

// 打印矩阵
void printMatrix_h(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 计算数组中元素的索引
            int index = i * cols + j;
            printf("%.8f   \t", matrix[index]);
        }
        std::cout << std::endl;
    }
}

// 打印矩阵
void printMatrix_int(int matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 计算数组中元素的索引
            int index = i * cols + j;
            printf("%d\t\t", matrix[index]);
        }
        std::cout << std::endl;
    }
}


//矩阵拷贝函数
template <typename T>
void xcopy(T matrix1[],T matrix2[], int rows, int cols ) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
			matrix2[i*cols+j] = matrix1[i*cols+j];
        }
    }
}

template <typename Ti,typename To>
void quantitize(Ti* matrix_in,To* matrix_out,int rows,int cols,float lambda,char type){

    if(type == 'q'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = ceil(matrix_in[i*cols+j]*lambda);
            }
        }
    }
    else if(type == 'd'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda;
            }
        }        
    }

    // int max_exp;
    // std::frexp(max, &max_exp);
}




template <typename T>
T get_max(T* matrix,int rows,int cols){
    T maxM=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            maxM = max(std::abs(maxM),std::abs(matrix[i*cols+j]));
        }
    }
}

template <typename T>
T get_min_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T minM=2553555;
            for(int j=0;j<cols;j++){
                minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
            }
            vec[i] = minM;
        }
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            T minM=2553555;
            for(int i=0;i<rows;i++){
                minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
            }
            vec[j] = minM;
        }
    }    


}

template <typename T>
T get_max_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T minM=0;
            for(int j=0;j<cols;j++){
                minM = max(std::abs(minM),std::abs(matrix[i*cols+j]));
            }
            vec[i] = minM;
        }
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            T minM=0;
            for(int i=0;i<rows;i++){
                minM = max(std::abs(minM),std::abs(matrix[i*cols+j]));
            }
            vec[j] = minM;
        }
    }    


}


template <typename T>
T get_avg_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T sum=0;
            for(int j=0;j<cols;j++){
                sum += std::abs(matrix[i*cols+j]);
            }
            vec[i] = sum/cols;
        }
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            T sum=0;
            for(int i=0;i<rows;i++){
                sum += std::abs(matrix[i*cols+j]);
            }
            vec[j] = sum/rows;
        }
    }    


}

template <typename T>
T xmadd(T* matrixA,T* matrixB,T* matrixC,int rows,int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixC[i*cols+j] = matrixA[i*cols+j] + matrixB[i*cols+j];
        }
    }
}

template <typename T>
// 定义矩阵乘矩阵函数
void xgemm(const T A[], const T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    T tmp[rowsA*colsB];
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            // 初始化结果矩阵中的元素为0
            tmp[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
                // 矩阵乘法的累加步骤
                tmp[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
            }
        }
    }
    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j]  = tmp[i * colsB + j];
		}
	}
    
}

template <typename T>
void get_R(T matrix_in[],T matrix_cmp[],T matrixR[],int rows,int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixR[i*cols+j] = (matrix_in[i*cols+j] - matrix_cmp[i*cols+j]);
        }
    }
}

template <typename T>
void get_error(T matrix_ref[],T matrix_cmp[],T matrixR[],int rows,int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixR[i*cols+j] = std::abs(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])/std::abs(matrix_ref[i*cols+j]);
        }
    }
}

template <typename T>
void reduce_Residual(T* matrix,T* matrixR,int rows,int cols,int N,T max,int threshold){
    int max_exp;
    std::frexp(max, &max_exp);    
    int judge = max_exp - N + threshold;

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            int aij_exp;
            std::frexp(matrix[i*cols+j], &aij_exp); 
            if(aij_exp>judge){
                matrixR[i*cols+j] = 0;
            }
        }
    }        
}

template <typename T>
void reduce_Matrix(T* matrix,T* vec,int rows,int cols,T Mlam_k,char type){

    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T judge = Mlam_k*vec[i];
            //printf("judge = %f\n",judge);
            for(int j=0;j<cols;j++){
                if(matrix[i*cols+j]<judge) matrix[i*cols+j]=0;
            }
        }        
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k*vec[j];
                if(matrix[i*cols+j]<judge) matrix[i*cols+j]=0;
            }
        }        
    }
}


template <typename T>
void generate_matrix(T* matrix,int rows,int cols){
    // 创建一个随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    // 创建一个均匀分布，范围是[0, 1)
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrix[i*cols+j] = dis(gen);
            if(i==j) matrix[i*cols+j]+=100;
            else  matrix[i*cols+j]/=1;
        }
    }
}


// 定义矩阵乘矩阵函数
template <typename T,int digit>
void xigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {



    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int A_int[rowsA*colsA],B_int[rowsB*colsB],C_int[rowsB*colsB],AR_int[rowsA*colsA],BR_int[rowsB*colsB];
    T A_p[rowsA*colsA],B_p[rowsB*colsB],C_rows[rowsA],C_cols[colsB],A_copy[rowsA*colsA],B_copy[rowsB*colsB],C_copy[rowsA*colsB];

    xcopy<T>(A,A_copy, rowsA, colsA);
    xcopy<T>(B,B_copy, rowsB, colsB);

    const int max_int = 1<<(digit-1) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C,rowsA,colsB,lambdaC,'d');


    if(type == 'c' || type == 'x'){
        //对量化后的int矩阵反量化得到A',B'
        quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
        quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

        //计算full size的残差矩阵
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);
        //对残差矩阵进行量化
        T max_mAR =get_max<T>(A_p,rowsA,colsA);
        T max_mBR =get_max<T>(B_p,rowsB,colsB);
        T lambdaAR = (T)max_int/max_mAR;
        T lambdaBR = (T)max_int/max_mBR;
        //对A,B残差矩阵进行直接量化
        quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
        quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');


        T lambdaAnew,lambdaBnew;
        if(type == 'x'){
            //稀疏化误差修补
            T ml_kA = threadhoud*lambdaB/colsA;
            T ml_kB = threadhoud*lambdaA/colsA;    


            get_avg_vec<T>(C,C_rows,rowsA,colsB,'r');
            get_avg_vec<T>(C,C_cols,rowsA,colsB,'c');
            
            reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,C_cols,rowsB,colsB, ml_kB ,'c');
                
            //对新AB进行量化
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = (T)max_int/max_mAnew;
            lambdaBnew = (T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');
        }




        //对int误差修复矩阵反量化得到误差修复矩阵float
        T lambdaCR1 = lambdaAnew*lambdaBR;
        T lambdaCR2 = lambdaAR*lambdaBnew;

        xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
        quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
        //使用修复矩阵补充误差
        xmadd<float>(C_copy,C,C,rowsA,colsB);

        
        xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
        quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
        //使用修复矩阵补充误差
        xmadd<float>(C_copy,C,C,rowsA,colsB);
    }


}



int main(){
    int N=10,M=10,K=10;

    float matrixA[10000],matrixB[10000],matrixC[10000],matrixCQ[10000],matrixR[10000];
    float matrixAp[10000],matrixBp[10000],matrixAR[10000],matrixBR[10000],matrixCR[10000],matrixCR1[10000],matrixCR2[10000],matrixCR12[10000];

    int matrix_intA[10000],matrix_intB[10000],matrix_intC[10000];

    int matrix_intAR[10000],matrix_intBR[10000],matrix_intCR1[10000],matrix_intCR2[10000];

    generate_matrix<float>(matrixA,M,K);
    generate_matrix<float>(matrixB,K,N);

    float max_mA =get_max<float>(matrixA,M,K);
    float max_mB =get_max<float>(matrixB,K,N);

    const int digit=10;

    const int max_int = 1<<(digit-1) - 1;
    float thread_f = 1.0/(float)(1<<6);

    float lambdaA = (float)max_int/max_mA;
    float lambdaB = (float)max_int/max_mB;


    //对A,B矩阵进行直接量化
    quantitize<float,int>(matrixA,matrix_intA,M,K,lambdaA,'q');
    quantitize<float,int>(matrixB,matrix_intB,K,N,lambdaB,'q');

    // printMatrix_int(matrix_intA,M,K);
    // std::cout<<"\n\n";

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);
    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');

    //计算量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);
    
    //相对误差结果矩阵
    std::cout<<"函数相对误差"<<"\n";
    printMatrix(matrixR,M,N);
    std::cout<<"\n\n";




    xgemm<int>(matrix_intA,matrix_intB,matrix_intC,M,K,K,N);


    //对结果矩阵C_INT进行反量化得到C'
    float lambdaC = lambdaA*lambdaB;
    quantitize<int,float>(matrix_intC,matrixCQ,M,N,lambdaC,'d');

    //拷贝一个C'标准量化结果矩阵供后面比较用
    float matrixCQ2[10000];
    xcopy<float>(matrixCQ,matrixCQ2,M,N);


    //计算量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);
    
    //相对误差结果矩阵
    std::cout<<"原始相对误差"<<"\n";
    printMatrix(matrixR,M,N);
    std::cout<<"\n\n";




    //对量化后的int矩阵反量化得到A',B'
    quantitize<int,float>(matrix_intA,matrixAp,M,K,lambdaA,'d');
    quantitize<int,float>(matrix_intB,matrixBp,K,N,lambdaB,'d');
    //计算full size的残差矩阵
    get_R<float>(matrixA,matrixAp,matrixAR,M,K);
    get_R<float>(matrixB,matrixBp,matrixBR,K,N);

    //对残差矩阵进行量化
    float max_mAR =get_max<float>(matrixAR,M,K);
    float max_mBR =get_max<float>(matrixBR,K,N);
    float lambdaAR = (float)max_int/max_mAR;
    float lambdaBR = (float)max_int/max_mBR;
    //对A,B残差矩阵进行直接量化
    quantitize<float,int>(matrixAR,matrix_intAR,M,K,lambdaAR,'q');
    quantitize<float,int>(matrixBR,matrix_intBR,K,N,lambdaBR,'q');

    //计算量化残差矩阵乘法的到两个误差修复矩阵-int
    xgemm<int>(matrix_intA,matrix_intBR,matrix_intCR1,M,K,K,N);
    xgemm<int>(matrix_intAR,matrix_intB,matrix_intCR2,M,K,K,N);

    //对int误差修复矩阵反量化得到误差修复矩阵float
    float lambdaCR1 = lambdaA*lambdaBR;
    float lambdaCR2 = lambdaAR*lambdaB;

    // std::cout<<lambdaCR1<<":"<<lambdaCR2<<"\n\n";
    quantitize<int,float>(matrix_intCR1,matrixCR1,M,N,lambdaCR1,'d');
    quantitize<int,float>(matrix_intCR2,matrixCR2,M,N,lambdaCR2,'d');

    //使用修复矩阵补充误差
    xmadd<float>(matrixCR1,matrixCR2,matrixCR12,M,N);
    xmadd<float>(matrixCR12,matrixCQ,matrixCQ,M,N);

    // printMatrix(matrixA,M,K);
    // std::cout<<"\n\n";

    // printMatrix(matrixAR,M,K);
    // std::cout<<"\n\n";



    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   

    std::cout<<"稠密化修补"<<"\n";
    printMatrix(matrixR,M,K);
    std::cout<<"\n\n";

    



    float maxC_rows[10000],maxC_cols[10000],maxRC_rows[10000],maxRC_cols[10000];


    //获取直接量化残差矩阵
    // get_R<float>(matrixC,matrixCQ2,matrixCR,M,K);
    // get_max_vec<float>(matrixCR,maxRC_rows,M,N,'r');
    // get_max_vec<float>(matrixCR,maxRC_cols,M,N,'c');

    // float l_2K = lambdaA*maxRC_cols[0]/(2*K);




    //稀疏化误差修补
    float ml_kA = thread_f*lambdaB/K;
    float ml_kB = thread_f*lambdaA/K;



    std::cout<<"thread_f="<<thread_f<<"\n\n";


    get_avg_vec<float>(matrixCQ2,maxC_rows,M,N,'r');
    get_avg_vec<float>(matrixCQ2,maxC_cols,M,N,'c');



    reduce_Matrix(matrixA,maxC_rows,M,N, ml_kA ,'r');
    reduce_Matrix(matrixB,maxC_cols,M,N, ml_kB ,'c');
    // printMatrix(matrixA,M,K);
    // printMatrix(matrixC,M,K);
    // std::cout<<"\n\n";
    // printMatrix(matrixCQ2,M,K);
    // std::cout<<"\n\n";
    // printMatrix_h(maxC_rows,1,N);
    // std::cout<<"\n\n";        


    // reduce_Residual<float>(matrixA,matrixAR,M,K,digit-1,max_mA,threadhoud_i);
    // reduce_Residual<float>(matrixB,matrixBR,M,K,digit-1,max_mA,threadhoud_i);


    //对新AB进行量化
    float max_mAnew =get_max<float>(matrixA,M,K);
    float max_mBnew =get_max<float>(matrixB,K,N);
    float lambdaAnew = (float)max_int/max_mAnew;
    float lambdaBnew = (float)max_int/max_mBnew;
    //对A,B残差矩阵进行直接量化
    quantitize<float,int>(matrixA,matrix_intA,M,K,lambdaAnew,'q');
    quantitize<float,int>(matrixB,matrix_intB,K,N,lambdaBnew,'q');


    // printMatrix_int(matrix_intAR,M,K);
    // std::cout<<"\n\n";


    //计算量化残差矩阵乘法的到两个误差修复矩阵-int
    xgemm<int>(matrix_intA,matrix_intBR,matrix_intCR1,M,K,K,N);
    xgemm<int>(matrix_intAR,matrix_intB,matrix_intCR2,M,K,K,N);

    //对int误差修复矩阵反量化得到误差修复矩阵float
    lambdaCR1 = lambdaAnew*lambdaBR;
    lambdaCR2 = lambdaAR*lambdaBnew;

    // std::cout<<lambdaCR1<<":"<<lambdaCR2<<"\n\n";
    quantitize<int,float>(matrix_intCR1,matrixCR1,M,N,lambdaCR1,'d');
    quantitize<int,float>(matrix_intCR2,matrixCR2,M,N,lambdaCR2,'d');

    //使用修复矩阵补充误差
    xmadd<float>(matrixCR1,matrixCR2,matrixCR12,M,N);
    xmadd<float>(matrixCR12,matrixCQ2,matrixCQ2,M,N);

    // printMatrix(matrixAR,M,K);
    // std::cout<<"\n\n";


    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ2,matrixR,M,N);   


    std::cout<<"稀疏化修补"<<"\n";
    printMatrix(matrixR,M,K);
    std::cout<<"\n\n";

    // printMatrix(matrixAp,M,K);
    // std::cout<<"\n\n";
    std::cout<<"稀疏A矩阵"<<"\n";
    printMatrix(matrixA,M,K);
    std::cout<<"\n\n";
}