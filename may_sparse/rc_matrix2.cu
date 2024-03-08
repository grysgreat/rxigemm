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
//nvcc -arch=sm_75 -std=c++17 -Xcompiler -fopenmp rc_matrix2.cu -o test -lcublas 
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>

// //秒级
// double duration_second = std::chrono::duration<double>(afterTime - beforeTime).count();
// //毫秒级
// double duration_millsecond = std::chrono::duration<double, std::milli>(afterTime - beforeTime).count();
// //微妙级
// double duration_microsecond = std::chrono::duration<double, std::micro>(afterTime - beforeTime).count();
// //纳秒级
// double duration_nanosecond = std::chrono::duration<double, std::nano>(afterTime - beforeTime).count();


template <typename T>
void generate_matrix(T* matrix,int rows,int cols,char type ){
    // 创建一个随机数引擎
    std::random_device rd;
    std::mt19937 gen(rd());
    // 创建一个均匀分布，范围是[0, 1)
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    //std::normal_real_distribution<float> dis(0.0, 1.0);

    if(type == 'u'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix[i*cols+j] = dis(gen);
                if(i==j&&i!=rows-1) matrix[i*cols+j] = (matrix[i*cols+j]);
                else  matrix[i*cols+j]=(matrix[i*cols+j]);
            }
        }        
    }
    else if(type == 'n'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::normal_distribution<double> dis(10.0, 3.0); // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'e'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::exponential_distribution<double> dis(1.0/(1.0/4.0)); // 定义随机数分布器对象dis，期望为0.0，标准差为1.0的正态分布
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'k'){
        std::chi_squared_distribution<> dis(0.5);
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'p'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::poisson_distribution<> dis(10);
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }
}


void generate_ZDmatrix(float* matrix,int rows,int cols ){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            if(i==j) matrix[i*cols+j] = i+j+3;
            else  matrix[i*cols+j] = 0.5;
        }
    }     
}


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

// 高精度打印矩阵
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

// 打印int矩阵
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

// 定义矩阵转置函数
template <typename T>
void xtrans(T matrix[],T result[] , int rows, int cols) {

	T tmp[rows*cols];
    // 执行矩阵转置   
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // 计算原矩阵中元素的索引
            int originalIndex = i * cols + j;
            // 计算转置后矩阵中元素的索引
            int transposedIndex = j * rows + i;
            // 进行转置
            tmp[transposedIndex] = matrix[originalIndex];
        }
    }
    xcopy<T>(tmp,result,cols,rows);
}




template <typename Ti,typename To>
void quantitize(Ti* matrix_in,To* matrix_out,int rows,int cols,float lambda,char type){

    if(type == 'q'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda);
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



template <typename Ti,typename To>
void quantitize_vec(Ti* matrix_in,To* matrix_out, float* lambda_vec,int rows,int cols,char type,char rc){

    if(type == 'q'){
        if(rc == 'r'){
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda_vec[i]);
                }
            }
        }
        else if(rc == 'c'){
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (matrix_in[i+j*cols]*lambda_vec[i]);
                }
            }        
        }
    }
    else if(type == 'd'){
        if(rc == 'r'){
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda_vec[i];
                }
            }
        }
        else if(rc == 'c'){
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (To)(matrix_in[i+j*cols]/lambda_vec[i]);
                }
            }        
        }    
    }
    return;
}

template <typename Ti,typename To>
void dequantitize_matrix(Ti* matrix_in,To* matrix_out, float* lambda_r,  float* lambda_c,int rows,int cols){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrix_out[i*cols+j] = (To)(matrix_in[i*cols+j]/(lambda_r[i]*lambda_c[j]));
        }
    }
}

template <typename T>
T get_lambda_vec(T* lambda_vec,T* max_vec,int digit, int len){
    int max_int = 1<<(digit-1) - 1;
    for(int i=0;i<len;i++){
        lambda_vec[i]=(T)max_int/max_vec[i];
    }
}

template <typename T>
T get_max(T* matrix,int rows,int cols){
    T maxM=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            maxM = max(std::abs(maxM),std::abs(matrix[i*cols+j]));
        }
    }
    return maxM;
}

template <typename T>
T get_avg(T* matrix,int rows,int cols){
    T sum=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sum = ((sum)+std::abs(matrix[i*cols+j]));
        }
    }
    T avg = sum/(T)(rows*cols);
    return avg;
}


template <typename T>
T get_min(T* matrix,int rows,int cols){
    T minM=1000000;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(std::abs(matrix[i*cols+j])>1e-20)
                minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
        }
    }
    return minM;
}


template <typename T>
void get_min_vec(T* matrix,T* vec, int rows,int cols,char type){
    
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
void get_max_vec(T* matrix,T* vec, int rows,int cols,char type){
    
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
void get_avg_vec(T* matrix,T* vec, int rows,int cols,char type){
    
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
void xmadd(T* matrixA,T* matrixB,T* matrixC,int rows,int cols){
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
    T* tmp =(T *)malloc(sizeof(T) * rowsA*colsB);

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
int get_nnz(T *denseMatrix, int numRows,int numCols){
    int nnz = 0; // Number of non-zero elements
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            if (denseMatrix[i*numCols+j] != 0) {
                ++nnz;
            }
        }
    }
    return nnz;
}


// Function to convert a dense matrix to CSR format
template <typename T>
void denseToCSR(T* denseMatrix,T *values,int *colIndex,int *rowPtr, int numRows, int numCols) {
    int nnz = 0;
    for (int i = 0; i < numRows; ++i) {
        rowPtr[i] = nnz;
        for (int j = 0; j < numCols; ++j) {
            if (denseMatrix[i*numCols+j] != 0) {
                values[nnz] = denseMatrix[i*numCols+j];
                colIndex[nnz] = j;
                ++nnz;
            }
        }
    }
    rowPtr[numRows] = nnz;

}


// Function to perform matrix-vector multiplication (CSR format) spmm 
template <typename T>
void sspmm(T *values,int *colIndex,int *rowPtr, T* B, T* C,int rowsA,int colsA,int rowsB, int colsB) {

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i*colsB+j] = 0.0;
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
                C[i*colsB+j] += values[k] * B[colIndex[k]*colsB+j];
            }
        }
    }
}

template <typename T>
void sspmm_nt(T *values,int *colIndex,int *rowPtr, T* B, T* C,int rowsA,int colsA,int rowsB, int colsB) {

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i+j*rowsA] = 0.0;
            for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
                
                C[i+j*rowsA] += values[k] * B[colIndex[k]+j*colsA];
            }
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
T get_Ferror(T matrix_ref[],T matrix_cmp[],int rows,int cols){

    T sumR=0,sum=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sumR+=(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])*(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j]);
            sum+=(matrix_ref[i*cols+j])*(matrix_ref[i*cols+j]);
        }
    }

    T ans = sqrt(sumR)/sqrt(sum);
    return ans;

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
    float cnt=0;
    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T judge = Mlam_k;
            //printf("judge = %f\n",judge);
            for(int j=0;j<cols;j++){
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix[i*cols+j]=0;
                    cnt++;
                }
            }
        }        
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k;
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix[i*cols+j]=0;
                    cnt++;
                }
            }
        }        
    }
    float spasity = 1-(cnt/(float)(rows*cols));
    //printf("spasity = %f\n",spasity);
}

template <typename T>
void reduce_Matrix_out(T* matrix,T* matrix_out,int rows,int cols,T Mlam_k,char type){
    float cnt=0;
    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T judge = Mlam_k;
            //printf("judge = %f\n",judge);
            for(int j=0;j<cols;j++){
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix_out[i*cols+j]=0;
                    cnt++;
                } else{
                    matrix_out[i*cols+j]=matrix[i*cols+j];
                }
            }
        }        
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k;
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix_out[i*cols+j]=0;
                    cnt++;
                }else{
                    matrix_out[i*cols+j]=matrix[i*cols+j];
                }
            }
        }        
    }
    float spasity = 1-(cnt/(float)(rows*cols));
    //printf("spasity = %f\n",spasity);
}


template <typename T>
void reduce_Matrix_origin(T* matrix,T* vec,T* lambda_vec,int rows,int cols,T Mlam_k,char type){

    if(type == 'r'){
        for(int i=0;i<rows;i++){
            T judge = Mlam_k*vec[i]*lambda_vec[i];
            //printf("judge = %f,Mlam_k=%f,vec=%f,lambda_vec=%f\n",judge,Mlam_k,vec[i],lambda_vec[i]);
            for(int j=0;j<cols;j++){
                if(matrix[i*cols+j]<judge) matrix[i*cols+j]=0;
            }
        }        
    }
    if(type == 'c'){
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k*vec[j]*lambda_vec[j];
                if(matrix[i*cols+j]<judge) matrix[i*cols+j]=0;
            }
        }        
    }
}

template <typename T>
void splitMatrix(T A[], T B[],  int rows, int cols, T judge){
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(abs(A[i*cols+j])<judge){
                B[i*cols+j] = A[i*cols+j];
                A[i*cols+j] = 0;
            } else {
                B[i*cols+j] = 0;
            }
        }
    }        
}

template <typename T>
T get_sparsity(T A[],int rows, int cols){
    int cnt=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(A[i*cols+j]!=0){
                cnt++;
            } 
        }
    }     
    T sp = (T)cnt / (T)(rows*cols);
    return sp;
}


// 定义矩阵乘矩阵函数---- 残差法
template <typename T,int digit>
void xigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
    int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *C_rows = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_cols = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);

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
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

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


        T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
        if(type == 'x'){
            //稀疏化误差修补
            T avg = get_avg<T>(B,rowsB,colsB);
            // std::cout<<"avg = "<<avg<<"\n";

            threadhoud/=avg;
            T ml_kA = threadhoud*lambdaB/colsA;
            T ml_kB = threadhoud*lambdaA/colsA;    

            get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
            get_avg_vec<T>(C_buffer,C_cols,colsA,colsB,'c');
            
            reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,C_cols,rowsB,rowsB, ml_kB ,'c');
            //对新AB进行量化
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');
            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            int nnzA = get_nnz<int>(A_int, rowsA, colsA);
            int nnzB = get_nnz<int>(B_int, rowsA, colsA);

            int* valuesA = new int[nnzA];
            int* colIndexA = new int[nnzA];
            int* rowPtrA = new int[rowsA + 1];
            denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


            sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);


            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


            xtrans<int>(B_int,B_int,rowsB,colsB);
            xtrans<int>(AR_int,AR_int,rowsA, colsA);
            int* valuesB = new int[nnzB];
            int* colIndexB = new int[nnzB];
            int* rowPtrB = new int[colsB + 1];
            denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
            sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
            xtrans<int>(C_int,C_int,colsB,rowsA);

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //使用修复矩阵补充误差

            if(lambdaCR2!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //使用修复矩阵补充误差
            if(lambdaCR2!=0&&max_mAR!=0) xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}



// 定义矩阵乘矩阵函数---- 残差法
template <typename T,int digit>
void rcxigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
    int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
    int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
    int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

    T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *C_rows = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *C_cols = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *A_rows = (T *)malloc(sizeof(T) * rowsA );
    T *B_cols = (T *)malloc(sizeof(T) * colsB);

    xcopy<T>(A,A_copy, rowsA, colsA);
    xcopy<T>(B,B_copy, rowsB, colsB);

    const int max_int = 1<<(digit-1) - 1;

    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, digit, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, digit, colsB);

    //对A,B矩阵进行直接量化
    quantitize_vec<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


    //对结果矩阵C_INT进行反量化得到C'

    dequantitize_matrix<int,T>(C_int,C_buffer,lambdaA_vec,lambdaB_vec,rowsA,colsB);

    if(type == 'c' || type == 'x'){
        
        //对量化后的int矩阵反量化得到A',B'
        quantitize_vec<int,T>(A_int,A_p,lambdaA_vec ,rowsA,colsA,'d','r');
        quantitize_vec<int,T>(B_int,B_p,lambdaB_vec ,rowsB,colsB,'d','c');

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




        T lambdaAnew ,lambdaBnew ;
        if(type == 'x'){
            //稀疏化误差修补
            T avgA = get_avg<T>(A,rowsA,colsA);
            T avgB = get_avg<T>(B,rowsB,colsB);
            // std::cout<<"avg = "<<avg<<"\n";

            //threadhoud/=avg;
            T ml_kA = 2*threadhoud*avgA;//*lambdaB/colsA;
            T ml_kB = 2*threadhoud*avgB;//*lambdaA/colsA;    
            // T ml_kA = threadhoud/colsA;
            // T ml_kB = threadhoud/colsA;   

            get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
            get_avg_vec<T>(C_buffer,C_cols,colsA,colsB,'c');


            get_avg_vec<T>(A_copy,A_rows,rowsA,colsB,'r');
            get_avg_vec<T>(B_copy,B_cols,colsA,colsB,'c');

            reduce_Matrix(A_copy,lambdaA_vec,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,lambdaB_vec,rowsB,rowsB, ml_kB ,'c');

            float sp = get_sparsity(B_copy,rowsA,colsA);
            float sp2 = get_sparsity(B,rowsA,colsA);
            std::cout<<sp;
            
            //对新AB进行量化
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');
            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            int nnzA = get_nnz<int>(A_int, rowsA, colsA);
            int nnzB = get_nnz<int>(B_int, rowsA, colsA);

            int* valuesA = new int[nnzA];
            int* colIndexA = new int[nnzA];
            int* rowPtrA = new int[rowsA + 1];
            denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


            sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);


            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR1!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


            xtrans<int>(B_int,B_int,rowsB,colsB);
            xtrans<int>(AR_int,AR_int,rowsA, colsA);
            int* valuesB = new int[nnzB];
            int* colIndexB = new int[nnzB];
            int* rowPtrB = new int[colsB + 1];
            denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
            sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
            xtrans<int>(C_int,C_int,colsB,rowsA);

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //使用修复矩阵补充误差

            if(lambdaCR2!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //对新AB进行量化
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');

            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //使用修复矩阵补充误差
            if(lambdaCR2!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}





// 计算向量的内积
template <typename T>
T dotProduct_v(T v1[], T v2[], int size) {
    T result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}





int test1(){
    int N=400,M=400,K=400;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');


    const int digit=8;
    float thread_f = 1;

    // printMatrix(matrixA,M,K);
    // std::cout<<"\n\n";
    // printMatrix(matrixB,M,K);
    // std::cout<<"\n\n";


    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);


    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"无修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"无修补相对误差 = "<<R0<<"\n";


    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稠密化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R1 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"稠密化修补相对误差 = "<<R1<<"\n";
    

    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"稀疏化修补相对误差 = "<<R2<<"\n";

    rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"rc无修补相对误差 = "<<R3<<"\n";


    rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R4 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"rc完全修补相对误差 = "<<R4<<"\n";

    rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R5 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"rc稀疏修补相对误差 = "<<R5<<"\n";


    rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,0.005,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R6 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"rc稀疏修补sp=1相对误差 = "<<R6<<"\n";

    return 0;
}




// Householder变换的QR分解,这个只支持方阵，按照标准定义 I-2UUT/UTU 生成的Q矩阵应该是方阵
template <typename T>
void xgeqrf_Householder_square(T inputMatrix[], T orthogonalMatrix[], T upperTriangularMatrix[], int rows, int cols,char type) {
	
	T orthogonalMatrix_tmp[rows*rows],inputMatrix_tmp[rows*cols];
	//拷贝一份输入矩阵 
	xcopy<T>(inputMatrix,inputMatrix_tmp,rows,cols);


	//初始化为单位矩阵 
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < rows; ++j) {
            if(i==j) orthogonalMatrix[i*rows+j]=1;
            else orthogonalMatrix[i*rows+j]=0;
        }
    }	
	
    for (int k = 0; k < cols-1; ++k) {
        // 复制输入矩阵的第k列到v向量
        T v[rows];
        for (int i = 0; i < rows; ++i) {
            v[i] = inputMatrix_tmp[i * cols + k];
        }
        for (int i = 0; i < k; ++i) {
            v[i] = 0;
        }
        
        
        // 计算Householder向量
        T normV = std::sqrt(dotProduct_v<T>(v, v, rows));
		v[k] -= normV;

        // 计算Householder矩阵H
        T vTv = dotProduct_v<T>(v, v, rows);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < rows; ++j) {
                if(vTv!=0){
                    orthogonalMatrix_tmp[i * rows + j] = -2 * (v[i] * v[j]) / vTv;
                }
                else{
                    orthogonalMatrix_tmp[i * rows + j] = 0;

                }
            }
            orthogonalMatrix_tmp[i * rows + i] += 1.0;
        }

        if(type == 'N'){
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xgemm<T>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
            xgemm<T>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		
        } else if(type == 'C'){
            type = 'c';
            const int digit = 4;
            float thread_f = 1.0/(float)(1<<6);
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xigemm<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols,thread_f,type);
            xigemm<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows,thread_f,type);	            
        }

        else {
            const int digit = 4;
            float thread_f = 1.0/(float)(1<<6);
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            rcxigemm<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols,thread_f,type);
            rcxigemm<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows,thread_f,type);	
        }
    


    }
    
	//保存转置的上三角矩阵和QT 
    xcopy<T>(inputMatrix_tmp,upperTriangularMatrix,rows,cols);		
	xtrans<T>(orthogonalMatrix,orthogonalMatrix,rows,rows);

	//保证下三角为0 
	for(int i=1;i<cols;i++){
		for(int j=0;j<i;j++){
			upperTriangularMatrix[i*cols+j]=1e-8;
		}
	}

}


int QRtest(){
    int rows=100,cols=100;
    int M = rows,N = cols,K = cols;

    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixQ_REF = (float *)malloc(sizeof(float) * N*K);
    float *matrixR_ref = (float *)malloc(sizeof(float) * M*N);
    float *matrix_resident = (float *)malloc(sizeof(float) * M*N);
    float *matrixA1 = (float *)malloc(sizeof(float) * M*N);
    float *matrixQ1 = (float *)malloc(sizeof(float) * M*N);
    float *matrixR1 = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,rows,cols,'e');

    auto start = std::chrono::high_resolution_clock::now();
    xgeqrf_Householder_square<float>(matrixA,matrixQ_REF,matrixR_ref, rows, cols,'N');
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"//reduce - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

    xgemm<float>(matrixQ_REF,matrixR_ref,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   


    float R0 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 
    std::cout<<"标准乘法相对误差 = "<<R0<<"\n";
    // printMatrix_h(matrixR_ref,rows,cols);
    // std::cout<<"\n\n";

    start = std::chrono::high_resolution_clock::now();

    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'o');

    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"//reduce - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;
  
    
    xgemm<float>(matrixQ1,matrixR1,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);  

    float R1 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 


    std::cout<<"普通量化乘法相对误差 = "<<R1<<"\n";
    // printMatrix_h(matrix_resident,rows,cols);
    // std::cout<<"\n\n";

    // std::cout<<"QREF"<<"\n";
    // printMatrix_h(matrixQ1,rows,cols);
    // std::cout<<"\n\n";
    // std::cout<<"RREF"<<"\n";
    // printMatrix_h(matrixR1,rows,cols);
    // std::cout<<"\n\n";

    start = std::chrono::high_resolution_clock::now();
    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'C');
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"//reduce - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

    xgemm<float>(matrixQ1,matrixR1,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   



    float R2 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 
    std::cout<<"稠密量化乘法相对误差 = "<<R2<<"\n";
    // printMatrix_h(matrixR1,rows,cols);
    // std::cout<<"\n\n";


    // get_error<float>(matrixR_ref,matrixR1,matrix_resident,rows,cols);   
    // std::cout<<"稠密量化R相对误差"<<"\n";
    // printMatrix_h(matrix_resident,rows,cols);
    // std::cout<<"\n\n";



    start = std::chrono::high_resolution_clock::now();
    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'x');
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"//reduce - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;


    xgemm<float>(matrixQ1,matrixR1,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   


    float R3 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 
    std::cout<<"稀疏量化乘法相对误差 = "<<R3<<"\n";


}


void series_wapper(int num,float thread_f,char type){
    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,type);
    generate_matrix<float>(matrixB,K,N,type);


    const int digit=8;

    // printMatrix(matrixA,M,K);
    // std::cout<<"\n\n";
    // printMatrix(matrixB,M,K);
    // std::cout<<"\n\n";


    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);



    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"无修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 

    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稠密化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R1 = get_Ferror<float>(matrixC,matrixCQ,M,N); 




    rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R5 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<";type = "<<type<<"; th = "<<thread_f<<"; precision="<<R5<<"; full="<<R1<<"; origin="<<R0<<"\n";


    return ;
}

void series_test(){
    std::vector<std::tuple<int, float, char>> inference_server_set = {
        // std::make_tuple(512, 0.9, 'k'),
        // std::make_tuple(512, 0.5, 'k'),
        // std::make_tuple(512, 0.1, 'k'),
        // std::make_tuple(512, 0.001, 'k'),
        // std::make_tuple(512, 0.002, 'k'),
        // std::make_tuple(512, 0.005, 'k'),
        // std::make_tuple(512, 0.01, 'k'),
        // std::make_tuple(512, 0.02, 'k'),
        // std::make_tuple(512, 0.05, 'k'),
        

        // std::make_tuple(512, 0.001, 'e'),
        // std::make_tuple(512, 0.002, 'e'),
        // std::make_tuple(512, 0.005, 'e'),
        // std::make_tuple(512, 0.01, 'e'),
        // std::make_tuple(512, 0.02, 'e'),
        // std::make_tuple(512, 0.05, 'e'),
        // std::make_tuple(512, 0.001, 'u'),
        // std::make_tuple(512, 0.002, 'u'),
        // std::make_tuple(512, 0.005, 'u'),
        // std::make_tuple(512, 0.01, 'u'),
        // std::make_tuple(512, 0.02, 'u'),
        // std::make_tuple(512, 0.05, 'u'),
        // std::make_tuple(512, 0.001, 'n'),
        // std::make_tuple(512, 0.002, 'n'),
        // std::make_tuple(512, 0.005, 'n'),
        // std::make_tuple(512, 0.01, 'n'),
        // std::make_tuple(512, 0.02, 'n'),
        // std::make_tuple(512, 0.03, 'n'),
        // std::make_tuple(512, 0.04, 'n'),
        // std::make_tuple(512, 0.05, 'n'),


        // std::make_tuple(512, 0.001, 'p'),
        // std::make_tuple(512, 0.002, 'p'),
        // std::make_tuple(512, 0.005, 'p'),
        // std::make_tuple(512, 0.01, 'p'),
        // std::make_tuple(512, 0.02, 'p'),
        // std::make_tuple(512, 0.05, 'p'),

        // std::make_tuple(256, 1, 'k'),
        std::make_tuple(256, 0.8, 'k'),
        // std::make_tuple(256, 0.6, 'k'),
        // std::make_tuple(256, 0.4, 'k'),
        // std::make_tuple(256, 0.2, 'k'),
        // std::make_tuple(256, 0.05, 'k'),

        // std::make_tuple(256, 1, 'e'),
        std::make_tuple(256, 0.8, 'e'),
        // std::make_tuple(256, 0.5, 'e'),
        // std::make_tuple(256, 0.2, 'e'),
        // std::make_tuple(256, 0.1, 'e'),
        // std::make_tuple(256, 0.05, 'e'),


        // std::make_tuple(256, 1, 'u'),
        std::make_tuple(256, 0.8, 'u'),
        // std::make_tuple(256, 0.5, 'u'),
        // std::make_tuple(256, 0.2, 'u'),
        // std::make_tuple(256, 0.1, 'u'),
        // std::make_tuple(256, 0.05, 'u'),

        // std::make_tuple(256, 1, 'n'),
        std::make_tuple(256, 0.8, 'n'),
        // std::make_tuple(256, 0.5, 'n'),
        // std::make_tuple(256, 0.2, 'n'),
        // std::make_tuple(256, 0.1, 'n'),
        // std::make_tuple(256, 0.05, 'n'),

        // std::make_tuple(256, 1, 'p'),
        std::make_tuple(256, 0.5, 'p'),
        // std::make_tuple(256, 0.5, 'p'),
        // std::make_tuple(256, 0.2, 'p'),
        // std::make_tuple(256, 0.1, 'p'),
        // std::make_tuple(256, 0.05, 'p'),
    };
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, th, type) = problem;
        series_wapper(num,th,type);
    }
}

void density_wapper(int num,char type){
    int N=num,M=num,K=num;
    float thread_f=0;
    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,type);
    generate_matrix<float>(matrixB,K,N,type);

    float array[6] = {1,0.8,0.5,0.2,0.1};
    for(int i=0;i<5;i++){
        thread_f= array[i];
 
        const int digit=4;
        rcxigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
        //计算修复误差后的量化矩阵乘法误差
        float R5 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<";type = "<<type<<"; th = "<<thread_f<<"\n";


    }

    return ;
}

void density_test(){
    std::vector<std::tuple<int, char>> inference_server_set = {


        std::make_tuple(512,  'k'),

        std::make_tuple(512,'e'),

        std::make_tuple(512,  'u'),

        std::make_tuple(512, 'n'),

        std::make_tuple(512, 'p'),

    };
    for (const auto &problem : inference_server_set) {
        int num;
        float th;
        char type;
        std::tie(num, type) = problem;
        density_wapper(num,type);
    }
}


void cpu_gemm_test(){
    int num = 2048;
    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);

    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');

    float judge=0.7;
    reduce_Matrix(matrixA,matrixC,M,K, judge ,'r');


    auto start = std::chrono::high_resolution_clock::now();
    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::micro> diff = end - start;
    int time  = diff.count();
    std::cout<<"//float gemm - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;



    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*K);
    int8_t *matrixB8 = (int8_t *)malloc(sizeof(int8_t) * N*K);
    int8_t *matrixC8 = (int8_t *)malloc(sizeof(int8_t) * N*K);

    for(int i=0;i<num*num;i++){
        matrixA8[i] = matrixA[i]*64;
        matrixB8[i] = matrixB[i]*64;
    }

     start = std::chrono::high_resolution_clock::now();
    //计算float和int矩阵乘法得到结果矩阵
    xgemm<int8_t>(matrixA8,matrixB8,matrixC8,M,K,K,N);
     end = std::chrono::high_resolution_clock::now();
     diff = end - start;
     time  = diff.count();
    std::cout<<"//int8 gemm - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;


     start = std::chrono::high_resolution_clock::now();
    //计算float和int矩阵乘法得到结果矩阵
    xmadd<int8_t>(matrixA8,matrixB8,matrixC8,M,K);
     end = std::chrono::high_resolution_clock::now();
     diff = end - start;
     time  = diff.count();
    std::cout<<"//int8 add - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;




    double *matrixA_db = (double *)malloc(sizeof(double) * M*K);
    double *matrixB_db = (double *)malloc(sizeof(double) * N*K);
    double *matrixC_db = (double *)malloc(sizeof(double) * N*K);

    for(int i=0;i<num*num;i++){
        matrixA_db[i] = matrixA[i]*64;
        matrixB_db[i] = matrixB[i]*64;
    }

    start = std::chrono::high_resolution_clock::now();
    //计算float和int矩阵乘法得到结果矩阵
    xgemm<double>(matrixA_db,matrixB_db,matrixC_db,M,K,K,N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count();
    std::cout<<"//double gemm - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;



    int *matrixA32 = (int *)malloc(sizeof(int) * M*K);
    int *matrixB32 = (int *)malloc(sizeof(int) * N*K);
    int *matrixC32 = (int *)malloc(sizeof(int) * N*K);

    for(int i=0;i<num*num;i++){
        matrixA32[i] = matrixA[i]*128;
        matrixB32[i] = matrixB[i]*128;
    }
    int nnzA = get_nnz<int>(matrixA32, M, K);
    int* valuesA = new int[nnzA];
    int* colIndexA = new int[nnzA];
    int* rowPtrA = new int[M + 1];
    denseToCSR(matrixA32, valuesA,colIndexA,rowPtrA, M, K);

    start = std::chrono::high_resolution_clock::now();
    sspmm<int>(valuesA,colIndexA,rowPtrA, matrixB32, matrixC32,M, K,K, N);
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count();

    float sp = (float)nnzA/((float)M*K);
    std::cout<<"//int32 spmm -  sp = " << sp<<" cpu time:"<< std::fixed << std::setprecision(6) << time << std::endl;




}


void spmm_test(){
    int num = 512;
    int N=num,M=num,K=num;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixA1 = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);

    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');

    float judge=0;
    auto start = std::chrono::high_resolution_clock::now();
    //计算float和int矩阵乘法得到结果矩阵
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double,std::micro> diff = end - start;
    int time  = diff.count();

    for(int i=0;i<100;i++){

        judge+=0.01;
        int *matrixA32 = (int *)malloc(sizeof(int) * M*K);
        int *matrixB32 = (int *)malloc(sizeof(int) * N*K);
        int *matrixC32 = (int *)malloc(sizeof(int) * N*K);
        reduce_Matrix_out(matrixA,matrixA1,M,K, judge ,'r');

        for(int i=0;i<num*num;i++){
            matrixA32[i] = matrixA1[i]*128;
            matrixB32[i] = matrixB[i]*128;
        }


        int nnzA = get_nnz<int>(matrixA32, M, K);
        int* valuesA = new int[nnzA];
        int* colIndexA = new int[nnzA];
        int* rowPtrA = new int[M + 1];
        denseToCSR(matrixA32, valuesA,colIndexA,rowPtrA, M, K);

        start = std::chrono::high_resolution_clock::now();
        sspmm<int>(valuesA,colIndexA,rowPtrA, matrixB32, matrixC32,M, K,K, N);
        end = std::chrono::high_resolution_clock::now();
        diff = end - start;
        time  = diff.count();

        float sp = (float)nnzA/((float)M*K);
        std::cout<<"//int32 spmm -  sp = " << sp<<" cpu time:"<< std::fixed << std::setprecision(6) << time << std::endl;

    }

}



int main(){
    //稀疏残差qr分解测试
    // QRtest();

    //不同量化矩阵乘测试
    // test1();

    //稀疏残差子操作quant，reduce GPU测试
    // int num1[12] = {128,256,512,1024,2048,4096};
    // for(int i=0;i<6;i++){
    //     M=num1[i];
    //     N=num1[i];
    //     K=num1[i];
    //     cuda_quant_speedTest();
    //     cuda_reduce_speedTest();
    // }


    //单纯的gemm测试
    // int num1[12] = {128,256,512,1024,2048};
    // for(int i=0;i<8;i++){
    //     M=num1[i];
    //     N=num1[i];
    //     K=num1[i];
    //     gemm_cpu_test();
    // }

    //打印测试
    // print_test();

    //稀疏化精度测试
    // error_th_N_test();

    // series_test();

    // cpu_gemm_test();

    //cpu spmm速度测试
    // spmm_test();

    density_test();
// 
    // series_test();



}