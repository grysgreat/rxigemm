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
#include <chrono>
#include <iomanip>
#include <cmath>
#include <cstring>
#include <omp.h>

const int max_omp_thread = omp_get_max_threads();

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
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::chi_squared_distribution<> dis(0.2);
                matrix[i*cols+j] = dis(gen);
            }
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
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda);
            }
        }
    }
    else if(type == 'd'){
        #pragma omp parallel for num_threads(max_omp_thread)
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

    #pragma omp parallel for reduction(max:maxM)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            maxM = max(maxM,std::abs(matrix[i*cols+j]));
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

    if(type == 'r'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            T judge = Mlam_k*vec[i];
            //printf("judge = %f\n",judge);
            for(int j=0;j<cols;j++){
                if(matrix[i*cols+j]<judge) {
                    matrix[i*cols+j]=0;
                }
            }
        }        
    }
    if(type == 'c'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k*vec[j];
                if(matrix[i*cols+j]<judge) {
                    matrix[i*cols+j]=0;

                }
            }
        }        
    }
    // float spasity = 1-(cnt/(float)(rows*cols));
    // //printf("spasity = %f\n",spasity);
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

            printf("%f\n",(float)nnzA/(float)(rowsA*colsA));

            int nnzB = get_nnz<int>(B_int, rowsA, colsA);

            int* valuesA = new int[nnzA];
            int* colIndexA = new int[nnzA];
            int* rowPtrA = new int[rowsA + 1];
            denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


            sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);


            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR1!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


            xtrans<int>(B_int,B_int,rowsB,colsB);
            xtrans<int>(AR_int,AR_int,rowsA, colsA);
            int* valuesB = new int[nnzB];
            int* colIndexB = new int[nnzB];
            int* rowPtrB = new int[colsB + 1];
            denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
            sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
            xtrans<int>(C_int,C_int,colsB,rowsA);

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差

            if(lambdaCR2!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}

// 带打印的
template <typename T,int digit>
void xigemm_with_print(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

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


    std::cout<<"A矩阵最大值 = "<<max_mA<<"\n";
    std::cout<<"lambda A = "<<lambdaA<<"\n";
    std::cout<<"量化A矩阵"<<"\n";
    printMatrix_int(A_int,rowsA,rowsA);
    std::cout<<"\n\n";


    std::cout<<"B矩阵最大值 = "<<max_mB<<"\n";
    std::cout<<"lambda A = "<<lambdaB<<"\n";
    std::cout<<"量化B矩阵"<<"\n";
    printMatrix_int(B_int,rowsA,rowsA);
    std::cout<<"\n\n";

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');


    std::cout<<"直接量化C矩阵"<<"\n";
    printMatrix(C_buffer,rowsA,rowsA);
    std::cout<<"\n\n";

    if(type == 'c' || type == 'x'){
        
        //对量化后的int矩阵反量化得到A',B'
        quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
        quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

        std::cout<<"反量化A矩阵"<<"\n";
        printMatrix(A_p,rowsA,rowsA);
        std::cout<<"\n\n";

        std::cout<<"反量化B矩阵"<<"\n";
        printMatrix(B_p,rowsA,rowsA);
        std::cout<<"\n\n";

        //计算full size的残差矩阵
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);

        std::cout<<"残差A矩阵"<<"\n";
        printMatrix(A_p,rowsA,rowsA);
        std::cout<<"\n\n";

        std::cout<<"残差B矩阵"<<"\n";
        printMatrix(B_p,rowsA,rowsA);
        std::cout<<"\n\n";


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
            get_avg_vec<T>(C_buffer,C_cols,rowsA,colsB,'c');
            
            std::cout<<"C矩阵行平均值"<<"\n";
            printMatrix(C_rows,1,rowsA);
            std::cout<<"\n\n";            

            std::cout<<"C矩阵列平均值"<<"\n";
            printMatrix(C_cols,1,rowsA);
            std::cout<<"\n\n";      

            reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,C_cols,rowsB,rowsB, ml_kB ,'c');

            std::cout<<"reduceA 矩阵"<<"\n";
            printMatrix(A_copy,rowsA,rowsA);
            std::cout<<"\n\n";            

            std::cout<<"reduceB 矩阵"<<"\n";
            printMatrix(B_copy,rowsA,rowsA);
            std::cout<<"\n\n";                  

            // float sparsity = get_sparsity(B_copy,rowsB,rowsB);
            // std::cout<<"sparsity = "<<sparsity<<"\n";
            // sparsity = get_sparsity(A_copy,rowsA,colsA);
            // std::cout<<"sparsity = "<<sparsity<<"\n";

        // std::cout<<"otho"<<"\n";
        // printMatrix_h(A_copy,rowsA,colsA);
        // std::cout<<"\n\n";

            //对新AB进行量化
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');


            std::cout<<"lambda A = "<<lambdaAnew<<"\n";
            std::cout<<"量化稀疏A'矩阵"<<"\n";
            printMatrix_int(A_int,rowsA,rowsA);
            std::cout<<"\n\n";


            std::cout<<"lambda B = "<<lambdaBnew<<"\n";
            std::cout<<"量化稀疏B'矩阵"<<"\n";
            printMatrix_int(B_int,rowsA,rowsA);
            std::cout<<"\n\n";
            
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
            if(lambdaCR1!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


            xtrans<int>(B_int,B_int,rowsB,colsB);
            xtrans<int>(AR_int,AR_int,rowsA, colsA);
            int* valuesB = new int[nnzB];
            int* colIndexB = new int[nnzB];
            int* rowPtrB = new int[colsB + 1];
            denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
            sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
            xtrans<int>(C_int,C_int,colsB,rowsA);

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            if(lambdaCR2!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //对int误差修复矩阵反量化得到误差修复矩阵float
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //使用修复矩阵补充误差
            xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }






    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}



// 定义矩阵乘矩阵函数---- 指数切分法
template <typename T,int digit>
void xigemm_e(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {




    T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);
    T *A_0 = (T *)malloc(sizeof(T) * rowsA * colsA);
    T *B_0 = (T *)malloc(sizeof(T) * rowsB * colsB);

    T *A_B_0 = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_0_B = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_B = (T *)malloc(sizeof(T) * rowsA * colsB);
    T *A_0_B_0 = (T *)malloc(sizeof(T) * rowsA * colsB);

    xcopy<float>(A,A_copy,rowsA,colsA);
    xcopy<float>(B,B_copy,rowsB,colsB);
    T max_mA =get_max<T>(A,rowsA,colsA);
    T max_mB =get_max<T>(B,rowsB,colsB);    
    T min_mA =get_min<T>(A,rowsA,colsA);
    T min_mB =get_min<T>(B,rowsB,colsB);    




    T judgeA = exp((log(max_mA) + log(min_mA))/2);
    T judgeB = exp((log(max_mB) + log(min_mB))/2);


    splitMatrix<float>(A_copy,A_0,rowsA,colsA,judgeA);
    splitMatrix<float>(B_copy,B_0,rowsB,colsB,judgeB);


    // printMatrix_h(B_0,rowsB,colsB);
    // std::cout<<"\n\n";
    float sa = get_sparsity(A_0,rowsA,colsA);
    float sb = get_sparsity(B_0,rowsB,colsB);
    //printf("sparsity = %f,%f , judgeA = %f, judgeB = %f,max_mA = %f, min_mA = %f\n", sa, sb,judgeA,judgeB,max_mA,min_mA);

    float thread_f = 1.0/(float)(1<<6);
    xigemm<T,digit>(A_copy, B_copy,A_B,rowsA,colsA,rowsB,colsB,thread_f,'o');
    xigemm<T,digit>(A_0, B_copy, A_0_B,rowsA,colsA,rowsB,colsB,thread_f,'o');
    xigemm<T,digit>(A_copy ,B_0,A_B_0,rowsA,colsA,rowsB,colsB,thread_f,'o');
    xigemm<T,digit>(A_0,B_0,A_0_B_0,rowsA,colsA,rowsB,colsB,thread_f,'o');

    // xgemm<T>(A_copy,B_copy,A_B,rowsA,colsA,rowsB,colsB);
    // xgemm<T>(A_0, B_copy, A_0_B,rowsA,colsA,rowsB,colsB);
    // xgemm<T>(A_copy, B_0, A_B_0,rowsA,colsA,rowsB,colsB);
    // xgemm<T>(A_0,B_0,A_0_B_0,rowsA,colsA,rowsB,colsB);



    xmadd<T>(A_B, A_0_B, C,rowsA,colsB);
    xmadd<T>(A_B_0,  C, C,rowsA,colsB);
    xmadd<T>(C,  A_0_B_0, C,rowsA,colsB);

}


// 定义矩阵乘矩阵函数---- 残差法
template <typename T,int digit>
void xigemm_with_low(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

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


    int max_int = 1<<(digit-1) - 1;
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

    {
        
        //对量化后的int矩阵反量化得到A',B'
        quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
        quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

        //计算full size的残差矩阵
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);

        //对残差矩阵进行量化
        T max_mAR =get_max<T>(A_p,rowsA,colsA);
        T max_mBR =get_max<T>(B_p,rowsB,colsB);

        int max_int_new = 1<<(12-1) - 1;
        T lambdaAR = (T)max_int_new/max_mAR;
        T lambdaBR = (T)max_int_new/max_mBR;
        T lambdaA = (T)max_int/max_mA;
        T lambdaB = (T)max_int/max_mB;

        //对A,B残差矩阵进行直接量化
        quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
        quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');

        quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
        quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

        T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
 
        //对int误差修复矩阵反量化得到误差修复矩阵float
        T lambdaCR1 = lambdaAnew*lambdaBR;
        T lambdaCR2 = lambdaAR*lambdaBnew;

        xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
        quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
        //使用修复矩阵补充误差
        xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        
        xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
        quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
        //使用修复矩阵补充误差
        xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        
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
                orthogonalMatrix_tmp[i * rows + j] = -2 * (v[i] * v[j]) / vTv;
            }
            orthogonalMatrix_tmp[i * rows + i] += 1.0;
        }

        if(type == 'N'){
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xgemm<T>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
            xgemm<T>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		
        }
        else if(type =='e') {
            const int digit = 8;
            xigemm_e<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
            xigemm_e<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		
        }
        else {
            const int digit = 8;
            float thread_f = 1.0/(float)(1<<6);
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xigemm<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols,thread_f,type);
            xigemm<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows,thread_f,type);	
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



int test1(){
    int N=800,M=800,K=200;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,'k');
    generate_matrix<float>(matrixB,K,N,'k');


    const int digit=8;
    float thread_f = 0.01;




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


    xigemm_with_low<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // std::cout<<"稀疏化修补"<<"\n";
    // printMatrix(matrixR,M,K);
    // std::cout<<"\n\n";
    float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"with low 相对误差 = "<<R3<<"\n";

    // xigemm_e<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N);
    // //计算修复误差后的量化矩阵乘法误差
    // get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
    // float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    // std::cout<<"指数切分修补相对误差 = "<<R3<<"\n";




    // printMatrix(matrixAp,M,K);
    // std::cout<<"\n\n";
    // std::cout<<"稀疏A矩阵"<<"\n";
    // printMatrix(matrixA,M,K);
    // std::cout<<"\n\n";
    return 0;
}


int QRtest(){
    int rows=100,cols=100;

    float matrixA[10000],matrixQ_REF[10000],matrixR_ref[10000],matrix_resident[10000],matrixA1[10000],matrixQ1[10000],matrixR1[10000];

    generate_matrix<float>(matrixA,rows,cols,'u');

    xgeqrf_Householder_square<float>(matrixA,matrixQ_REF,matrixR_ref, rows, cols,'N');
    xgemm<float>(matrixQ_REF,matrixR_ref,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   


    float R0 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 
    std::cout<<"标准乘法相对误差 = "<<R0<<"\n";
    // printMatrix_h(matrixR_ref,rows,cols);
    // std::cout<<"\n\n";

    auto start = std::chrono::high_resolution_clock::now();

    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'o');

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
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
    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'c');
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

    // printMatrix_h(matrix_resident,rows,cols);
    // std::cout<<"\n\n";


    xgeqrf_Householder_square<float>(matrixA,matrixQ1,matrixR1, rows, cols,'e');
    xgemm<float>(matrixQ1,matrixR1,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   

    float R4 = get_Ferror<float>(matrixA,matrixA1,rows,cols); 
    std::cout<<"指数切分量化乘法相对误差 = "<<R4<<"\n";
}

__global__ void quantitize_cuda8(float * matrix_in,int8_t * matrix_out,int nx,int ny,float lambda)
{
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*ny;

    matrix_out[idx] = (matrix_in[idx]*lambda);

}

__global__ void binaryMatrixMultiply(float* A, float* B, float* C, int m, int n, int p) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < p) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            if (A[row * n + k] == 1.0f) {
                sum += B[k * p + col];
            }
        }
        C[row * p + col] = sum;
    }
}

#define num 8192
int M=num,N=num,K=num;
int cuda_quant_speedTest(){
    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');
    generate_matrix<float>(matrixB,M,N,'u');

    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();

    float max_mAR =get_max<float>(matrixA,M,N);
    const int max_int = 1<<(8-1) - 1;
    float lambdaAnew = (float)max_int/max_mAR;
    quantitize<float,int8_t>(matrixA,matrixA8,M,N,lambdaAnew,'q');
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//quant - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

    // for(int i=0;i<=10;i++){
    //     int a = matrixA8[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout<<std::endl;
    float *matrixA_dev;
    float *matrixB_dev;
    float *matrixC_dev;
    int8_t* matrixA8_dev;
    int8_t* matrixA8_dev_ans= (int8_t *)malloc(sizeof(int8_t) * M*N);
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixB_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixC_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&matrixA8_dev, sizeof(int8_t) * M*N);


    // start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_dev, matrixB, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    // end = std::chrono::high_resolution_clock::now();
    // diff = end - start;
    // time  = diff.count()*1000*1000;
    // std::cout <<"cuda copt time:"<< std::fixed << std::setprecision(6) << time << std::endl;





    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((M)/block.x, (N)/block.y);

    start = std::chrono::high_resolution_clock::now();
    quantitize_cuda8<<<grid,block>>>(matrixA_dev,matrixA8_dev,M,N,lambdaAnew);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//quant - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

    cudaMemcpy(matrixA8_dev_ans, matrixA8_dev, sizeof(int8_t) * M*N, cudaMemcpyDeviceToHost);
    

    // for(int i=0;i<=10;i++){
    //     int a = matrixA8_dev_ans[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout<<std::endl;

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    start = std::chrono::high_resolution_clock::now();
    // 调用kernel
    binaryMatrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(matrixA_dev, matrixB_dev, matrixC_dev, M, N, K);    
    cudaDeviceSynchronize();
    
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//bgemm - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

}


__global__ void cuda_reduce_Matrix(float* matrix,float* vec,int rows,int cols,float Mlam_k){
    
    int ix = threadIdx.x+blockDim.x*blockIdx.x;
    int iy = threadIdx.y+blockDim.y*blockIdx.y;
    int idx = ix+iy*rows;
    if (ix<cols && iy<rows)
    {
        //if(idx<10)printf("%f,%f\n",vec[iy],matrix[idx]);
        float judge = Mlam_k*vec[iy];
        matrix[idx] = matrix[idx]<judge?0:matrix[idx];
    }
}



int cuda_reduce_speedTest(){
    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    float *matrixA_cpy = (float *)malloc(sizeof(float) * M*N);
    float *A_rows = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');

    xcopy<float>(matrixA,matrixA_cpy,M,N);
    
    float ml_kA = 1;
    get_avg_vec<float>(matrixA,A_rows,M,N,'r');

    auto start = std::chrono::high_resolution_clock::now();

    reduce_Matrix(matrixA_cpy,A_rows,M,N, ml_kA ,'r');

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//reduce - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;
    // for(int i=0;i<=10;i++){
    //     float a = matrixA_cpy[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout << std::endl;

    float *matrixA_dev;
    float* A_rows_dev;
    float* matrixA_dev_ans= (float *)malloc(sizeof(float) * M*N);
    cudaMalloc((void**)&matrixA_dev, sizeof(float) * M*N);
    cudaMalloc((void**)&A_rows_dev, sizeof(float) * M*N);

    cudaMemcpy(matrixA_dev, matrixA, sizeof(float) * M*N, cudaMemcpyHostToDevice);
    cudaMemcpy(A_rows_dev, A_rows, sizeof(float) * M*N, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    //二维线程网格，128×128
    dim3 grid((M)/block.x, (N)/block.y);

    start = std::chrono::high_resolution_clock::now();
    cuda_reduce_Matrix<<<grid,block>>>(matrixA_dev,A_rows_dev,M,N, ml_kA);
    cudaDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//reduce - gpu time:" << std::fixed << std::setprecision(6) << time << std::endl;

    cudaMemcpy(matrixA_dev_ans, matrixA_dev, sizeof(float) * M*N, cudaMemcpyDeviceToHost);
    
    // for(int i=0;i<=10;i++){
    //     float a = matrixA_dev_ans[i];
    //     std::cout<<a<<" ";
    // }
    // std::cout << std::endl;
}


// 定义矩阵乘矩阵函数
void xgemm_i8(const int8_t A[], const int8_t B[], int32_t C[], int rowsA, int colsA, int rowsB, int colsB) {

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    int32_t* tmp =(int32_t *)malloc(sizeof(int32_t) * rowsA*colsB);
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            // 初始化结果矩阵中的元素为0
            tmp[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
                // 矩阵乘法的累加步骤
                tmp[i * colsB + j] += static_cast<int32_t>(A[i * colsA + k]) * static_cast<int32_t>(B[k * colsB + j]);
            }
        }
    }
    
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            C[i * colsB + j]  = tmp[i * colsB + j];
		}
	}
    
}


int gemm_cpu_test(){

    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * K*N);

    float *matrixC = (float *)malloc(sizeof(float) * K*N);

    int32_t  *matrixC32 = (int32_t *)malloc(sizeof(int32_t) * M*N);
    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');


    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*K);
    int8_t *matrixB8 = (int8_t *)malloc(sizeof(int8_t) * K*N);    
    const int max_int = 1<<(8-1) - 1;
    float lambdaA = (float)max_int;
    quantitize<float,int8_t>(matrixA,matrixA8, M, K,lambdaA,'q');
    quantitize<float,int8_t>(matrixB,matrixB8, K, N,lambdaA,'q');

    auto start = std::chrono::high_resolution_clock::now();
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//" << std::fixed << std::setprecision(6) << time << std::endl;

     start = std::chrono::high_resolution_clock::now();
    xgemm_i8(matrixA8,matrixB8,matrixC32,M,K,K,N);
     end = std::chrono::high_resolution_clock::now();
    diff = end - start;
     time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//" << std::fixed << std::setprecision(6) << time << std::endl;


}




//输出测试，跑每一步数据，需要去掉上面注释
int print_test(){
    int N=3,M=3,K=3;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');

    std::cout<<"输入A矩阵"<<"\n";
    printMatrix(matrixA,M,K);
    std::cout<<"\n\n";
    std::cout<<"输入B矩阵"<<"\n";
    printMatrix(matrixB,M,K);
    std::cout<<"\n\n";


    const int digit=8;
    float thread_f = 0.05;

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);


    xigemm_with_print<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
    //计算修复误差后的量化矩阵乘法误差
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   

    std::cout<<"floatC矩阵"<<"\n";
    printMatrix(matrixC,M,K);
    std::cout<<"\n\n";

    std::cout<<"稀疏量化C矩阵"<<"\n";
    printMatrix(matrixCQ,M,K);
    std::cout<<"\n\n";

    float mr = get_max<float>(matrixR,M,N);
    std::cout<<"稀疏化修补绝对误差 = "<<mr<<"\n";
    float R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"稀疏化修补相对误差 = "<<R2<<"\n";


    xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
    get_error<float>(matrixC,matrixCQ,matrixR,M,N);   

    R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
    std::cout<<"直接量化相对误差 = "<<R2<<"\n";
    mr = get_max<float>(matrixR,M,N);
    std::cout<<"直接量化绝对误差 = "<<mr<<"\n";

    return 0;
}


//不同精度和阈值下的误差
int error_th_N_test(){

    int size[10] = {1024,4096};
    float fp[10] = {1,0.5,0.1,0.01,0.001};
    int num1 = 512;
    int N=num1,M=num1,K=num1;


    float *matrixA = (float *)malloc(sizeof(float) * M*K);
    float *matrixB = (float *)malloc(sizeof(float) * N*K);
    float *matrixC = (float *)malloc(sizeof(float) * M*N);
    float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
    float *matrixR = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,K,'u');
    generate_matrix<float>(matrixB,K,N,'u');


    //计算float和int矩阵乘法得到结果矩阵
    xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);

    float thread_f = 0.0001;

    for(int f=0;f<5;f++){
        thread_f = fp[f];
        const int digit1 = 4;
        xigemm<float,digit1>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
        float R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit1<<";thread = "<<thread_f<<";相对误差 = "<<R0<<"\n";

        thread_f = fp[f];
        const int digit2 = 8;
        xigemm<float,digit2>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
        R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit2<<";thread = "<<thread_f<<";相对误差 = "<<R0<<"\n";

    }
    

        thread_f = fp[0];
        const int digit1 = 4;
        xigemm<float,digit1>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
        float R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit1<<";无误差补偿相对误差 = "<<R0<<"\n";

        thread_f = fp[0];

        xigemm<float,digit1>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
         R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit1<<";完全残差补偿相对误差 = "<<R0<<"\n";

        thread_f = fp[0];
        const int digit2 = 8;
        xigemm<float,digit2>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
         R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit2<<";无误差补偿相对误差 = "<<R0<<"\n";

        thread_f = fp[0];

        xigemm<float,digit2>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
        //计算修复误差后的量化矩阵乘法误差
        get_error<float>(matrixC,matrixCQ,matrixR,M,N);               
         R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
        std::cout<<"digit = "<<digit2<<";完全残差补偿相对误差 = "<<R0<<"\n";



    return 0;
}



int cpu_quant_speedTest(){
    float *matrixA = (float *)malloc(sizeof(float) * M*N);
    int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*N);
    float *matrixB = (float *)malloc(sizeof(float) * M*N);
    generate_matrix<float>(matrixA,M,N,'u');
    generate_matrix<float>(matrixB,M,N,'u');



    // 开始计时，使用chrono计时，不支持其它计时方式
    auto start = std::chrono::high_resolution_clock::now();
    float max_mAR =get_max<float>(matrixA,M,N);
    const int max_int = 1<<(8-1) - 1;
    float lambdaAnew = (float)max_int/max_mAR;
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    int time  = diff.count()*1000*1000;
    std::cout<<"size="<<M<<"//max - cpu time:" << std::fixed << std::setprecision(6) << time;



    start = std::chrono::high_resolution_clock::now();
    quantitize<float,int8_t>(matrixA,matrixA8,M,N,lambdaAnew,'q');
     end = std::chrono::high_resolution_clock::now();
     diff = end - start;
     time  = diff.count()*1000*1000;
    std::cout<<"//quant - cpu time:" << std::fixed << std::setprecision(6) << time;

    float ml_kA = 1;
    get_avg_vec<float>(matrixA,matrixB,M,N,'r');
     start = std::chrono::high_resolution_clock::now();
    reduce_Matrix(matrixA,matrixB,M,N, ml_kA ,'r');
     end = std::chrono::high_resolution_clock::now();
    diff = end - start;
     time  = diff.count()*1000*1000;
    std::cout<<"//reduce - cpu time:" << std::fixed << std::setprecision(6) << time << std::endl;
 
}



int main(){

    //cuda_quant_speedTest();
    //稀疏残差qr分解测试
    //QRtest();

    //不同量化矩阵乘测试
    // test1();

    //稀疏残差子操作quant，reduce GPU测试
    int num1[12] = {128,256,512,1024,2048,4096,8192,16384};
    for(int i=0;i<7;i++){
        M=num1[i];
        N=num1[i];
        K=num1[i];
        //cpu_quant_speedTest();
        cuda_quant_speedTest();
        // cuda_reduce_speedTest();
    }


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

}