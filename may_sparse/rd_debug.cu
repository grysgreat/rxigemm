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
            if(i==j) matrix[i*cols+j]-=0;
            else  matrix[i*cols+j]/=1;
        }
    }
}


// 定义矩阵乘矩阵函数
template <typename T,int digit>
void xigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type,int print) {

        if(print==0){
            std::cout<<"错误b矩阵-before1"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";

        }

    // 确保可以进行矩阵乘法的尺寸
    if (colsA != rowsB) {
        std::cerr << "无法进行矩阵乘法，尺寸不匹配。" << std::endl;
        return;
    }
    
    int A_int[1000],B_int[1000],C_int[1000],AR_int[1000],BR_int[1000];
    T A_p[1000],B_p[1000],C_rows[1000],C_cols[1000],A_copy[1000],B_copy[1000],C_copy[1000],C_buffer[1000];

    xcopy<T>(A,A_copy, rowsA, colsA);
    xcopy<T>(B,B_copy, rowsB, colsB);

        if(print==0){
            std::cout<<"错误b矩阵-before2"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";

        }

    const int max_int = 1<<(digit-1) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    //对A,B矩阵进行直接量化
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

        if(print==0){
            std::cout<<"错误b矩阵-before3"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";

        }

    //计算float和int矩阵乘法得到结果矩阵
    xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


        if(print==0){
            std::cout<<"错误b矩阵-before4"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";

        }


    //对结果矩阵C_INT进行反量化得到C'
    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

        if(print==0){
            std::cout<<"错误b矩阵-before5"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";

        }


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

        if(print==0){
            std::cout<<"错误b矩阵-in"<<"\n";
            printMatrix(B,rowsA,rowsA);
            std::cout<<"\n\n";
            std::cout<<"错误bp残差矩阵"<<"\n";
            printMatrix(B_p,rowsA,rowsA);
            std::cout<<"\n\n";
        }
        T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
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
        xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        
        xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
        quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
        //使用修复矩阵补充误差
        xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        xcopy<T>(C_buffer,C, rowsA, colsB);

    }



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

        int print = 0;
        if(type == 'N'){

            if(k==print){
                std::cout<<"正确A矩阵"<<"\n";
                printMatrix_h(orthogonalMatrix_tmp,rows,rows);
                std::cout<<"\n\n";           
                std::cout<<"正确B矩阵"<<"\n";
                printMatrix_h(inputMatrix_tmp,rows,cols);
                std::cout<<"\n\n";   
            }
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xgemm<T>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
            xgemm<T>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		

            if(k==print){
                std::cout<<"正确C矩阵"<<"\n";
                printMatrix_h(inputMatrix_tmp,rows,cols);
                std::cout<<"\n\n";   
     
            }
        }
        else {
            if(k==print){
                std::cout<<"错误A矩阵"<<"\n";
                printMatrix_h(orthogonalMatrix_tmp,rows,rows);
                std::cout<<"\n\n";           
                std::cout<<"错误B矩阵"<<"\n";
                printMatrix_h(inputMatrix_tmp,rows,cols);
                std::cout<<"\n\n";   
            }            

            const int digit = 14;
            float thread_f = 1.0/(float)(1<<12);
            //将正交矩阵 orthogonalMatrix_tmp 应用到 inputMatrix_tmp 中，并保存正交矩阵乘积结果 
            xigemm<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols,thread_f,type,k);
            xigemm<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows,thread_f,type,k);		            
        
            if(k==print){
                std::cout<<"错误C矩阵"<<"\n";
                printMatrix_h(inputMatrix_tmp,rows,cols);
                std::cout<<"\n\n";   
     
            }     

        
        }
    


    }
    
	//保存转置的上三角矩阵和QT 
    xcopy<T>(inputMatrix_tmp,upperTriangularMatrix,rows,cols);		
	xtrans<T>(orthogonalMatrix,orthogonalMatrix,rows,rows);
	
}



// int test1(){
//     int N=10,M=10,K=10;

//     float matrixA[10000],matrixB[10000],matrixC[10000],matrixCQ[10000],matrixR[10000];

//     generate_matrix<float>(matrixA,M,K);
//     generate_matrix<float>(matrixB,K,N);


//     const int digit=10;
//     float thread_f = 1.0/(float)(1<<6);




//     //计算float和int矩阵乘法得到结果矩阵
//     xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);


//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
//     //计算修复误差后的量化矩阵乘法误差
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     std::cout<<"无修补"<<"\n";
//     printMatrix(matrixR,M,K);
//     std::cout<<"\n\n";



//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
//     //计算修复误差后的量化矩阵乘法误差
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     std::cout<<"稠密化修补"<<"\n";
//     printMatrix(matrixR,M,K);
//     std::cout<<"\n\n";

    

//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
//     //计算修复误差后的量化矩阵乘法误差
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     std::cout<<"稀疏化修补"<<"\n";
//     printMatrix(matrixR,M,K);
//     std::cout<<"\n\n";

//     // printMatrix(matrixAp,M,K);
//     // std::cout<<"\n\n";
//     // std::cout<<"稀疏A矩阵"<<"\n";
//     // printMatrix(matrixA,M,K);
//     // std::cout<<"\n\n";
//     return 0;
// }


int QRtest(){
    int rows=10,cols=10;

    float matrixA[10000],matrixQ_REF[10000],matrixR_ref[10000],matrix_resident[10000],matrixA1[10000];

    generate_matrix<float>(matrixA,rows,cols);

    xgeqrf_Householder_square<float>(matrixA,matrixQ_REF,matrixR_ref, rows, cols,'N');
    xgemm<float>(matrixQ_REF,matrixR_ref,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   

    std::cout<<"标准乘法相对误差"<<"\n";
    printMatrix_h(matrix_resident,rows,cols);
    std::cout<<"\n\n";


    xgeqrf_Householder_square<float>(matrixA,matrixQ_REF,matrixR_ref, rows, cols,'c');
    xgemm<float>(matrixQ_REF,matrixR_ref,matrixA1,rows,rows,rows,cols);
    get_error<float>(matrixA,matrixA1,matrix_resident,rows,cols);   


    std::cout<<"稠密量化乘法相对误差"<<"\n";
    printMatrix_h(matrix_resident,rows,cols);
    std::cout<<"\n\n";

}

int main(){
    QRtest();
}