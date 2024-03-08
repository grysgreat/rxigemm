// #include "math.h"
// #include <assert.h>
// #include <stdio.h>
// #include <iostream>
// #include <stdlib.h>
// #include <string.h>
// #include <vector>
// #include <random>
// #include <algorithm>
// #include <cmath>
// #include <random>
// //nvcc -arch=sm_75 -std=c++17 -Xcompiler -fopenmp rc_matrix.cu -o test -lcublas 
// #include <chrono>
// #include <iomanip>
// #include <cmath>
// #include <cstring>

// // ��ӡ����
// void printMatrix(float matrix[], int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             // ����������Ԫ�ص�����
//             int index = i * cols + j;
//             printf("%.4f   \t", matrix[index]);
//         }
//         std::cout << std::endl;
//     }
// }

// // �߾��ȴ�ӡ����
// void printMatrix_h(float matrix[], int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             // ����������Ԫ�ص�����
//             int index = i * cols + j;
//             printf("%.8f   \t", matrix[index]);
//         }
//         std::cout << std::endl;
//     }
// }

// // ��ӡint����
// void printMatrix_int(int matrix[], int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             // ����������Ԫ�ص�����
//             int index = i * cols + j;
//             printf("%d\t\t", matrix[index]);
//         }
//         std::cout << std::endl;
//     }
// }


// //���󿽱�����
// template <typename T>
// void xcopy(T matrix1[],T matrix2[], int rows, int cols ) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
// 			matrix2[i*cols+j] = matrix1[i*cols+j];
//         }
//     }
// }

// // �������ת�ú���
// template <typename T>
// void xtrans(T matrix[],T result[] , int rows, int cols) {

// 	T tmp[rows*cols];
//     // ִ�о���ת��   
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             // ����ԭ������Ԫ�ص�����
//             int originalIndex = i * cols + j;
//             // ����ת�ú������Ԫ�ص�����
//             int transposedIndex = j * rows + i;
//             // ����ת��
//             tmp[transposedIndex] = matrix[originalIndex];
//         }
//     }
//     xcopy<T>(tmp,result,cols,rows);
// }




// template <typename Ti,typename To>
// void quantitize(Ti* matrix_in,To* matrix_out,int rows,int cols,float lambda,char type){

//     if(type == 'q'){
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++){
//                 matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda);
//             }
//         }
//     }
//     else if(type == 'd'){
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++){
//                 matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda;
//             }
//         }        
//     }
// }


// template <typename Ti,typename To>
// void quantitize_vec(Ti* matrix_in,To* matrix_out, float* lambda_vec,int rows,int cols,char type){

//     if(type == 'q'){
//         if(type == 'r'){
//             for(int i=0;i<rows;i++){
//                 for(int j=0;j<cols;j++){
//                     matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda_vec[i]);
//                 }
//             }
//         }
//         else if(type == 'c'){
//             for(int i=0;i<cols;i++){
//                 for(int j=0;j<rows;j++){
//                     matrix_out[i+j*cols] = (matrix_in[i+j*cols]*lambda_vec[i]);
//                 }
//             }        
//         }
//     }
//     else if(type == 'd'){
//         if(type == 'r'){
//             for(int i=0;i<rows;i++){
//                 for(int j=0;j<cols;j++){
//                     matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda_vec[i];
//                 }
//             }
//         }
//         else if(type == 'c'){
//             for(int i=0;i<cols;i++){
//                 for(int j=0;j<rows;j++){
//                     matrix_out[i+j*cols] = (To)(matrix_in[i+j*cols]/lambda_vec[i]);
//                 }
//             }        
//         }    
//     }
// }

// template <typename Ti,typename To>
// void dequantitize_matrix(Ti* matrix_in,To* matrix_out, float* lambda_r,  float* lambda_c,int rows,int cols){
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             matrix_out[i*cols+j] = (To)(matrix_in[i*cols+j]/(lambda_r[i]*lambda_c[j]));
//         }
//     }
// }

// // template <typename T>
// // void get_max_vec(T* matrix,T* max_vec, int rows,int cols,char type){
// //     if(type == 'r'){
// //         for(int i=0;i<rows;i++){
// //             for(int j=0;j<cols;j++){
// //                 max_vec[i] = max(std::abs(max_vec[i]),std::abs(matrix[i*cols+j]));
// //             }
// //         }
// //     }
// //     else if(type == 'c'){
// //         for(int i=0;i<cols;i++){
// //             for(int j=0;j<rows;j++){
// //                 max_vec[i] = max(std::abs(max_vec[i]),std::abs(matrix[i+j*cols]));
// //             }
// //         }        
// //     }

// // }


// template <typename T>
// T get_max(T* matrix,int rows,int cols){
//     T maxM=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             maxM = max(std::abs(maxM),std::abs(matrix[i*cols+j]));
//         }
//     }
//     return maxM;
// }

// template <typename T>
// T get_avg(T* matrix,int rows,int cols){
//     T sum=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             sum = ((sum)+std::abs(matrix[i*cols+j]));
//         }
//     }
//     T avg = sum/(T)(rows*cols);
//     return avg;
// }


// template <typename T>
// T get_min(T* matrix,int rows,int cols){
//     T minM=1000000;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             if(std::abs(matrix[i*cols+j])>1e-20)
//                 minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
//         }
//     }
//     return minM;
// }


// template <typename T>
// void get_min_vec(T* matrix,T* vec, int rows,int cols,char type){
    
//     if(type == 'r'){
//         for(int i=0;i<rows;i++){
//             T minM=2553555;
//             for(int j=0;j<cols;j++){
//                 minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
//             }
//             vec[i] = minM;
//         }
//     }
//     if(type == 'c'){
//         for(int j=0;j<cols;j++){
//             T minM=2553555;
//             for(int i=0;i<rows;i++){
//                 minM = min(std::abs(minM),std::abs(matrix[i*cols+j]));
//             }
//             vec[j] = minM;
//         }
//     }    


// }

// template <typename T>
// void get_max_vec(T* matrix,T* vec, int rows,int cols,char type){
    
//     if(type == 'r'){
//         for(int i=0;i<rows;i++){
//             T minM=0;
//             for(int j=0;j<cols;j++){
//                 minM = max(std::abs(minM),std::abs(matrix[i*cols+j]));
//             }
//             vec[i] = minM;
//         }
//     }
//     if(type == 'c'){
//         for(int j=0;j<cols;j++){
//             T minM=0;
//             for(int i=0;i<rows;i++){
//                 minM = max(std::abs(minM),std::abs(matrix[i*cols+j]));
//             }
//             vec[j] = minM;
//         }
//     }    


// }


// template <typename T>
// void get_avg_vec(T* matrix,T* vec, int rows,int cols,char type){
    
//     if(type == 'r'){
//         for(int i=0;i<rows;i++){
//             T sum=0;
//             for(int j=0;j<cols;j++){
//                 sum += std::abs(matrix[i*cols+j]);
//             }
//             vec[i] = sum/cols;
//         }
//     }
//     if(type == 'c'){
//         for(int j=0;j<cols;j++){
//             T sum=0;
//             for(int i=0;i<rows;i++){
//                 sum += std::abs(matrix[i*cols+j]);
//             }
//             vec[j] = sum/rows;
//         }
//     }    


// }

// template <typename T>
// void xmadd(T* matrixA,T* matrixB,T* matrixC,int rows,int cols){
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             matrixC[i*cols+j] = matrixA[i*cols+j] + matrixB[i*cols+j];
//         }
//     }
// }

// template <typename T>
// // �������˾�����
// void xgemm(const T A[], const T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {

//     // ȷ�����Խ��о���˷��ĳߴ�
//     if (colsA != rowsB) {
//         std::cout << "�޷����о���˷����ߴ粻ƥ�䡣" << std::endl;
//         return;
//     }
//     T* tmp =(T *)malloc(sizeof(T) * rowsA*colsB);

//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             // ��ʼ����������е�Ԫ��Ϊ0
//             tmp[i * colsB + j] = 0;
//             for (int k = 0; k < colsA; ++k) {
//                 // ����˷����ۼӲ���
//                 tmp[i * colsB + j] += A[i * colsA + k] * B[k * colsB + j];
//             }
//         }
//     }
    
//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             C[i * colsB + j]  = tmp[i * colsB + j];
// 		}
// 	}
    
// }

// template <typename T>
// int get_nnz(T *denseMatrix, int numRows,int numCols){
//     int nnz = 0; // Number of non-zero elements
//     for (int i = 0; i < numRows; ++i) {
//         for (int j = 0; j < numCols; ++j) {
//             if (denseMatrix[i*numCols+j] != 0) {
//                 ++nnz;
//             }
//         }
//     }
//     return nnz;
// }


// // Function to convert a dense matrix to CSR format
// template <typename T>
// void denseToCSR(T* denseMatrix,T *values,int *colIndex,int *rowPtr, int numRows, int numCols) {
//     int nnz = 0;
//     for (int i = 0; i < numRows; ++i) {
//         rowPtr[i] = nnz;
//         for (int j = 0; j < numCols; ++j) {
//             if (denseMatrix[i*numCols+j] != 0) {
//                 values[nnz] = denseMatrix[i*numCols+j];
//                 colIndex[nnz] = j;
//                 ++nnz;
//             }
//         }
//     }
//     rowPtr[numRows] = nnz;

// }


// // Function to perform matrix-vector multiplication (CSR format) spmm 
// template <typename T>
// void sspmm(T *values,int *colIndex,int *rowPtr, T* B, T* C,int rowsA,int colsA,int rowsB, int colsB) {

//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             C[i*colsB+j] = 0.0;
//             for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
//                 C[i*colsB+j] += values[k] * B[colIndex[k]*colsB+j];
//             }
//         }
//     }
// }

// template <typename T>
// void sspmm_nt(T *values,int *colIndex,int *rowPtr, T* B, T* C,int rowsA,int colsA,int rowsB, int colsB) {

//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             C[i+j*rowsA] = 0.0;
//             for (int k = rowPtr[i]; k < rowPtr[i + 1]; ++k) {
                
//                 C[i+j*rowsA] += values[k] * B[colIndex[k]+j*colsA];
//             }
//         }
//     }
// }




// template <typename T>
// void get_R(T matrix_in[],T matrix_cmp[],T matrixR[],int rows,int cols){
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             matrixR[i*cols+j] = (matrix_in[i*cols+j] - matrix_cmp[i*cols+j]);
//         }
//     }
// }

// template <typename T>
// void get_error(T matrix_ref[],T matrix_cmp[],T matrixR[],int rows,int cols){
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             matrixR[i*cols+j] = std::abs(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])/std::abs(matrix_ref[i*cols+j]);
//         }
//     }
// }

// template <typename T>
// T get_Ferror(T matrix_ref[],T matrix_cmp[],int rows,int cols){

//     T sumR=0,sum=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             sumR+=(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j])*(matrix_ref[i*cols+j] - matrix_cmp[i*cols+j]);
//             sum+=(matrix_ref[i*cols+j])*(matrix_ref[i*cols+j]);
//         }
//     }

//     T ans = sqrt(sumR)/sqrt(sum);
//     return ans;

// }


// template <typename T>
// void reduce_Residual(T* matrix,T* matrixR,int rows,int cols,int N,T max,int threshold){
//     int max_exp;
//     std::frexp(max, &max_exp);    
//     int judge = max_exp - N + threshold;

//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             int aij_exp;
//             std::frexp(matrix[i*cols+j], &aij_exp); 
//             if(aij_exp>judge){
//                 matrixR[i*cols+j] = 0;
//             }
//         }
//     }        
// }



// template <typename T>
// void reduce_Matrix(T* matrix,T* vec,int rows,int cols,T Mlam_k,char type){
//     float cnt=0;
//     if(type == 'r'){
//         for(int i=0;i<rows;i++){
//             T judge = Mlam_k*vec[i];
//             //printf("judge = %f\n",judge);
//             for(int j=0;j<cols;j++){
//                 if(matrix[i*cols+j]<judge) {
//                     matrix[i*cols+j]=0;
//                     cnt++;
//                 }
//             }
//         }        
//     }
//     if(type == 'c'){
//         for(int j=0;j<cols;j++){
//             for(int i=0;i<rows;i++){
//                 T judge = Mlam_k*vec[j];
//                 if(matrix[i*cols+j]<judge) {
//                     matrix[i*cols+j]=0;
//                     cnt++;
//                 }
//             }
//         }        
//     }
//     float spasity = 1-(cnt/(float)(rows*cols));
//     //printf("spasity = %f\n",spasity);
// }


// template <typename T>
// void generate_matrix(T* matrix,int rows,int cols,char type ){
//     // ����һ�����������
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     // ����һ�����ȷֲ�����Χ��[0, 1)
//     std::uniform_real_distribution<float> dis(0.0, 1.0);
//     //std::normal_real_distribution<float> dis(0.0, 1.0);

//     if(type == 'u'){
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++){
//                 matrix[i*cols+j] = dis(gen);
//                 if(i==j&&i!=rows-1) matrix[i*cols+j] = (matrix[i*cols+j]);
//                 else  matrix[i*cols+j]=(matrix[i*cols+j]);
//             }
//         }        
//     }
//     else{
//         for(int i=0;i<rows;i++){
//             for(int j=0;j<cols;j++){
//                 T U1 = rand() * 1.0f / RAND_MAX; // 0~1���ȷֲ�
//                 T U2 = rand() * 1.0f / RAND_MAX; // 0~1���ȷֲ�
//                 T Z = sqrt(-2 * log(U1))*cos(2 * M_PI * U2);// ��ֵΪ0������Ϊ1����̬�ֲ�
//                 T Y = 1 +  Z; // ��ֵΪ1������Ϊ4����̬�ֲ�
//                 matrix[i*cols+j] = Y;
//             }
//         }        
//     }
// }
// template <typename T>
// void splitMatrix(T A[], T B[],  int rows, int cols, T judge){
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             if(abs(A[i*cols+j])<judge){
//                 B[i*cols+j] = A[i*cols+j];
//                 A[i*cols+j] = 0;
//             } else {
//                 B[i*cols+j] = 0;
//             }
//         }
//     }        
// }

// template <typename T>
// T get_sparsity(T A[],int rows, int cols){
//     int cnt=0;
//     for(int i=0;i<rows;i++){
//         for(int j=0;j<cols;j++){
//             if(A[i*cols+j]!=0){
//                 cnt++;
//             } 
//         }
//     }     
//     T sp = (T)cnt / (T)(rows*cols);
//     return sp;
// }


// // �������˾�����---- �в
// template <typename T,int digit>
// void xigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

//     // ȷ�����Խ��о���˷��ĳߴ�
//     if (colsA != rowsB) {
//         std::cout << "�޷����о���˷����ߴ粻ƥ�䡣" << std::endl;
//         return;
//     }
    
//     int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
//     int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
//     int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

//     T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
//     T *C_rows = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_cols = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);

//     xcopy<T>(A,A_copy, rowsA, colsA);
//     xcopy<T>(B,B_copy, rowsB, colsB);


//     const int max_int = 1<<(digit-1) - 1;
//     T max_mA =get_max<T>(A,rowsA, colsA);
//     T max_mB =get_max<T>(B,rowsB, colsB);
//     T lambdaA = (T)max_int/max_mA;
//     T lambdaB = (T)max_int/max_mB;

//     //��A,B�������ֱ������
//     quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
//     quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

//     //����float��int����˷��õ��������
//     xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


//     //�Խ������C_INT���з������õ�C'
//     T lambdaC = lambdaA*lambdaB;
//     quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

//     if(type == 'c' || type == 'x'){
        
//         //���������int���������õ�A',B'
//         quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
//         quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

//         //����full size�Ĳв����
//         get_R<T>(A,A_p,A_p,rowsA,colsA);
//         get_R<T>(B,B_p,B_p,rowsB,colsB);

//         //�Բв�����������
//         T max_mAR =get_max<T>(A_p,rowsA,colsA);
//         T max_mBR =get_max<T>(B_p,rowsB,colsB);
//         T lambdaAR = (T)max_int/max_mAR;
//         T lambdaBR = (T)max_int/max_mBR;
//         //��A,B�в�������ֱ������
//         quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
//         quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');


//         T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
//         if(type == 'x'){
//             //ϡ�軯����޲�
//             T avg = get_avg<T>(B,rowsB,colsB);
//             // std::cout<<"avg = "<<avg<<"\n";

//             threadhoud/=avg;
//             T ml_kA = threadhoud*lambdaB/colsA;
//             T ml_kB = threadhoud*lambdaA/colsA;    

//             get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
//             get_avg_vec<T>(C_buffer,C_cols,colsA,colsB,'c');
            
//             reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
//             reduce_Matrix(B_copy,C_cols,rowsB,rowsB, ml_kB ,'c');
//             //����AB��������
//             T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
//             T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
//             lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
//             lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
//             quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
//             quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');
//             //��int����޸����������õ�����޸�����float
//             T lambdaCR1 = lambdaAnew*lambdaBR;
//             T lambdaCR2 = lambdaAR*lambdaBnew;
//             int nnzA = get_nnz<int>(A_int, rowsA, colsA);
//             int nnzB = get_nnz<int>(B_int, rowsA, colsA);

//             int* valuesA = new int[nnzA];
//             int* colIndexA = new int[nnzA];
//             int* rowPtrA = new int[rowsA + 1];
//             denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


//             sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);


//             //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             if(lambdaCR1!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


//             xtrans<int>(B_int,B_int,rowsB,colsB);
//             xtrans<int>(AR_int,AR_int,rowsA, colsA);
//             int* valuesB = new int[nnzB];
//             int* colIndexB = new int[nnzB];
//             int* rowPtrB = new int[colsB + 1];
//             denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
//             sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
//             xtrans<int>(C_int,C_int,colsB,rowsA);

//             //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����

//             if(lambdaCR2!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


//         } else {
//             //��int����޸����������õ�����޸�����float
//             T lambdaCR1 = lambdaAnew*lambdaBR;
//             T lambdaCR2 = lambdaAR*lambdaBnew;

//             xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
//             xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

//         }
//     }

//     xcopy<T>(C_buffer,C, rowsA, colsB);

// }



// template <typename T,int digit>
// void rcxigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

//     // ȷ�����Խ��о���˷��ĳߴ�
//     if (colsA != rowsB) {
//         std::cout << "�޷����о���˷����ߴ粻ƥ�䡣" << std::endl;
//         return;
//     }
    
//     int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
//     int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
//     int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

//     T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
//     T *C_rows = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_cols = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);

//     xcopy<T>(A,A_copy, rowsA, colsA);
//     xcopy<T>(B,B_copy, rowsB, colsB);


//     const int max_int = 1<<(digit-1) - 1;
//     T max_mA =get_max<T>(A,rowsA, colsA);
//     T max_mB =get_max<T>(B,rowsB, colsB);
//     T lambdaA = (T)max_int/max_mA;
//     T lambdaB = (T)max_int/max_mB;

//     //��A,B�������ֱ������
//     quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
//     quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

//     //����float��int����˷��õ��������
//     xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


//     //�Խ������C_INT���з������õ�C'
//     T lambdaC = lambdaA*lambdaB;
//     quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

// }


// // ����ӡ��
// template <typename T,int digit>
// void xigemm_with_print(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

//     // ȷ�����Խ��о���˷��ĳߴ�
//     if (colsA != rowsB) {
//         std::cout << "�޷����о���˷����ߴ粻ƥ�䡣" << std::endl;
//         return;
//     }
    
//     int *A_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *B_int = (int *)malloc(sizeof(int) * rowsB*colsB);
//     int *C_int = (int *)malloc(sizeof(int) * rowsA*colsB);
//     int *AR_int = (int *)malloc(sizeof(int) * rowsA*colsA);
//     int *BR_int = (int *)malloc(sizeof(int) * rowsB*colsB);

//     T *C_copy = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_buffer = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_p = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_p = (T *)malloc(sizeof(T) * rowsB * colsB);
//     T *C_rows = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *C_cols = (T *)malloc(sizeof(T) * rowsA * colsB);
//     T *A_copy = (T *)malloc(sizeof(T) * rowsA * colsA);
//     T *B_copy = (T *)malloc(sizeof(T) * rowsB * colsB);


//     xcopy<T>(A,A_copy, rowsA, colsA);
//     xcopy<T>(B,B_copy, rowsB, colsB);


//     const int max_int = 1<<(digit-1) - 1;
//     T max_mA =get_max<T>(A,rowsA, colsA);
//     T max_mB =get_max<T>(B,rowsB, colsB);
//     T lambdaA = (T)max_int/max_mA;
//     T lambdaB = (T)max_int/max_mB;

//     //��A,B�������ֱ������
//     quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
//     quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');


//     std::cout<<"A�������ֵ = "<<max_mA<<"\n";
//     std::cout<<"lambda A = "<<lambdaA<<"\n";
//     std::cout<<"����A����"<<"\n";
//     printMatrix_int(A_int,rowsA,rowsA);
//     std::cout<<"\n\n";


//     std::cout<<"B�������ֵ = "<<max_mB<<"\n";
//     std::cout<<"lambda A = "<<lambdaB<<"\n";
//     std::cout<<"����B����"<<"\n";
//     printMatrix_int(B_int,rowsA,rowsA);
//     std::cout<<"\n\n";

//     //����float��int����˷��õ��������
//     xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);


//     //�Խ������C_INT���з������õ�C'
//     T lambdaC = lambdaA*lambdaB;
//     quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');


//     std::cout<<"ֱ������C����"<<"\n";
//     printMatrix(C_buffer,rowsA,rowsA);
//     std::cout<<"\n\n";

//     if(type == 'c' || type == 'x'){
        
//         //���������int���������õ�A',B'
//         quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
//         quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

//         std::cout<<"������A����"<<"\n";
//         printMatrix(A_p,rowsA,rowsA);
//         std::cout<<"\n\n";

//         std::cout<<"������B����"<<"\n";
//         printMatrix(B_p,rowsA,rowsA);
//         std::cout<<"\n\n";

//         //����full size�Ĳв����
//         get_R<T>(A,A_p,A_p,rowsA,colsA);
//         get_R<T>(B,B_p,B_p,rowsB,colsB);

//         std::cout<<"�в�A����"<<"\n";
//         printMatrix(A_p,rowsA,rowsA);
//         std::cout<<"\n\n";

//         std::cout<<"�в�B����"<<"\n";
//         printMatrix(B_p,rowsA,rowsA);
//         std::cout<<"\n\n";


//         //�Բв�����������
//         T max_mAR =get_max<T>(A_p,rowsA,colsA);
//         T max_mBR =get_max<T>(B_p,rowsB,colsB);
//         T lambdaAR = (T)max_int/max_mAR;
//         T lambdaBR = (T)max_int/max_mBR;
//         //��A,B�в�������ֱ������
//         quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
//         quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');


//         T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
//         if(type == 'x'){
//             //ϡ�軯����޲�

//             T avg = get_avg<T>(B,rowsB,colsB);
//             // std::cout<<"avg = "<<avg<<"\n";


//             threadhoud/=avg;
//             T ml_kA = threadhoud*lambdaB/colsA;
//             T ml_kB = threadhoud*lambdaA/colsA;    




//             get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
//             get_avg_vec<T>(C_buffer,C_cols,rowsA,colsB,'c');
            
//             std::cout<<"C������ƽ��ֵ"<<"\n";
//             printMatrix(C_rows,1,rowsA);
//             std::cout<<"\n\n";            

//             std::cout<<"C������ƽ��ֵ"<<"\n";
//             printMatrix(C_cols,1,rowsA);
//             std::cout<<"\n\n";      

//             reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
//             reduce_Matrix(B_copy,C_cols,rowsB,rowsB, ml_kB ,'c');

//             std::cout<<"reduceA ����"<<"\n";
//             printMatrix(A_copy,rowsA,rowsA);
//             std::cout<<"\n\n";            

//             std::cout<<"reduceB ����"<<"\n";
//             printMatrix(B_copy,rowsA,rowsA);
//             std::cout<<"\n\n";                  

//             // float sparsity = get_sparsity(B_copy,rowsB,rowsB);
//             // std::cout<<"sparsity = "<<sparsity<<"\n";
//             // sparsity = get_sparsity(A_copy,rowsA,colsA);
//             // std::cout<<"sparsity = "<<sparsity<<"\n";

//         // std::cout<<"otho"<<"\n";
//         // printMatrix_h(A_copy,rowsA,colsA);
//         // std::cout<<"\n\n";

//             //����AB��������
//             T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
//             T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
//             lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
//             lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
//             quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
//             quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');


//             std::cout<<"lambda A = "<<lambdaAnew<<"\n";
//             std::cout<<"����ϡ��A'����"<<"\n";
//             printMatrix_int(A_int,rowsA,rowsA);
//             std::cout<<"\n\n";


//             std::cout<<"lambda B = "<<lambdaBnew<<"\n";
//             std::cout<<"����ϡ��B'����"<<"\n";
//             printMatrix_int(B_int,rowsA,rowsA);
//             std::cout<<"\n\n";
            
//             //��int����޸����������õ�����޸�����float
//             T lambdaCR1 = lambdaAnew*lambdaBR;
//             T lambdaCR2 = lambdaAR*lambdaBnew;
//             int nnzA = get_nnz<int>(A_int, rowsA, colsA);
//             int nnzB = get_nnz<int>(B_int, rowsA, colsA);

//             int* valuesA = new int[nnzA];
//             int* colIndexA = new int[nnzA];
//             int* rowPtrA = new int[rowsA + 1];
//             denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


//             sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);


//             //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             if(lambdaCR1!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


//             xtrans<int>(B_int,B_int,rowsB,colsB);
//             xtrans<int>(AR_int,AR_int,rowsA, colsA);
//             int* valuesB = new int[nnzB];
//             int* colIndexB = new int[nnzB];
//             int* rowPtrB = new int[colsB + 1];
//             denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
//             sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
//             xtrans<int>(C_int,C_int,colsB,rowsA);

//             //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             if(lambdaCR2!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


//         } else {
//             //��int����޸����������õ�����޸�����float
//             T lambdaCR1 = lambdaAnew*lambdaBR;
//             T lambdaCR2 = lambdaAR*lambdaBnew;

//             xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
//             xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
//             quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
//             //ʹ���޸����󲹳����
//             xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

//         }






//     }

//     xcopy<T>(C_buffer,C, rowsA, colsB);

// }




// // �����������ڻ�
// template <typename T>
// T dotProduct_v(T v1[], T v2[], int size) {
//     T result = 0.0;
//     for (int i = 0; i < size; ++i) {
//         result += v1[i] * v2[i];
//     }
//     return result;
// }

// // Householder�任��QR�ֽ�,���ֻ֧�ַ��󣬰��ձ�׼���� I-2UUT/UTU ���ɵ�Q����Ӧ���Ƿ���
// template <typename T>
// void xgeqrf_Householder_square(T inputMatrix[], T orthogonalMatrix[], T upperTriangularMatrix[], int rows, int cols,char type) {
	
// 	T orthogonalMatrix_tmp[rows*rows],inputMatrix_tmp[rows*cols];
// 	//����һ��������� 
// 	xcopy<T>(inputMatrix,inputMatrix_tmp,rows,cols);


// 	//��ʼ��Ϊ��λ���� 
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < rows; ++j) {
//             if(i==j) orthogonalMatrix[i*rows+j]=1;
//             else orthogonalMatrix[i*rows+j]=0;
//         }
//     }	
	
//     for (int k = 0; k < cols-1; ++k) {
//         // �����������ĵ�k�е�v����
//         T v[rows];
//         for (int i = 0; i < rows; ++i) {
//             v[i] = inputMatrix_tmp[i * cols + k];
//         }
//         for (int i = 0; i < k; ++i) {
//             v[i] = 0;
//         }
        
        
//         // ����Householder����
//         T normV = std::sqrt(dotProduct_v<T>(v, v, rows));
// 		v[k] -= normV;

//         // ����Householder����H
//         T vTv = dotProduct_v<T>(v, v, rows);
//         for (int i = 0; i < rows; ++i) {
//             for (int j = 0; j < rows; ++j) {
//                 orthogonalMatrix_tmp[i * rows + j] = -2 * (v[i] * v[j]) / vTv;
//             }
//             orthogonalMatrix_tmp[i * rows + i] += 1.0;
//         }

//         if(type == 'N'){
//             //���������� orthogonalMatrix_tmp Ӧ�õ� inputMatrix_tmp �У���������������˻���� 
//             xgemm<T>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
//             xgemm<T>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		
//         }
//         else if(type =='e') {
//             const int digit = 8;
//             xigemm_e<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols);
//             xigemm_e<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows);		
//         }
//         else {
//             const int digit = 8;
//             float thread_f = 1.0/(float)(1<<6);
//             //���������� orthogonalMatrix_tmp Ӧ�õ� inputMatrix_tmp �У���������������˻���� 
//             xigemm<T,digit>(orthogonalMatrix_tmp,inputMatrix_tmp,inputMatrix_tmp,rows,rows,rows,cols,thread_f,type);
//             xigemm<T,digit>(orthogonalMatrix_tmp,orthogonalMatrix,orthogonalMatrix,rows,rows,rows,rows,thread_f,type);	
//         }
    


//     }
    
// 	//����ת�õ������Ǿ����QT 
//     xcopy<T>(inputMatrix_tmp,upperTriangularMatrix,rows,cols);		
// 	xtrans<T>(orthogonalMatrix,orthogonalMatrix,rows,rows);

// 	//��֤������Ϊ0 
// 	for(int i=1;i<cols;i++){
// 		for(int j=0;j<i;j++){
// 			upperTriangularMatrix[i*cols+j]=1e-8;
// 		}
// 	}

// }



// int test1(){
//     int N=800,M=800,K=200;


//     float *matrixA = (float *)malloc(sizeof(float) * M*K);
//     float *matrixB = (float *)malloc(sizeof(float) * N*K);
//     float *matrixC = (float *)malloc(sizeof(float) * M*N);
//     float *matrixCQ = (float *)malloc(sizeof(float) * M*N);
//     float *matrixR = (float *)malloc(sizeof(float) * M*N);
//     generate_matrix<float>(matrixA,M,K,'u');
//     generate_matrix<float>(matrixB,K,N,'u');


//     const int digit=12;
//     float thread_f = 0.0001;




//     //����float��int����˷��õ��������
//     xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);


//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'o');
//     //�����޸��������������˷����
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     // std::cout<<"���޲�"<<"\n";
//     // printMatrix(matrixR,M,K);
//     // std::cout<<"\n\n";
//     float R0 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
//     std::cout<<"���޲������� = "<<R0<<"\n";


//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'c');
//     //�����޸��������������˷����
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     // std::cout<<"���ܻ��޲�"<<"\n";
//     // printMatrix(matrixR,M,K);
//     // std::cout<<"\n\n";
//     float R1 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
//     std::cout<<"���ܻ��޲������� = "<<R1<<"\n";
    

//     xigemm<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N,thread_f,'x');
//     //�����޸��������������˷����
//     get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     // std::cout<<"ϡ�軯�޲�"<<"\n";
//     // printMatrix(matrixR,M,K);
//     // std::cout<<"\n\n";
//     float R2 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
//     std::cout<<"ϡ�軯�޲������� = "<<R2<<"\n";


//     // xigemm_e<float,digit>(matrixA,matrixB,matrixCQ,M,K,K,N);
//     // //�����޸��������������˷����
//     // get_error<float>(matrixC,matrixCQ,matrixR,M,N);   
//     // float R3 = get_Ferror<float>(matrixC,matrixCQ,M,N); 
//     // std::cout<<"ָ���з��޲������� = "<<R3<<"\n";




//     // printMatrix(matrixAp,M,K);
//     // std::cout<<"\n\n";
//     // std::cout<<"ϡ��A����"<<"\n";
//     // printMatrix(matrixA,M,K);
//     // std::cout<<"\n\n";
//     return 0;
// }




// // �������˾�����
// void xgemm_i8(const int8_t A[], const int8_t B[], int32_t C[], int rowsA, int colsA, int rowsB, int colsB) {

//     // ȷ�����Խ��о���˷��ĳߴ�
//     if (colsA != rowsB) {
//         std::cout << "�޷����о���˷����ߴ粻ƥ�䡣" << std::endl;
//         return;
//     }
//     int32_t* tmp =(int32_t *)malloc(sizeof(int32_t) * rowsA*colsB);
//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             // ��ʼ����������е�Ԫ��Ϊ0
//             tmp[i * colsB + j] = 0;
//             for (int k = 0; k < colsA; ++k) {
//                 // ����˷����ۼӲ���
//                 tmp[i * colsB + j] += static_cast<int32_t>(A[i * colsA + k]) * static_cast<int32_t>(B[k * colsB + j]);
//             }
//         }
//     }
    
//     for (int i = 0; i < rowsA; ++i) {
//         for (int j = 0; j < colsB; ++j) {
//             C[i * colsB + j]  = tmp[i * colsB + j];
// 		}
// 	}
    
// }


// int gemm_cpu_test(){

//     float *matrixA = (float *)malloc(sizeof(float) * M*K);
//     float *matrixB = (float *)malloc(sizeof(float) * K*N);

//     float *matrixC = (float *)malloc(sizeof(float) * K*N);

//     int32_t  *matrixC32 = (int32_t *)malloc(sizeof(int32_t) * M*N);
//     generate_matrix<float>(matrixA,M,K,'u');
//     generate_matrix<float>(matrixB,K,N,'u');


//     int8_t *matrixA8 = (int8_t *)malloc(sizeof(int8_t) * M*K);
//     int8_t *matrixB8 = (int8_t *)malloc(sizeof(int8_t) * K*N);    
//     const int max_int = 1<<(8-1) - 1;
//     float lambdaA = (float)max_int;
//     quantitize<float,int8_t>(matrixA,matrixA8, M, K,lambdaA,'q');
//     quantitize<float,int8_t>(matrixB,matrixB8, K, N,lambdaA,'q');

//     auto start = std::chrono::high_resolution_clock::now();
//     xgemm<float>(matrixA,matrixB,matrixC,M,K,K,N);
//     auto end = std::chrono::high_resolution_clock::now();
//     std::chrono::duration<double> diff = end - start;
//     int time  = diff.count()*1000*1000;
//     std::cout<<"size="<<M<<"//" << std::fixed << std::setprecision(6) << time << std::endl;

//      start = std::chrono::high_resolution_clock::now();
//     xgemm_i8(matrixA8,matrixB8,matrixC32,M,K,K,N);
//      end = std::chrono::high_resolution_clock::now();
//     diff = end - start;
//      time  = diff.count()*1000*1000;
//     std::cout<<"size="<<M<<"//" << std::fixed << std::setprecision(6) << time << std::endl;


// }










// int main(){
//     //ϡ��в�qr�ֽ����
//     //QRtest();

//     //��ͬ��������˲���
//     //test1();

//     //ϡ��в��Ӳ���quant��reduce GPU����
//     // int num1[12] = {128,256,512,1024,2048,4096,8192,16384};
//     // for(int i=0;i<8;i++){
//     //     M=num1[i];
//     //     N=num1[i];
//     //     K=num1[i];
//     //     cuda_quant_speedTest();
//     //     cuda_reduce_speedTest();
//     // }


//     //������gemm����
//     // int num1[12] = {128,256,512,1024,2048};
//     // for(int i=0;i<8;i++){
//     //     M=num1[i];
//     //     N=num1[i];
//     //     K=num1[i];
//     //     gemm_cpu_test();
//     // }

//     //��ӡ����
//     // print_test();

//     //ϡ�軯���Ȳ���
//     // error_th_N_test();

// }