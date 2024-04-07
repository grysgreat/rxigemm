
#include <stdio.h>
#include <omp.h>
#include "math.h"


#define SPARSITY 0.35

const int max_omp_thread = omp_get_max_threads();


template <typename T>
void xcopy(T matrix1[],T matrix2[], int rows, int cols ) {
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
			matrix2[i*cols+j] = matrix1[i*cols+j];
        }
    }
}


template <typename T>
void xtrans(T matrix[],T result[] , int rows, int cols) {


    T* tmp =(T *)malloc(sizeof(T) * rows*cols);
    #pragma omp parallel for num_threads(max_omp_thread)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {

            int originalIndex = i * cols + j;

            int transposedIndex = j * rows + i;
            tmp[transposedIndex] = matrix[originalIndex];
        }
    }
    xcopy<T>(tmp,result,cols,rows);
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
T get_max(T* matrix,int rows,int cols){
    T maxM=0;

    #pragma omp parallel for reduction(max:maxM)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){

            maxM = std::max(maxM,std::abs(matrix[i*cols+j]));
        }
    }
    return maxM;
}

template <typename T>
T get_avg(T* matrix,int rows,int cols){
    T sum=0;
    #pragma omp parallel for reduction(+:sum)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sum = ((sum)+std::abs(matrix[i*cols+j]));
        }
    }
    T avg = sum/(T)(rows*cols);
    return avg;
}


template <typename T>
void get_avg_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            T sum=0;
            for(int j=0;j<cols;j++){
                sum += std::abs(matrix[i*cols+j]);
            }
            vec[i] = sum/cols;
        }
    }
    if(type == 'c'){
        #pragma omp parallel for num_threads(max_omp_thread)        
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
void get_max_vec(T* matrix,T* vec, int rows,int cols,char type){
    
    if(type == 'r'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            T maxM=0;
            for(int j=0;j<cols;j++){
                maxM = std::max(std::abs(maxM),std::abs(matrix[i*cols+j]));
            }
            vec[i] = maxM;
        }
    }
    if(type == 'c'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int j=0;j<cols;j++){
            T maxM=0;
            for(int i=0;i<rows;i++){
                maxM = std::max(std::abs(maxM),std::abs(matrix[i*cols+j]));
            }
            vec[j] = maxM;
        }
    }    
}


template <typename T>
void get_lambda_vec(T* lambda_vec,T* max_vec,int digit, int len){


    int max_int = (1<<(digit-1)) - 1;
    // #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<len;i++){
        lambda_vec[i]=(T)max_int/max_vec[i];
    }
}

template <typename T>
void get_Res_matrix(T matrix_in[],T matrix_cmp[],T matrixR[],int rows,int cols){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrixR[i*cols+j] = (matrix_in[i*cols+j] - matrix_cmp[i*cols+j]);
        }
    }
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
}


template <typename Ti,typename To>
void quantitize_vec(Ti* matrix_in,To* matrix_out, float* lambda_vec,int rows,int cols,char type,char rc){

    if(type == 'q'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (matrix_in[i*cols+j]*lambda_vec[i]);
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<cols;i++){
                for(int j=0;j<rows;j++){
                    matrix_out[i+j*cols] = (matrix_in[i+j*cols]*lambda_vec[i]);
                }
            }        
        }
    }
    else if(type == 'd'){
        if(rc == 'r'){
            #pragma omp parallel for num_threads(max_omp_thread)
            for(int i=0;i<rows;i++){
                for(int j=0;j<cols;j++){
                    matrix_out[i*cols+j] = (To)matrix_in[i*cols+j]/lambda_vec[i];
                }
            }
        }
        else if(rc == 'c'){
            #pragma omp parallel for num_threads(max_omp_thread)
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
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            matrix_out[i*cols+j] = (To)(matrix_in[i*cols+j]/(lambda_r[i]*lambda_c[j]));
        }
    }
}


template <typename T>
void reduce_Matrix(T* matrix,T* vec,int rows,int cols,T Mlam_k,char type){

    if(type == 'r'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int i=0;i<rows;i++){
            T judge = Mlam_k;
            for(int j=0;j<cols;j++){
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix[i*cols+j]=0;
                }
            }
        }        
    }
    if(type == 'c'){
        #pragma omp parallel for num_threads(max_omp_thread)
        for(int j=0;j<cols;j++){
            for(int i=0;i<rows;i++){
                T judge = Mlam_k;
                if(std::abs(matrix[i*cols+j])<judge) {
                    matrix[i*cols+j]=0;
                }
            }
        }        
    }
}



// Function to convert a dense matrix to CSR format
template <typename T>
void denseToCSR(T* denseMatrix,T *values,long long int *colIndex,long long int *rowPtr, int numRows, int numCols) {
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



template <typename T>
T get_nnz_avg(T* matrix,int rows,int cols,int nnz){
    double sum=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            sum = ((sum)+std::abs(matrix[i*cols+j]));
        }
    }
    T avg = sum/(double)(nnz);
    return avg;
}


// Function to perform matrix-vector multiplication (CSR format) spmm 
template <typename T>
void sspmm(T *values,int *colIndex,int *rowPtr, T* B, T* C,int rowsA,int colsA,int rowsB, int colsB) {
    #pragma omp parallel for num_threads(max_omp_thread)
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
void xgemm(const T A[], const T B[], T C[], int rowsA, int colsA, int rowsB, int colsB) {


    T* tmp =(T *)malloc(sizeof(T) * rowsA*colsB);

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            tmp[i * colsB + j] = 0;
            for (int k = 0; k < colsA; ++k) {
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


void ilag2i8(int* matrix,int8_t* matrix_out,int len){
    for(int i=0;i<len;i++){
        matrix_out[i] = (int8_t)matrix[i];
    }
}

void ilag2s(int* matrix_A, float *A ,int lenth){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<lenth;i++){
        A[i]=matrix_A[i];
    }
}

void slag2i(float* matrix_A, int *A ,int lenth){
    #pragma omp parallel for num_threads(max_omp_thread)
    for(int i=0;i<lenth;i++){
        A[i]=(int)matrix_A[i];
    }
}
