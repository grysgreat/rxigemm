
#include <array>
#include <vector>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <unordered_map>
#include "aligned.h"
#include <omp.h> 
#include <mkl.h>

#include <Eigen/Dense>

using spmm_type = float;


void printMatrix_eigen(Eigen::MatrixXi matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%d   \t", matrix(i,j));
        }
        std::cout << std::endl;
    }
}

void eigen_copy(int *A ,Eigen::MatrixXi &matrix_A, int rowA,int colA){

    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            matrix_A(i,j) = A[i*colA+j];
            //printf("%d\t", matrix_A(i,j));
        }
    }
}
void eigen_copy(Eigen::MatrixXi matrix_A, int *A ,int rowA,int colA){

    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            A[i*colA+j]=matrix_A(i,j);
        }
    }
}
void eigen_copy(float *A ,Eigen::MatrixXf &matrix_A, int rowA,int colA){

    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            matrix_A(i,j) = A[i*colA+j];
            //printf("%d\t", matrix_A(i,j));
        }
    }
}
void eigen_copy(Eigen::MatrixXf matrix_A, float *A ,int rowA,int colA){

    for(int i=0;i<rowA;i++){
        for(int j=0;j<colA;j++){
            A[i*colA+j]=matrix_A(i,j);
        }
    }
}


void eigen_IGEMM(int *A ,int *B ,int *C, int rowA,int colA, int rowB,int colB){

    Eigen::MatrixXi matrix_A{rowA, colA};
    Eigen::MatrixXi matrix_B{rowB, colB};

    eigen_copy(A , matrix_A, rowA,colA);
    eigen_copy(B , matrix_B, rowB,colB);
    Eigen::MatrixXi matrix_C = matrix_A*matrix_B;

    // printMatrix_eigen(matrix_A, rowA,colA);
    // printMatrix_eigen( matrix_B, rowB,colB);

    eigen_copy(matrix_C , C, rowA,colB);


    return;
}


void eigen_SGEMM(float *A ,float *B ,float *C, int rowA,int colA, int rowB,int colB){

    Eigen::MatrixXf matrix_A{rowA, colA};
    Eigen::MatrixXf matrix_B{rowB, colB};

    eigen_copy(A , matrix_A, rowA,colA);
    eigen_copy(B , matrix_B, rowB,colB);
    Eigen::MatrixXf matrix_C = matrix_A*matrix_B;

    // printMatrix_eigen(matrix_A, rowA,colA);
    // printMatrix_eigen( matrix_B, rowB,colB);

    eigen_copy(matrix_C , C, rowA,colB);


    return;
}



void mkl_SPMM(MKL_INT *colIndexA, MKL_INT* rowPtrA, int* valuesA, int* B, int* C, MKL_INT rowsA, MKL_INT colsA ,MKL_INT rowsB ,MKL_INT colsB,int nnz) {
    char transa = 'N';
    float alpha = 1.0;
    float beta = 0.0;
    char matdescra[6];
    matdescra[0] = 'G'; // general
    matdescra[3] = 'C'; // zero-based indexing

    float *valueA_f = (float *)malloc(sizeof(float) * nnz);
    float *matrixB = (float *)malloc(sizeof(float) * rowsB*colsB);
    float *matrixC = (float *)malloc(sizeof(float) * rowsA*colsB);

    ilag2s(valuesA, valueA_f ,nnz);
    ilag2s(B, matrixB , rowsB*colsB);

    mkl_scsrmm(&transa, &rowsA, &colsB, &colsA, &alpha, matdescra, valueA_f, colIndexA, rowPtrA, &rowPtrA[1], matrixB, &colsB, &beta, matrixC, &colsB);

    slag2i(matrixC,C,rowsA*colsB);
 
    return ;
}



