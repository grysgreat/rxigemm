
#include "aligned.h"
#include "operator_matrix.hpp"
#include "blas_eigen.hpp"


template <typename T,int digit>
void xigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {

    // Ensure that matrix multiplication can be performed at a size
    if (colsA != rowsB) {
        std::cerr << "Matrix multiplication is not possible, the dimensions do not match." << std::endl;
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


    const int max_int = (1<<(digit-1)) - 1;
    T max_mA =get_max<T>(A,rowsA, colsA);
    T max_mB =get_max<T>(B,rowsB, colsB);
    T lambdaA = (T)max_int/max_mA;
    T lambdaB = (T)max_int/max_mB;

    

    //Direct quantization of A and B matrices
    quantitize<T,int>(A,A_int,rowsA, colsA,lambdaA,'q');
    quantitize<T,int>(B,B_int,rowsB, colsB,lambdaB,'q');

    //Compute the multiplication of float and int matrices to get the resulting matrix
    //xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    T lambdaC = lambdaA*lambdaB;
    quantitize<int,T>(C_int,C_buffer,rowsA,colsB,lambdaC,'d');

    if(type == 'c' || type == 'x'){
        
        //Dequantization of the quantized int matrix gives A',B'
        quantitize<int,T>(A_int,A_p,rowsA,colsA,lambdaA,'d');
        quantitize<int,T>(B_int,B_p,rowsB,colsB,lambdaB,'d');

        //Calculate the residuals matrix for full size
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);

        //The residual matrix is quantized
        T max_mAR =get_max<T>(A_p,rowsA,colsA);
        T max_mBR =get_max<T>(B_p,rowsB,colsB);


    
        T lambdaAR = (T)max_int/max_mAR;
        T lambdaBR = (T)max_int/max_mBR;
        //Direct quantization of A and B residual matrices
        quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
        quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');


        T lambdaAnew = lambdaA,lambdaBnew = lambdaB;
        if(type == 'x'){
            //Sparse error repair
            T avg = get_avg<T>(B,rowsB,colsB);
            // std::cout<<"avg = "<<avg<<"\n";

            threadhoud/=avg;
            T ml_kA = threadhoud*lambdaB/colsA;
            T ml_kB = threadhoud*lambdaA/colsA;    

            get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
            get_avg_vec<T>(C_buffer,C_cols,colsA,colsB,'c');
            
            reduce_Matrix(A_copy,C_rows,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,C_cols,rowsB,rowsB, ml_kB ,'c');
            //Quantify the new AB
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');
            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            int nnzA = get_nnz<int>(A_int, rowsA, colsA);
            int nnzB = get_nnz<int>(B_int, rowsA, colsA);

            int* valuesA = new  int[nnzA];
            long long int* colIndexA = new long long int[nnzA];
            long long int* rowPtrA = new long long int[rowsA + 1];
            denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);

            mkl_SPMM(colIndexA,rowPtrA,valuesA,BR_int,C_int,rowsA, colsA,rowsB, colsB,nnzA);
            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


            xtrans<int>(B_int,B_int,rowsB,colsB);
            xtrans<int>(AR_int,AR_int,rowsA, colsA);
            int* valuesB = new  int[nnzB];
            long long int* colIndexB = new long long int[nnzB];
            long long int* rowPtrB = new long long int[colsB + 1];
            denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
            mkl_SPMM(colIndexB,rowPtrB,valuesB,AR_int,C_int,colsB, rowsB,colsA, rowsA,nnzB);
            xtrans<int>(C_int,C_int,colsB,rowsA);

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix

            if(lambdaCR2!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR2!=0&&max_mAR!=0) xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }
    xcopy<T>(C_buffer,C, rowsA, colsB);
}



// Define matrix by matrix function ---- residual method + vectorization method
template <typename T,int digit>
void rcxigemm(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {


    
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

    const int max_int = (1<<(digit-1)) - 1;

    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, digit, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, digit, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    //Compute the multiplication of float and int matrices to get the resulting matrix


    // ilag2i8(A_int,A8_int,rowsA*colsA);
    // ilag2i8(B_int,B8_int,rowsB*colsB);
    
    //xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C_buffer,lambdaA_vec,lambdaB_vec,rowsA,colsB);

    if(type == 'c' || type == 'x'){
        
        //Dequantization of the quantized int matrix gives A',B'
        quantitize_vec<int,T>(A_int,A_p,lambdaA_vec ,rowsA,colsA,'d','r');
        quantitize_vec<int,T>(B_int,B_p,lambdaB_vec ,rowsB,colsB,'d','c');

        //Calculate the residuals matrix for full size
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);

        //The residual matrix is quantized
        T max_mAR =get_max<T>(A_p,rowsA,colsA);
        T max_mBR =get_max<T>(B_p,rowsB,colsB);
        T lambdaAR = (T)max_int/max_mAR;
        T lambdaBR = (T)max_int/max_mBR;
        //Direct quantization of A and B residual matrices
        quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
        quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');




        T lambdaAnew ,lambdaBnew ;
        if(type == 'x'){
            //Sparse error repair
            int nnzA0 = get_nnz<float>(A, rowsA, colsA);
            int nnzB0 = get_nnz<float>(B, rowsB, colsB);
            T avgA = get_nnz_avg<T>(A,rowsA,colsA,nnzA0);
            T avgB = get_nnz_avg<T>(B,rowsB,colsB,nnzB0);

            T ml_kA = 2*threadhoud*avgA;//*lambdaB/colsA;
            T ml_kB = 2*threadhoud*avgB;//*lambdaA/colsA;    

            reduce_Matrix(A_copy,lambdaA_vec,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,lambdaB_vec,rowsB,rowsB, ml_kB ,'c');

            float sp = get_sparsity(B_copy,rowsA,colsA);
            float sp2 = get_sparsity(A_copy,rowsA,colsA);

            if(sp2>SPARSITY) xcopy<T>(A,A_copy, rowsA, colsA);
            if(sp>SPARSITY) xcopy<T>(B,B_copy, rowsB, colsB);




            //Quantify the new AB
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');

            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            int nnzA = get_nnz<int>(A_int, rowsA, colsA);
            int nnzB = get_nnz<int>(B_int, rowsA, colsA);


            if(sp2<=SPARSITY){
                int* valuesA = new int[nnzA];
                long long int* colIndexA = new long long int[nnzA];
                long long int* rowPtrA = new long long int[rowsA + 1];
                denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


                //sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);
                mkl_SPMM(colIndexA,rowPtrA,valuesA,BR_int,C_int,rowsA, colsA,rowsB, colsB,nnzA);

            } else {
                eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            }


            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            if(sp<=SPARSITY){
                xtrans<int>(B_int,B_int,rowsB,colsB);
                xtrans<int>(AR_int,AR_int,rowsA, colsA);
                int* valuesB = new int[nnzB];
                long long int* colIndexB = new long long int[nnzB];
                long long int* rowPtrB = new long long int[colsB + 1];
                denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
                mkl_SPMM(colIndexB,rowPtrB,valuesB,AR_int,C_int,colsB, rowsB,colsA, rowsA,nnzB);                // sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
                
                xtrans<int>(C_int,C_int,colsB,rowsA);
            } else {
                eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            }
            
            

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix

            if(lambdaCR2!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //Quantify the new AB
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');

            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR2!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}


// Define matrix by matrix function ---- residual method + vectorization method
// print density
template <typename T,int digit>
void rcxigemm_print_sp(T A[], T B[], T C[], int rowsA, int colsA, int rowsB, int colsB,float threadhoud,char type) {


    
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

    const int max_int = (1<<(digit-1)) - 1;

    T* max_A_r = (T *)malloc(sizeof(T) * rowsA);
    get_max_vec<T>(A,max_A_r,rowsA, colsA,'r');
    T* max_B_c = (T *)malloc(sizeof(T) * colsB);
    get_max_vec<T>(B,max_B_c,rowsB, colsB,'c');
    T* lambdaA_vec = (T *)malloc(sizeof(T) * rowsA); 
    T* lambdaB_vec = (T *)malloc(sizeof(T) * colsB);

    get_lambda_vec(lambdaA_vec,max_A_r, digit, rowsA);
    get_lambda_vec(lambdaB_vec,max_B_c, digit, colsB);

    //Direct quantization of A and B matrices
    quantitize_vec<T,int>(A,A_int,lambdaA_vec ,rowsA,colsA,'q','r');
    quantitize_vec<T,int>(B,B_int,lambdaB_vec ,rowsB,colsB,'q','c');

    //Compute the multiplication of float and int matrices to get the resulting matrix


    // ilag2i8(A_int,A8_int,rowsA*colsA);
    // ilag2i8(B_int,B8_int,rowsB*colsB);
    
    //xgemm<int>(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    eigen_IGEMM(A_int,B_int,C_int,rowsA,colsA,rowsB,colsB);

    dequantitize_matrix<int,T>(C_int,C_buffer,lambdaA_vec,lambdaB_vec,rowsA,colsB);

    if(type == 'c' || type == 'x'){
        
        //Dequantization of the quantized int matrix gives A',B'
        quantitize_vec<int,T>(A_int,A_p,lambdaA_vec ,rowsA,colsA,'d','r');
        quantitize_vec<int,T>(B_int,B_p,lambdaB_vec ,rowsB,colsB,'d','c');

        //Calculate the residuals matrix for full size
        get_R<T>(A,A_p,A_p,rowsA,colsA);
        get_R<T>(B,B_p,B_p,rowsB,colsB);

        //The residual matrix is quantized
        T max_mAR =get_max<T>(A_p,rowsA,colsA);
        T max_mBR =get_max<T>(B_p,rowsB,colsB);
        T lambdaAR = (T)max_int/max_mAR;
        T lambdaBR = (T)max_int/max_mBR;
        //Direct quantization of A and B residual matrices
        quantitize<T,int>(A_p,AR_int,rowsA,colsA,lambdaAR,'q');
        quantitize<T,int>(B_p,BR_int,rowsB,colsB,lambdaBR,'q');




        T lambdaAnew ,lambdaBnew ;
        if(type == 'x'){
            //Sparse error repair
            int nnzA0 = get_nnz<float>(A, rowsA, colsA);
            int nnzB0 = get_nnz<float>(B, rowsB, colsB);
            T avgA = get_nnz_avg<T>(A,rowsA,colsA,nnzA0);
            T avgB = get_nnz_avg<T>(B,rowsB,colsB,nnzB0);
            // std::cout<<"avg = "<<avg<<"\n";

            //threadhoud/=avg;
            T ml_kA = 2*threadhoud*avgA;//*lambdaB/colsA;
            T ml_kB = 2*threadhoud*avgB;//*lambdaA/colsA;    
            // T ml_kA = threadhoud/colsA;
            // T ml_kB = threadhoud/colsA;   

            // get_avg_vec<T>(C_buffer,C_rows,rowsA,colsB,'r');
            // get_avg_vec<T>(C_buffer,C_cols,colsA,colsB,'c');


            // get_avg_vec<T>(A_copy,A_rows,rowsA,colsB,'r');
            // get_avg_vec<T>(B_copy,B_cols,colsA,colsB,'c');

            reduce_Matrix(A_copy,lambdaA_vec,rowsA,colsA, ml_kA ,'r');
            reduce_Matrix(B_copy,lambdaB_vec,rowsB,rowsB, ml_kB ,'c');

            float sp = get_sparsity(B_copy,rowsA,colsA);
            float sp2 = get_sparsity(A_copy,rowsA,colsA);
            printf("%.6lf",sp);
            //std::cout<<"sp1 = "<<sp<<"||sp2 = "<<sp2<<std::endl;
            // std::cout<<"ml_kA = "<<ml_kA<<"||ml_kB = "<<ml_kB<<std::endl;
            // std::cout<<"avgA = "<<avgA<<"||avgB = "<<avgB<<std::endl;
            // std::cout<<"NNZA = "<<nnzA0<<"||NNZB = "<<nnzB0<<std::endl;

            if(sp2>SPARSITY) xcopy<T>(A,A_copy, rowsA, colsA);
            if(sp>SPARSITY) xcopy<T>(B,B_copy, rowsB, colsB);




            //Quantify the new AB
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');

            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;
            int nnzA = get_nnz<int>(A_int, rowsA, colsA);
            int nnzB = get_nnz<int>(B_int, rowsA, colsA);


            if(sp2<=SPARSITY){
                int* valuesA = new int[nnzA];
                long long int* colIndexA = new long long int[nnzA];
                long long int* rowPtrA = new long long int[rowsA + 1];
                denseToCSR(A_int, valuesA,colIndexA,rowPtrA, rowsA, colsA);


                //sspmm<int>(valuesA,colIndexA,rowPtrA, BR_int, C_int,rowsA, colsA,rowsB, colsB);
                mkl_SPMM(colIndexA,rowPtrA,valuesA,BR_int,C_int,rowsA, colsA,rowsB, colsB,nnzA);

            } else {
                eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            }


            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            if(sp<=SPARSITY){
                xtrans<int>(B_int,B_int,rowsB,colsB);
                xtrans<int>(AR_int,AR_int,rowsA, colsA);
                int* valuesB = new int[nnzB];
                long long int* colIndexB = new long long int[nnzB];
                long long int* rowPtrB = new long long int[colsB + 1];
                denseToCSR(B_int, valuesB,colIndexB,rowPtrB, colsB, rowsB);
                mkl_SPMM(colIndexB,rowPtrB,valuesB,AR_int,C_int,colsB, rowsB,colsA, rowsA,nnzB);                // sspmm<int>(valuesB,colIndexB,rowPtrB, AR_int, C_int,colsB, rowsB,colsA, rowsA);
                
                xtrans<int>(C_int,C_int,colsB,rowsA);
            } else {
                eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            }
            
            

            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix

            if(lambdaCR2!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);


        } else {
            //Quantify the new AB
            T max_mAnew =get_max<T>(A_copy,rowsA,colsA);
            T max_mBnew =get_max<T>(B_copy,rowsB,colsB);
            lambdaAnew = max_mAnew==0?0:(T)max_int/max_mAnew;
            lambdaBnew = max_mBnew==0?0:(T)max_int/max_mBnew;
            quantitize<T,int>(A_copy,A_int,rowsA,colsA,lambdaAnew,'q');
            quantitize<T,int>(B_copy,B_int,rowsB,colsB,lambdaBnew,'q');

            //The error repair matrix float is obtained by inverse quantization of the int error repair matrix
            T lambdaCR1 = lambdaAnew*lambdaBR;
            T lambdaCR2 = lambdaAR*lambdaBnew;

            eigen_IGEMM(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            //xgemm<int>(A_int,BR_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR1,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR1!=0&&max_mBR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

            
            //xgemm<int>(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            eigen_IGEMM(AR_int,B_int,C_int,rowsA,colsA,rowsB,colsB);
            quantitize<int,T>(C_int,C_copy,rowsA,colsB,lambdaCR2,'d');
            //The error is supplemented by a repair matrix
            if(lambdaCR2!=0&&max_mAR!=0)xmadd<float>(C_copy,C_buffer,C_buffer,rowsA,colsB);

        }
    }

    xcopy<T>(C_buffer,C, rowsA, colsB);

}
