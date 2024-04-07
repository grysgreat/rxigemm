#include <stdio.h>
#include <iostream>
#include <random>
#include <omp.h>

template <typename T>
void generate_matrix(T* matrix,int rows,int cols,char type ){
    // Create a random number engine
    std::random_device rd;
    std::mt19937 gen(rd());
    // Create a uniform distribution with the range [0, 1)
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    //std::normal_real_distribution<float> dis(0.0, 1.0);
    int max_omp_thread = omp_get_max_threads();
    if(type == 'u'){
        #pragma omp parallel for num_threads(max_omp_thread)
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
                std::normal_distribution<double> dis(10.0, 3.0); 
                matrix[i*cols+j] = dis(gen);
            }
        }        
    }else if(type == 'e'){
        for(int i=0;i<rows;i++){
            for(int j=0;j<cols;j++){
                std::exponential_distribution<double> dis(1.0/(1.0/4.0)); 
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



void printMatrix(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%.4f   \t", matrix[index]);
        }
        std::cout << std::endl;
    }
}

void printMatrix_latex(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if(j!=2) printf("%.2f&", matrix[index]);
            else printf("%.2f\\\\", matrix[index]);
        }
        std::cout << std::endl;
    }
}

void printMatrix_int_latex(int matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            if(j!=2) printf("%d&", matrix[index]);
            else printf("%d\\\\", matrix[index]);
        }
        std::cout << std::endl;
    }
}

void printMatrix_h(float matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%.8f   \t", matrix[index]);
        }
        std::cout << std::endl;
    }
}

void printMatrix_int(int matrix[], int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int index = i * cols + j;
            printf("%d\t\t", matrix[index]);
        }
        std::cout << std::endl;
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
float get_sparsity(T A[],int rows, int cols){
    int cnt=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            if(A[i*cols+j]!=0){
                cnt++;
            } 
        }
    }     
    float sp = (float)cnt / (float)(rows*cols);
    return sp;
}

