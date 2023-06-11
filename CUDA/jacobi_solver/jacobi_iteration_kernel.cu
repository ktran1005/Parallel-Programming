#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */
__global__ void jacobi_iteration_kernel_naive(float *A, float *B, float *X, float *X_new, int num_rows, int num_cols, double *ssd)
{
    __shared__ double local_ssd[MATRIX_SIZE];
    int row = blockDim.y * blockIdx.y + threadIdx.y; /* Obtain row number. */
    int j;
    double sum;

    float old_value = X[row];
    sum = -A[row * num_cols + row] * X[row];
    for(j = 0; j < num_cols; j++)
	    sum += A[row * num_cols + j] * X[j];
    __syncthreads();
    
    X_new[row] = (B[row] - sum) / A[row * num_cols + row]; 
    double localSsd = (old_value - X_new[row]) * (old_value - X_new[row]);
    local_ssd[threadIdx.y] = localSsd;
    __syncthreads();

    for (int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.y  < stride) {
            local_ssd[threadIdx.y] += local_ssd[threadIdx.y + stride];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) 
        atomicAdd(ssd, local_ssd[0]);

    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *B, float *X, float *X_new, int num_rows, int num_cols, double *ssd)
{
    __shared__ double local_ssd[MATRIX_SIZE];

    int column = blockDim.y * blockIdx.y + threadIdx.y; /* Obtain column number */  
    float old_value = X[column];
    int j;

    double sum =  -A[column * num_cols + column] * X[column];
    for (j = 0; j < num_cols; j++) {
            sum += A[j * num_cols + column] * X[j];            
    }
    X_new[column] = (B[column] - sum) / A[column * num_cols + column]; 
    __syncthreads();

    double localSsd = (old_value - X_new[column]) * (old_value - X_new[column]);
    local_ssd[threadIdx.y] = localSsd;
    __syncthreads();

    for (int stride = blockDim.y >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.y  < stride) {
            local_ssd[threadIdx.y] += local_ssd[threadIdx.y + stride];
        }
        __syncthreads();
    }

    if (threadIdx.y == 0) 
        atomicAdd(ssd, local_ssd[0]);
        
    return;
}

