/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date modified: April 26, 2023
 *
 * Student names(s): Charles Tran
 * Date: 5/1/2023
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

typedef struct thread_data_s {
    int tid;                /* Thread identifier */
    int num_threads;        /* Number of thread */
    int num_elements;       /* Number of elements in grid */
    Matrix *sharedMatrix;
    int offset;
    int chunk_size;
} thread_data_t;

pthread_barrier_t barrier; 
void *gauss_eliminate_solver(void *args);

void print_matrix(Matrix M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s matrix-size & num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        fprintf(stderr, "num-threads: number of thread\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    Matrix A;			                                            /* Input matrix */
    Matrix U_reference;		                                        /* Upper triangular matrix computed by reference code */
    Matrix U_mt;			                                        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time (NULL));                                             /* Seed random number generator */
    A = allocate_matrix(matrix_size, matrix_size, 1);               /* Allocate and populate random square matrix */
    U_reference = allocate_matrix (matrix_size, matrix_size, 0);    /* Allocate space for reference result */
    U_mt = allocate_matrix (matrix_size, matrix_size, 0);           /* Allocate space for multi-threaded result */

    pthread_barrier_init(&barrier, NULL, num_threads);
    /* Copy contents A matrix into U matrices */
    int i, j;
    for (i = 0; i < A.num_rows; i++) {
        for (j = 0; j < A.num_rows; j++) {
            U_reference.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
            U_mt.elements[A.num_rows * i + j] = A.elements[A.num_rows * i + j];
        }
    }
    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);
    int status = compute_gold(U_reference.elements, A.num_rows);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0) {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }
  
    status = perform_simple_check(U_reference);	/* Check that principal diagonal elements are 1 */ 
    if (status < 0) {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");
  
    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");
    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(U_mt, num_threads);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec\
        + (stop.tv_usec - start.tv_usec) / (float)1000000));
    fprintf(stderr, "Multi-threaded Gaussian elimination was successful.\n");
    
    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(A.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}


/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
    pthread_t *tid = (pthread_t *) malloc(sizeof(pthread_t) * U.num_rows);
    thread_data_t *thread_data = (thread_data_t *) malloc(sizeof(thread_data_t) * num_threads);
    int i;
    int num_elements = U.num_columns * U.num_rows;
    int chunk_size = (int)floor((float)num_elements/(float)num_threads);

    for (i = 0; i < num_threads; i++){
        thread_data[i].tid = i;
        thread_data[i].num_threads = num_threads;
        thread_data[i].num_elements = U.num_rows;
        thread_data[i].sharedMatrix = &U;
        thread_data[i].offset = i * chunk_size;
        thread_data[i].chunk_size = chunk_size;
        if(pthread_create(&tid[i], NULL, gauss_eliminate_solver, (void*)&thread_data[i]) != 0)
            perror("pthread_create");
    }

    for (i = 0; i < thread_data->num_threads; i++)
        pthread_join(tid[i], NULL);
    
    free((void *)tid);
    free((void *)thread_data);
}

void *gauss_eliminate_solver(void *args) {
    thread_data_t *thread_data = (thread_data_t*) args;
    int i, j;
    int num_elements = thread_data->num_elements; 
    float* U = thread_data->sharedMatrix->elements;
    int offset = thread_data->tid;
    int stride = thread_data->num_threads;

    for (i = 0; i < num_elements; i++) {
        if (thread_data->tid == 0) {
            for (j = (i + 1); j < num_elements; j++) { 
                if (U[num_elements * i  + i ] == 0) {
                    fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                    exit (EXIT_FAILURE);
                }
                U[num_elements * i  + j] = (float)(U[num_elements * i  + j] / U[num_elements * i + i ]); /* Division step */
            };
            U[num_elements * i + i] = 1; /* Set the principal diagonal entry in U to 1 */            
        }
        pthread_barrier_wait(&barrier);
        
        while (offset < num_elements) {
            if (offset > i){
                for (j = i+1; j < num_elements; j++)             
                    U[num_elements * offset + j] = U[num_elements * offset + j] - (U[num_elements * offset +i] * U[num_elements * i + j]);	/* Elimination step */

                U[num_elements*offset + i] = 0;
            }
            
            offset += stride;
        }

        offset = thread_data->tid;
        pthread_barrier_wait(&barrier);

    }
    pthread_exit(NULL);

}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *A, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if(fabsf(A[i] - B[i]) > tolerance)
            return -1;
    return 0;
}


/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));
  
    for (i = 0; i < size; i++) {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }
  
    return M;
}

/* Return a random floating-point number between [min, max] */ 
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;
  
    return 0;
}


