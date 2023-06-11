/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: APril 26, 2023
 *
 * Student name(s): Charles Tran
 * Date modified: 5/1/2023
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/



#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <math.h>
#include "jacobi_solver.h"
#include <semaphore.h>
#include <pthread.h>

/* Uncomment the line below to spit out debug information */ 
//#define DEBUG

typedef struct thread_data_s {
    int tid;                        /* Thread identifier */
    int num_threads;                /* Number of thread in worker pool */
    int result;                     /* Store the result */
    int num_iter;                  /* Pointer to number of iteration */
	int max_iter;
    pthread_mutex_t *sharedMutex;   /* Mutex shared */
	int chunk_size;
	int offset;
	double *diff;
	const matrix_t *A;
	const matrix_t *B;
	matrix_t *src;
    matrix_t *dest;
} thread_data_t;


pthread_mutex_t mtx;
pthread_barrier_t barrier; 


int main(int argc, char **argv) 
{
	if (argc < 3) {
		fprintf(stderr, "Usage: %s matrix-size num-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
        fprintf(stderr, "num-threads: number of worker threads to create\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x_v1;      /* Solution computed by pthread code using chunking */
    matrix_t mt_solution_x_v2;      /* Solution computed by pthread code using striding */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	

    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x_v1 = allocate_matrix(matrix_size, 1, 0);
    mt_solution_x_v2 = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif
	pthread_mutex_init(&mtx, NULL);
	pthread_barrier_init(&barrier, NULL, num_threads);

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    compute_gold(A, reference_x, B, max_iter);
	// print_matrix(reference_x);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x_v1 and mt_solution_x_v2.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using chunking\n");
	compute_using_pthreads_v1(A, mt_solution_x_v1, B, max_iter, num_threads);
	// print_matrix(mt_solution_x_v1);
    display_jacobi_solution(A, mt_solution_x_v1, B); /* Display statistics */
    
	fprintf(stderr, "\nPerforming Jacobi iteration using pthreads using striding\n");
	compute_using_pthreads_v2(A, mt_solution_x_v2, B, max_iter, num_threads);
	// print_matrix(mt_solution_x_v2);
    display_jacobi_solution(A, mt_solution_x_v2, B); /* Display statistics */

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x_v1.elements);
    free(mt_solution_x_v2.elements);
	
    exit(EXIT_SUCCESS);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using chunking. 
 * Result must be placed in mt_sol_x_v1. */
void compute_using_pthreads_v1(const matrix_t A, matrix_t mt_sol_x_v1, const matrix_t B, int max_iter, int num_threads)
{
	int i;
	int num_iter = 0;
	double diff = 0.0;
	int num_elements = A.num_rows;
	pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
	int chunk_size = (int)floor((float)num_elements/(float)num_threads); // Compute the chunk size
	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	matrix_t dest = allocate_matrix(A.num_rows, 1, 0);
	
	for (i=0; i < num_threads; i++) {
		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].max_iter = max_iter;
		thread_data[i].chunk_size = chunk_size;
		thread_data[i].A = &A;
		thread_data[i].B = &B;
		thread_data[i].src = &mt_sol_x_v1;
		thread_data[i].dest = &dest;
		thread_data[i].num_iter = num_iter;
		thread_data[i].diff = &diff;
		
		pthread_create(&tid[i], NULL, compute_v1, (void *)&thread_data[i]);
	}

	/* Wait for threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);
	
	if (thread_data->num_iter < thread_data->max_iter)
        fprintf(stderr, "Convergence achieved after %d iterations\n", thread_data->num_iter);
    else
        fprintf(stderr, "Maximum allowed iterations reached\n");
	  
	free((void *) tid);
	free((void *) thread_data);


}

void *compute_v1(void *args) {

	thread_data_t *thread_data = (thread_data_t *)args;
	int converged = 0;
	int chunk_size = thread_data->chunk_size;
	int tid = thread_data->tid;
	int i;
	int j;
	double mse;	
	const matrix_t* A= thread_data->A;
	const matrix_t* B= thread_data->B;
	matrix_t* src= thread_data->src;
	matrix_t* dest= thread_data->dest;

	if (thread_data->tid < thread_data->num_threads - 1) {
		while (!converged) {
			double p_diff = 0.0;
			if (thread_data->tid == 0) {
				*thread_data->diff = 0.0;
				mse = 0.0;
			}
			
			pthread_barrier_wait(&barrier);
			
			for (i = chunk_size * tid; i < chunk_size * tid + chunk_size; i++) {
				double sum = 0.0;
				for (j = 0; j < A->num_columns; j++) {
					if (i != j){
						sum += A->elements[i * A->num_columns + j] * src->elements[j];
					}
				}
				dest->elements[i] = (B->elements[i] - sum) / A->elements[i * A->num_columns + i];
				p_diff += ( dest->elements[i] - src->elements[i] )  * ( dest->elements[i] - src->elements[i]);		
			}
		

			pthread_mutex_lock(&mtx);
			*thread_data->diff += p_diff;
			pthread_mutex_unlock(&mtx);
			pthread_barrier_wait(&barrier);

			thread_data->num_iter++;
			mse = sqrt(*thread_data->diff);
			if ((mse <= THRESHOLD) || (thread_data->num_iter == thread_data->max_iter))
				converged = 1;
			
		
			matrix_t *temp = src;
			src = dest;
			dest = temp;
			pthread_barrier_wait(&barrier);
		}

	}
	else {
		while (!converged) {
			double p_diff = 0.0;
			if (thread_data->tid == 0) {
				*thread_data->diff = 0.0;
				mse = 0.0;
			}

			pthread_barrier_wait(&barrier);
			for (i = chunk_size * tid; i < chunk_size * tid + chunk_size; i++) {
				double sum = 0.0;
				for (j = 0; j < A->num_columns; j++) {
					if (i != j){
						sum += A->elements[i * A->num_columns + j] * src->elements[j];
					}
				}
				dest->elements[i] = (B->elements[i] - sum) / A->elements[i * A->num_columns + i];
				p_diff += ( dest->elements[i] - src->elements[i] )  * ( dest->elements[i] - src->elements[i]);		
			}
		
			pthread_mutex_lock(&mtx);
			*thread_data->diff += p_diff;
			pthread_mutex_unlock(&mtx);
			pthread_barrier_wait(&barrier);

			thread_data->num_iter++;
			mse = sqrt(*thread_data->diff);
			if ((mse <= THRESHOLD) || (thread_data->num_iter == thread_data->max_iter))
				converged = 1;
			
		
			matrix_t *temp = src;
			src = dest;
			dest = temp;
			pthread_barrier_wait(&barrier);
		}

	}
	pthread_exit(NULL);
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads using striding. 
 * Result must be placed in mt_sol_x_v2. */
void compute_using_pthreads_v2(const matrix_t A, matrix_t mt_sol_x_v2, const matrix_t B, int max_iter, int num_threads)
{
	int i;
	int num_iter = 0;
	double diff = 0.0;
	pthread_t *tid = (pthread_t *)malloc(sizeof(pthread_t) * num_threads);
	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	matrix_t dest = allocate_matrix(A.num_rows, 1, 0);
	for (i=0; i < num_threads; i++) {
		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].max_iter = max_iter;
		thread_data[i].offset = 0;
		thread_data[i].chunk_size = 0;
		thread_data[i].A = &A;
		thread_data[i].B = &B;
		thread_data[i].src = &mt_sol_x_v2;
		thread_data[i].dest = &dest;
		thread_data[i].num_iter = num_iter;
		thread_data[i].diff = &diff;
		pthread_create(&tid[i], NULL, compute_v2, (void *)&thread_data[i]);
	}

	/* Wait for threads to finish */
    for (i = 0; i < num_threads; i++)
        pthread_join(tid[i], NULL);
	
	if (thread_data->num_iter < thread_data->max_iter)
        fprintf(stderr, "Convergence achieved after %d iterations\n", thread_data->num_iter);
    else
        fprintf(stderr, "Maximum allowed iterations reached\n");
	  
	free((void *) tid);
	free((void *) thread_data);
}

void *compute_v2(void *args) {
	thread_data_t *thread_data = (thread_data_t *)args;
	int converged = 0;
	
	double mse;
    int offset = thread_data->tid;
    int stride = thread_data->num_threads;

	const matrix_t* A = thread_data->A;
	const matrix_t* B = thread_data->B;


	matrix_t* src= thread_data->src;
	matrix_t* dest= thread_data->dest;

	int j;

	while (!converged) {
		double p_diff = 0.0;
		if (thread_data->tid == 0) {
			*thread_data->diff = 0.0;
			mse = 0;
		}

		pthread_barrier_wait(&barrier);
		while (offset < A->num_rows) {
			double sum = 0.0;
			for (j = 0; j < A->num_columns; j++) {				
				if (offset != j){
					sum += A->elements[offset * A->num_columns + j] * src->elements[j];
				}
			}
			dest->elements[offset] = (B->elements[offset] - sum) /A->elements[offset * A->num_columns + offset];
			p_diff += (src->elements[offset] - dest->elements[offset] )  * (src->elements[offset] - dest->elements[offset]); 
			offset += stride;
		}

		pthread_mutex_lock(&mtx);
		*thread_data->diff += p_diff;
		pthread_mutex_unlock(&mtx);		
		pthread_barrier_wait(&barrier);

		thread_data->num_iter++;
		mse = sqrt(*thread_data->diff);
		
		if ((mse <= THRESHOLD) || (thread_data->num_iter == thread_data->max_iter))
			converged = 1;


		matrix_t *temp = src;
		src = dest;
		dest = temp;
		offset = thread_data->tid;
		pthread_barrier_wait(&barrier);
		
	}
	pthread_exit(NULL);
}	


/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/

matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
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

/* Print matrix to screen */
void print_matrix(const matrix_t M)
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

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}



