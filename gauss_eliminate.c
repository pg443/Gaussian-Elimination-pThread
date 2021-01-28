/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): FIXME
 * Date: FIXME
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
#include <semaphore.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

typedef struct barrier_struct {
    sem_t counter_sem; /* Protects access to the counter */
    sem_t barrier_sem; /*Signals that barrier is safe to cross */
    int counter; /* The counter value */
} BARRIER;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int, int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);
void barrier_sync (BARRIER *, int, int);
void *worker_thread(void *);

/* Arguments struct for the threads */
typedef struct {
    Matrix* A;
    int tid;
    int num_elements;
    int num_of_threads;   
} args_for_thread;

/* Create the barrier data structure */
BARRIER barrier;  
BARRIER barrier2;
int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s matrix-size number-of-threads\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        fprintf(stderr, "number-of-threads: the number of parallel threads to be created\n");
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
    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
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
    gauss_eliminate_using_pthreads(U_mt, matrix_size, num_threads);
    gettimeofday(&stop, NULL);

    fprintf(stderr, "CPU run time = %f s\n", (float)(stop.tv_sec - start.tv_sec\
                + (stop.tv_usec - start.tv_usec) / (float)1000000));

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
void gauss_eliminate_using_pthreads(Matrix U, int num_elems, int num_threads)
{
    /* Initialize the barrier data structure */
    barrier.counter = 0;
    sem_init (&barrier.counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to 1 */
    sem_init (&barrier.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to 0 */

    /* Initialize the barrier data structure */
    barrier2.counter = 0;
    sem_init (&barrier2.counter_sem, 0, 1); /* Initialize the semaphore protecting the counter to 1 */
    sem_init (&barrier2.barrier_sem, 0, 0); /* Initialize the semaphore protecting the barrier to 0 */

    pthread_t *thread_id = (pthread_t *)malloc (num_threads * sizeof(pthread_t)); /* Data structure to store the thread IDs */
    pthread_attr_t attributes;      /* Thread attributes */
    pthread_attr_init(&attributes); /* Initialize thread attributes to default values */
    args_for_thread* thread_arg = (args_for_thread*)malloc(num_threads * sizeof(args_for_thread));

    int i;
    for (i = 0; i < num_threads; i++){
        thread_arg[i].A = &U;
        thread_arg[i].tid = i;
        thread_arg[i].num_elements = num_elems;
        thread_arg[i].num_of_threads = num_threads;
        pthread_create(&thread_id[i], &attributes, worker_thread, (void *)&thread_arg[i]);
    }
    for (i = 0; i < num_threads; i++)
        pthread_join(thread_id[i], NULL);

    free((void *)thread_id);
    free((void*)thread_arg);
}

void *worker_thread(void *args){
    args_for_thread *thread_data = (args_for_thread *)args;
    float *U = thread_data->A->elements;
    int num_elements = thread_data->num_elements;
    int i, j, k, chunk, num_elem, end, start;
    int last_thread = 0;
    if (thread_data->num_of_threads == thread_data->tid + 1) last_thread++;

    for (k = 0; k < num_elements; k++){
        num_elem = num_elements - k - 1;
        chunk = (int)floor((float)num_elem/(float) thread_data->num_of_threads);
        start = k + 1 + (thread_data->tid * chunk);
        if(last_thread){
            end = num_elements;
        }else{
            end =  start + chunk;
        }

        for (j = start; j < end; j++) {   /* Reduce the current row. */
            if (U[num_elements * k + k] == 0) {
                fprintf(stderr, "Numerical instability. The principal diagonal element is zero.\n");
                return -1;
            }            
            U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	/* Division step */
        }

        barrier_sync (&barrier, thread_data->tid, thread_data->num_of_threads);
        U[num_elements * k + k] = 1;	/* Set the principal diagonal entry in U to 1 */ 

        for (i = start; i < end; i++) {
            for (j = k+1; j < num_elements; j++)
                U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);	/* Elimination step */
            
            U[num_elements * i + k] = 0;
        }
        barrier_sync (&barrier2, thread_data->tid, thread_data->num_of_threads);
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

void print_matrix(const Matrix A){
    int i,j;
    for (i = 0; i < A.num_rows; i++){
        for(j = 0; j < A.num_columns; j++){
            printf("%f\t", A.elements[A.num_rows * i + j]);
        }
        printf("\n");
    }
}

/* The function that implements the barrier synchronization. */
void 
barrier_sync (BARRIER *barrier, int thread_number, int num_threads)
{
    sem_wait (&(barrier->counter_sem)); /* Obtain the lock on the counter */

    /* Check if all threads before us, that is NUM_THREADS-1 threads have reached this point */
    if (barrier->counter == (num_threads - 1)) {
        barrier->counter = 0; /* Reset the counter */
					 
        sem_post (&(barrier->counter_sem)); 
		int i;
        /* Signal the blocked threads that it is now safe to cross the barrier */			 
        // printf("Thread number %d is signalling other threads to proceed. \n", thread_number); 			 
        for (i = 0; i < (num_threads - 1); i++)
            sem_post (&(barrier->barrier_sem));
    } 
    else {
        barrier->counter++; // Increment the counter
        sem_post (&(barrier->counter_sem)); // Release the lock on the counter
        sem_wait (&(barrier->barrier_sem)); // Block on the barrier semaphore and wait for someone to signal us when it is safe to cross
    }
}
	