/*
*				In His Exalted Name
*	Title:	Prefix Sum Sequential Code
*	Author: Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	Date:	29/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include <math.h>

#define NUM_THREADS 4

void omp_check();
void fill_array(int *a, size_t n);
void prefix_sum(int *a, size_t n);
void print_array(int *a, size_t n);

int main(int argc, char *argv[]) {
	double start;
	start = omp_get_wtime();
	// Check for correct compilation settings
	omp_check();
	omp_set_num_threads(NUM_THREADS);
	// Input N
	size_t n = 0;
	int p = 5;
	n = (int)(pow(10, p));
	//n *= 2;
    // n = 16;
	printf("[-] Parallel 2 P: %d\n", p);
	// scanf("%uld\n", &n);
	
	// Allocate memory for array
	int * a = (int *)malloc(n * sizeof a);
	// Fill array with numbers 1..n
	fill_array(a, n);
	// Print array
	// print_array(a, n);
	// Compute prefix sum
	prefix_sum(a, n);
	// Print array
	// print_array(a, n);
	// Free allocated memory
    printf("\nhere");
	free(a);
    printf("\nthere");
	printf("%f\n", omp_get_wtime() - start);
    printf("\ntherfore");
	//system("PAUSE");
	return EXIT_SUCCESS;
}

void prefix_sum(int *a, size_t n) {
    int step = (int) log2(n);
    int neighbour;
	#pragma omp parallel
	{
        for (int i = 0; i <= step; i++)
        { 
            neighbour = (int) pow(2, i);
            #pragma omp single
            for (int j = n; j >= neighbour; j--)
            {
                #pragma omp task
                {
                    a[j] = a[j] + a[j - neighbour];
                }
            }
        }
	}
}


void print_array(int *a, size_t n) {
	int i;
	printf("[-] array: ");
	for (i = 0; i < n; ++i) {
		printf("%d, ", a[i]);
	}
	printf("\b\b  \n");
}

void fill_array(int *a, size_t n) {
	int i;
	for (i = 0; i < n; ++i) {
		a[i] = i + 1;
	}
}

void omp_check() {
	printf("------------ Info -------------\n");
	#ifdef _DEBUG
		printf("[!] Configuration: Debug.\n");
		#pragma message ("Change configuration to Release for a fast execution.")
	#else
		printf("[-] Configuration: Release.\n");
	#endif // _DEBUG

	#ifdef _M_X64
		printf("[-] Platform: x64\n");
	#elif _M_IX86 
		printf("[-] Platform: x86\n");
		#pragma message ("Change platform to x64 for more memory.")
	#endif // _M_IX86 
	#ifdef _OPENMP
		printf("[-] OpenMP is on.\n");
		printf("[-] OpenMP version: %d\n", _OPENMP);
	#else
		printf("[!] OpenMP is off.\n");
		printf("[#] Enable OpenMP.\n");
	#endif // _OPENMP
		printf("[-] Maximum threads: %d\n", omp_get_max_threads());
		printf("[-] Nested Parallelism: %s\n", omp_get_nested() ? "On" : "Off");
	#pragma message("Enable nested parallelism if you wish to have parallel region within parallel region.")
	printf("===============================\n");
}
