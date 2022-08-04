/*
*	In His Exalted Name
*	Matrix Addition - Sequential Code
*	Ahmad Siavashi, Email: siavashi@aut.ac.ir
*	15/04/2018
*/

// Let it be.
#define _CRT_SECURE_NO_WARNINGS
#define NUMBER_OF_EXPERIMENT 1
#define NUMBER_OF_SQUARES 64

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <omp.h>

typedef struct {
	int *A, *B, *C;
	int n, m;
} DataSet;

void fillDataSet(DataSet *dataSet);
void printDataSet(DataSet dataSet);
void closeDataSet(DataSet dataSet);
void add(DataSet dataSet, int numberOfThreads);

int main(int argc, char *argv[]) {
#ifndef _OPENMP
	printf("OpenMP is not supported, sorry!\n");
	getchar();
	return 0;
#endif

	double starttime, elapsedtime;
	double timeSum = 0;
	double times[NUMBER_OF_EXPERIMENT];
	int numberOfThreads = 0;
	DataSet dataSet;
	if (argc < 4) {
		printf("[-] Invalid No. of arguments.\n");
		printf("[-] Try -> <n> <m> <t>\n");
		printf(">>> ");
		scanf("%d %d %d", &dataSet.n, &dataSet.m, &numberOfThreads);
	}
	else {
		dataSet.n = atoi(argv[1]);
		dataSet.m = atoi(argv[2]);
		numberOfThreads = atoi(argv[3]);
		printf("%d %d %d\n", dataSet.n, dataSet.m, numberOfThreads);
	}
	for (int i = 0; i < NUMBER_OF_EXPERIMENT; i++) {
		fillDataSet(&dataSet);
		starttime = omp_get_wtime();
		omp_set_num_threads(numberOfThreads);
		add(dataSet, numberOfThreads);
		elapsedtime = omp_get_wtime() - starttime;
		times[i] = elapsedtime;
		//printDataSet(dataSet);
		closeDataSet(dataSet);
	}

	for (int i = 0; i < NUMBER_OF_EXPERIMENT; i++) {
		timeSum += times[i];
	}
	printf("Parallel 2\n");
	printf("Average time: %f\n", timeSum / NUMBER_OF_EXPERIMENT);
	return EXIT_SUCCESS;
}

void fillDataSet(DataSet *dataSet) {
	int i, j;

	dataSet->A = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->B = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);
	dataSet->C = (int *)malloc(sizeof(int) * dataSet->n * dataSet->m);

	srand(time(NULL));

	for (i = 0; i < dataSet->n; i++) {
		for (j = 0; j < dataSet->m; j++) {
			dataSet->A[i*dataSet->m + j] = rand() % 100;
			dataSet->B[i*dataSet->m + j] = rand() % 100;
		}
	}
}

void printDataSet(DataSet dataSet) {
	int i, j;

	printf("[-] Matrix A\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.A[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix B\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-4d", dataSet.B[i*dataSet.m + j]);
		}
		putchar('\n');
	}

	printf("[-] Matrix C\n");
	for (i = 0; i < dataSet.n; i++) {
		for (j = 0; j < dataSet.m; j++) {
			printf("%-8d", dataSet.C[i*dataSet.m + j]);
		}
		putchar('\n');
	}
}

void closeDataSet(DataSet dataSet) {
	free(dataSet.A);
	free(dataSet.B);
	free(dataSet.C);
}

void add(DataSet dataSet, int numberOfThreads) {
	int i, j, k, iStart, iEnd, jStart, jEnd;
    int chunksInRow = sqrt(NUMBER_OF_SQUARES);
    int cellsInChunk = dataSet.n / chunksInRow;
	#pragma omp parallel for private(i, j, iStart, iEnd, jStart, jEnd)
    for (k = 0; k < NUMBER_OF_SQUARES; k++) {
        iStart = floor(k / chunksInRow) * cellsInChunk;
        iEnd = iStart + cellsInChunk;
        jStart = cellsInChunk * (k % chunksInRow);
        jEnd = jStart + cellsInChunk;
        for (i = iStart; i < iEnd; i++) {
            for (j = jStart; j < jEnd; j++) {
                dataSet.C[i * dataSet.m + j] = dataSet.A[i * dataSet.m + j] + dataSet.B[i * dataSet.m + j];
            }
        }
    }
	#pragma opm barrier
}
