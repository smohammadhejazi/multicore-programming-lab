#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include "math.h"

void fillVector(int * v, size_t n);
void addVector(int * a, int *b, int *c, size_t n);
void printVector(int * v, size_t n);
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);
__global__ void addKernel(int *c, const int *a, const int *b);
__global__ void addKernel1(int *c, const int *a, const int *b);
__global__ void addKernel2(int *c, const int *a, const int *b);

const int n = 50;
const int vectorSize = n * 1024;
double start;
double finish;

int main()
{
	#ifndef _OPENMP
		printf("OpenMP is not supported, sorry!\n");
		getchar();
		return 0;
	#endif

	int a[vectorSize], b[vectorSize], c[vectorSize];

	fillVector(a, vectorSize);
	fillVector(b, vectorSize);

	start = omp_get_wtime();
	addWithCuda(c, a, b, vectorSize);
	finish = omp_get_wtime();

	printf("%f\n", finish - start);
	//printVector(a, vectorSize);
	//printf("*************\n");
	//printVector(b, vectorSize);
	//printf("************\n");
	//printVector(c, vectorSize);
	return EXIT_SUCCESS;
}

// Fills a vector with data
void fillVector(int * v, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		v[i] = i;
	}
}

// Adds two vectors
void addVector(int * c, int *a, int *b, size_t n) {
	int i;
	for (i = 0; i < n; i++) {
		c[i] = a[i] + b[i];
	}
}

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size) {
	int *dev_a = 0;
	int *dev_b = 0;
	int *dev_c = 0;
	cudaError_t cudaStatus;

	// chose gpu
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	// allocate vectors in gpu
	cudaStatus = cudaMalloc((void**)&dev_c, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_a, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	cudaStatus = cudaMalloc((void**)&dev_b, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	// copy vector a and b to gpu memory
	cudaStatus = cudaMemcpy(dev_a, a, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}
	cudaStatus = cudaMemcpy(dev_b, b, vectorSize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	// add vectors
	addKernel2 <<<n, 1024>>>(dev_c, dev_a, dev_b);

	// wait for gpu to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
	}

	// get last error
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// copy vector c back to cpu
	cudaStatus = cudaMemcpy(c, dev_c, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(dev_c);
	cudaFree(dev_a);
	cudaFree(dev_b);

	return cudaStatus;
}

// Adds two vector in gpu
__global__ void addKernel(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	c[i] = a[i] + b[i];
}

__global__ void addKernel1(int *c, const int *a, const int *b)
{
	int i = threadIdx.x;
	int offset = n * i;
	for (int j = 0; j < n; j++) {
		c[j + offset] = a[j + offset] + b[j + offset];
	}
}

__global__ void addKernel2(int *c, const int *a, const int *b)
{
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	while (tid < n * 1024) {
		c[tid] = a[tid] + b[tid];       
		tid += blockDim.x;       
								 
	}
}

// Prints a vector to the stdout.
void printVector(int * v, size_t n) {
	int i;
	printf("[-] Vector elements: ");
	for (i = 0; i < n; i++) {
		printf("%d, ", v[i]);
	}
	printf("\b\b  \n");
}
