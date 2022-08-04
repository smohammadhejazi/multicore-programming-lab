#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "omp.h"
#include "math.h"

cudaError_t getIds(int *a, unsigned int size);
__global__ void printKernel(int *a);

const int block = 2;
const int threadPerBlock = 64;
const int idsNum = 4;
const int vectorSize = block * threadPerBlock * idsNum;

int main()
{
	int a[vectorSize];
	getIds(a, vectorSize);
	for (int i = 0; i < block * threadPerBlock; i++) {
		int index = i * idsNum;
		printf("Calculated GThread: %d - Block: %d - Wrap: %d - LThread:%d\n",
			a[index + 0], a[index + 1], a[index + 2], a[index + 3]);
	}
	return EXIT_SUCCESS;
}


cudaError_t getIds(int *a, unsigned int size) {
	int *dev_a = 0;
	cudaError_t cudaStatus;

	// chose gpu
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		printf("cudaSetDevice failed! Do you have a CUDA-capable GPU installed?");
	}

	// allocate vectors in gpu
	cudaStatus = cudaMalloc((void**)&dev_a, vectorSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("cudaMalloc failed!");
	}

	// add vectors
	printKernel << <2, 64 >> > (dev_a);

	// wait for gpu to finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		printf("cudaDeviceSynchronize returned error code %d after launching printKernel!\n", cudaStatus);
	}

	// get last error
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("printKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
	}

	// copy vector c back to cpu
	cudaStatus = cudaMemcpy(a, dev_a, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("cudaMemcpy failed!");
	}

	cudaFree(dev_a);

	return cudaStatus;
}

__global__ void printKernel(int *a)
{
	int globalID = blockDim.x * blockIdx.x + threadIdx.x;
	int index = globalID * idsNum;
	a[index] = globalID;									// Global Id
	a[index + 1] = blockIdx.x;								// Block Id
	a[index + 2] = threadIdx.x / warpSize;					// Wrap Id
	a[index + 3] = threadIdx.x;								// Local Id
}