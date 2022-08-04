// System includes
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA runtime
#include <curand.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void printArray(float *A, int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			printf("%f ", A[i * n + j]);
		}
		printf("\n");
	}
}

/**
* Matrix multiplication (CUDA Kernel) on the device: C = A * B
*/
#define TILE_WIDTH 16
__global__ void
matrixMulCUDA(float *C, float *A, float *B, int n)
{
	int k;
	int row = threadIdx.y, col = threadIdx.x;
	float sum = 0.0f;
	for (k = 0; k < n; ++k) {
		sum += A[row * n + k] * B[k * n + col];
	}
	C[row * n + col] = sum;
}

#define TILE_WIDTH 64
__global__ void
matrixMulCUDA1(float *C, float *A, float *B, int n)
{
	int start_row = threadIdx.y * TILE_WIDTH;
	int end_row = start_row + TILE_WIDTH;
	int start_col = threadIdx.x * TILE_WIDTH;
	int end_col = start_col + TILE_WIDTH;
	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			float C_val = 0;
			for (int k = 0; k < n; ++k) {
				float A_elem = A[row * n + k];
				float B_elem = B[k * n + col];
				C_val += A_elem * B_elem;
			}
			C[row*n + col] = C_val;
		}
	}
}

__global__ void
matrixMulCUDA2(float *C, float *A, float *B, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	float C_val = 0;
	for (int k = 0; k < n; ++k) {
		float A_elem = A[row * n + k];
		float B_elem = B[k * n + col];
		C_val += A_elem * B_elem;
	}
	C[row*n + col] = C_val;
}

#define TILE_WIDTH 4
#define BLOCK_DIM 32
__global__ void
matrixMulCUDA3(float *C, float *A, float *B, int n)
{
	int start_row = blockDim.y * blockIdx.y + threadIdx.y * TILE_WIDTH;
	int end_row = start_row + TILE_WIDTH;
	int start_col = blockDim.x * blockIdx.x + threadIdx.x * TILE_WIDTH;
	int end_col = start_col + TILE_WIDTH;

	for (int row = start_row; row < end_row; row++) {
		for (int col = start_col; col < end_col; col++) {
			float CValue = 0;

			__shared__ float As[BLOCK_DIM][BLOCK_DIM];
			__shared__ float Bs[BLOCK_DIM][BLOCK_DIM];

			for (int z = 0; z < n / (BLOCK_DIM * TILE_WIDTH); z++) {

				As[threadIdx.y][threadIdx.x] = A[row * n + z * BLOCK_DIM + threadIdx.x];
				Bs[threadIdx.y][threadIdx.x] = B[(z * BLOCK_DIM + threadIdx.y) * n + col];
				
				__syncthreads();

				for (int k = 0; k < BLOCK_DIM; ++k)
					CValue += As[threadIdx.y][k] * Bs[k][threadIdx.x];

				__syncthreads();
			}

			C[row * n + col] = CValue;
		}
	}
}

void constantInit(float *data, int size, float val)
{
	for (int i = 0; i < size; ++i)
	{
		data[i] = val;
	}
}

void randInit(float *data, int size)
{
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			data[i * size + j] = rand() % 10;
		}
	}
}


/**
* Run a simple test of matrix multiplication using CUDA
*/
int matrixMultiply(int argc, char **argv, int n)
{
	// Allocate host memory for matrices A and B
	unsigned int size_A = n * n;
	unsigned int mem_size_A = sizeof(float)* size_A;
	float *h_A = (float *)malloc(mem_size_A);
	unsigned int size_B = n * n;
	unsigned int mem_size_B = sizeof(float)* size_B;
	float *h_B = (float *)malloc(mem_size_B);

	// Initialize host memory
	const float valB = 0.01f;
	//constantInit(h_A, size_A, 1.0f);
	//constantInit(h_B, size_B, valB);

	randInit(h_A, n);
	randInit(h_B, n);

	// Allocate device memory
	float *d_A, *d_B, *d_C;

	// Allocate host matrix C
	unsigned int mem_size_C = n * n * sizeof(float);
	float *h_C = (float *)malloc(mem_size_C);

	if (h_C == NULL)
	{
		fprintf(stderr, "Failed to allocate host matrix C!\n");
		exit(EXIT_FAILURE);
	}

	cudaError_t error;

	error = cudaMalloc((void **)&d_A, mem_size_A);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_A returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_B, mem_size_B);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_B returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMalloc((void **)&d_C, mem_size_C);

	if (error != cudaSuccess)
	{
		printf("cudaMalloc d_C returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// copy host memory to device
	error = cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_A,h_A) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	error = cudaMemcpy(d_B, h_B, mem_size_B, cudaMemcpyHostToDevice);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (d_B,h_B) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	// Setup execution parameters
	dim3 threads(32, 32, 1);
	dim3 grid(16, 16, 1);

	// Create and start timer
	printf("Computing result using CUDA Kernel...\n");

	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	error = cudaEventCreate(&start);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	cudaEvent_t stop;
	error = cudaEventCreate(&stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to create stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the start event
	error = cudaEventRecord(start, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record start event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Execute the kernel
	//matrixMulCUDA <<<grid, threads>>> (d_C, d_A, d_B, n);
	//matrixMulCUDA1 <<<grid, threads>>> (d_C, d_A, d_B, n);
	//matrixMulCUDA2 <<<grid, threads>>> (d_C, d_A, d_B, n);
	matrixMulCUDA3 <<<grid, threads>>> (d_C, d_A, d_B, n);

	error = cudaGetLastError();
	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch kernel!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Record the stop event
	error = cudaEventRecord(stop, NULL);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to record stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Wait for the stop event to complete
	error = cudaEventSynchronize(stop);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize on the stop event (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	float msecTotal = 0.0f;
	error = cudaEventElapsedTime(&msecTotal, start, stop);

	printf("Elapsed time in msec = %f\n", msecTotal);

	if (error != cudaSuccess)
	{
		fprintf(stderr, "Failed to get time elapsed between events (error code %s)!\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}

	// Copy result from device to host
	error = cudaMemcpy(h_C, d_C, mem_size_C, cudaMemcpyDeviceToHost);

	if (error != cudaSuccess)
	{
		printf("cudaMemcpy (h_C,d_C) returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
		exit(EXIT_FAILURE);
	}

	//printf("Array A:\n");
	//printArray(h_A, n);
	//printf("Array B:\n");
	//printArray(h_B, n);
	//printf("Array C:\n");
	//printArray(h_C, n);


	// Clean up memory
	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;

}


/**
* Program main
*/
int main(int argc, char **argv)
{
	printf("[Matrix Multiply Using CUDA] - Starting...\n");

	// By default, we use device 0
	int devID = 0;
	cudaSetDevice(devID);

	cudaError_t error;
	cudaDeviceProp deviceProp;
	error = cudaGetDevice(&devID);

	if (error != cudaSuccess)
	{
		printf("cudaGetDevice returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}

	error = cudaGetDeviceProperties(&deviceProp, devID);

	if (deviceProp.computeMode == cudaComputeModeProhibited)
	{
		fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
		exit(EXIT_SUCCESS);
	}

	if (error != cudaSuccess)
	{
		printf("cudaGetDeviceProperties returned error %s (code %d), line(%d)\n", cudaGetErrorString(error), error, __LINE__);
	}
	else
	{
		printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", devID, deviceProp.name, deviceProp.major, deviceProp.minor);
	}

	// Size of square matrices
	size_t n = 0;
	printf("[-] N = ");
	scanf("%u", &n);

	printf("MatrixA(%d,%d), MatrixB(%d,%d)\n", n, n, n, n);

	int matrix_result = matrixMultiply(argc, argv, n);

	exit(matrix_result);
}
