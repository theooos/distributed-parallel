/*
 * Theodore Gregory - 1453831
 *
 * Tasks:
 * Block scan
 * Full scan for large vectors
 * Bank Conflict Avoidance Optimisation
 *
 * Timings:
 * Block scan without BCAO
 * Block scan with BCAO
 * Full scan without BCAO
 * Full scan with BCAO
 *
 * Hardware:
 * CPU - Intel - Core i5-6600 3.3GHz Quad-Core Processor
 * GPU - Zotac - GeForce GTX 1080 8GB AMP! Edition Video Card
 *
 * Implementation details:
 * TODO Any details or performance strategies I implemented which improve upon a base level of the target goals
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

// A helper macro to simplify handling cuda error checking
#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}

#define BLOCK_SIZE 1024

#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void
block(int *g_odata, int *g_idata, int n)
{
	__shared__ int temp[BLOCK_SIZE*2];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2*thid] = g_idata[2*thid];     // load input into shared memory
	temp[2*thid+1] = g_idata[2*thid+1];
	for (int d = n>>1; d > 0; d >>= 1)  // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid == 0) { temp[n - 1] = 0; } // clear the last element
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[2*thid] = temp[2*thid]; // write results to device memory
	g_odata[2*thid+1] = temp[2*thid+1];
}


static void compare_results(const int *vector1, const int *vector2, int num_elements)
{
	for (int i = 0; i < num_elements; ++i){
		if (fabs(vector1[i] - vector2[i]) > 1e-5f){
			fprintf(stderr, "Result verification failed at element %d!  h%d : d%d\n", i, vector1[i], vector2[i]);
			exit (EXIT_FAILURE);
		}
	}
}


int main(void)
{
	cudaError_t err = cudaSuccess;

	// ***************** Initial variable construction ******************
	cudaEvent_t start, stop;
	float time_bscan, time_bscan_bcao, time_fscan, time_fscan_bcao;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint num_elements = 2048;
	size_t size = num_elements * sizeof(int);
	int grid_size = 1 + (num_elements - 1) / BLOCK_SIZE;

	// Allocate the input and output vector
	int *h_input_array = (int *)malloc(size);
	int *h_gpu_results = (int *)malloc(size);
	int *h_host_results = (int *)malloc(size);

	// Verify that allocations succeeded
	if(h_input_array == NULL || h_host_results == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Initialise the host input and output vectors
	for(int i = 0; i < num_elements; i++){
		h_input_array[i] = rand()%10;
	}
	h_host_results[0] = 0;
	for(int i = 1; i < num_elements; i++){
		h_host_results[i] = h_host_results[i-1] + h_input_array[i-1];
	}

	// Check host vectors are as expected
	printf("%d %d %d %d\n", h_input_array[0], h_input_array[1], h_input_array[2], h_input_array[num_elements-1]);
	printf("%d %d %d %d\n", h_host_results[0], h_host_results[1], h_host_results[2], h_host_results[num_elements-1]);

	// Initialise GPU arrays
	int *d_input_array = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_input_array, size), "Failed to allocate d_input_array");
	int *d_gpu_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size), "Failed to allocate d_gpu_results");

	// Copy the host input vector to the device memory
	CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice), "Failed to copy input vector from host to device");


	// *************************** BSCAN **********************************
	cudaEventRecord(start, 0);
	block<<<1, BLOCK_SIZE>>>(d_gpu_results, d_input_array, num_elements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_ERROR(cudaGetLastError(), "Failed to launch bscan kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_bscan, start, stop), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");

	printf("%d %d %d %d\n", h_gpu_results[0], h_gpu_results[1], h_gpu_results[2], h_gpu_results[num_elements-1]);

	compare_results(h_host_results, h_gpu_results, num_elements);

	printf("block: %.5fms", time_bscan);

	// *************************** BSCAN BCAO *****************************



	// *************************** FSCAN **********************************



	// *************************** FSCAN BCAO******************************




	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
	return 0;
}
