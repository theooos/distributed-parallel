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

#define NUM_BANKS 32
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

__global__ void
block_bcao(int *g_odata, int *g_idata, int n)
{
	__shared__ int temp[BLOCK_SIZE*2];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	int ai = thid;
	int bi = thid + (n/2);
	int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
	int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
	temp[ai + bankOffsetA] = g_idata[ai];
	temp[bi + bankOffsetB] = g_idata[bi];
	for (int d = n>>1; d > 0; d >>= 1)  // build sum in place up the tree
	{
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			temp[bi] += temp[ai];
		}
		offset *= 2;
	}
	if (thid==0) { temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;}
	for (int d = 1; d < n; d *= 2) // traverse down tree & build scan
	{
		offset >>= 1;
		__syncthreads();
		if (thid < d)
		{
			int ai = offset*(2*thid+1)-1;
			int bi = offset*(2*thid+2)-1;
			ai += CONFLICT_FREE_OFFSET(ai);
			bi += CONFLICT_FREE_OFFSET(bi);
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}
	__syncthreads();
	g_odata[ai] = temp[ai + bankOffsetA];
	g_odata[bi] = temp[bi + bankOffsetB];
}


__global__ void
apply_block_ends(const int *block_ends, int *input, int in_length, int stride){
	int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int input_i = thid + stride;
	if(input_i < in_length){
		input[input_i] += block_ends[input_i/stride];
	}
}

__global__ void
full(int *g_odata, int *g_idata, int n)
{
	__shared__ int temp[2048];  // allocated on invocation
	int thid = threadIdx.x;
	int offset = 1;
	temp[2*thid] = g_idata[2*(thid + blockDim.x * blockIdx.x)];     // load input into shared memory
	temp[2*thid+1] = g_idata[2*(thid + blockDim.x * blockIdx.x) + 1];
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
	g_odata[2 * (thid + blockDim.x * blockIdx.x)] = temp[2*thid]; // write results to device memory
	g_odata[2 * (thid + blockDim.x * blockIdx.x) + 1] = temp[2*thid+1];
}

__global__ void
fuller(int *g_idata, int *g_odata, int stride, int in_length, int *block_ends)
{
	__shared__ int temp[2048];  // allocated on invocation
	int thid = threadIdx.x;
	int real_index = thid + blockDim.x * blockIdx.x;
	if(real_index < in_length/2){
		int offset = 1;
		temp[2*thid] = g_idata[2*real_index];     // load input into shared memory
		temp[2*thid+1] = g_idata[2*real_index+1];
		for (int d = stride>>1; d > 0; d >>= 1)  // build sum in place up the tree
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
		if (thid == 0) {
			block_ends[real_index/2048] = temp[stride - 1];
			temp[stride - 1] = 0;
		}
		for (int d = 1; d < stride; d *= 2) // traverse down tree & build scan
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
		g_odata[2*real_index] = temp[2*thid]; // write results to device memory
		g_odata[2*real_index+1] = temp[2*thid+1];
	}
}

static void compare_results(const int *host, const int *device, int num_elements)
{
	for (int i = 0; i < num_elements; ++i){
		if (fabs(host[i] - device[i]) > 1e-5f){
			fprintf(stderr, "Result verification failed at element %d!  h%d : d%d\n", i, host[i], device[i]);
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

	uint total_elements = 2048;
	size_t size = total_elements * sizeof(int);

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
	for(int i = 0; i < total_elements; i++){
		h_input_array[i] = rand()%10;
	}
	h_host_results[0] = 0;
	for(int i = 1; i < total_elements; i++){
		h_host_results[i] = h_host_results[i-1] + h_input_array[i-1];
	}

	// Check host vectors are as expected
	printf("%d %d %d %d\n", h_input_array[0], h_input_array[1], h_input_array[2], h_input_array[total_elements-1]);
	printf("%d %d %d %d\n", h_host_results[0], h_host_results[1], h_host_results[2], h_host_results[total_elements-1]);

	// Initialise GPU arrays
	int *d_input_array = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_input_array, size), "Failed to allocate d_input_array");
	int *d_gpu_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size), "Failed to allocate d_gpu_results");

	// Copy the host input vector to the device memory
	CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice), "Failed to copy input vector from host to device");


	// *************************** BSCAN **********************************
	cudaEventRecord(start, 0);
	block<<<1, BLOCK_SIZE>>>(d_gpu_results, d_input_array, total_elements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_ERROR(cudaGetLastError(), "Failed to launch bscan kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_bscan, start, stop), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");

	printf("%d %d %d %d\n", h_gpu_results[0], h_gpu_results[1], h_gpu_results[2], h_gpu_results[total_elements-1]);

	compare_results(h_host_results, h_gpu_results, total_elements);

	printf("block: %.5fms\n", time_bscan);
	cudaDeviceSynchronize();

	// *************************** BSCAN BCAO *****************************
	cudaEventRecord(start, 0);
	block_bcao<<<1, BLOCK_SIZE>>>(d_gpu_results, d_input_array, total_elements);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_ERROR(cudaGetLastError(), "Failed to launch bscan_bcao kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_bscan_bcao, start, stop), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");

	printf("%d %d %d %d\n", h_gpu_results[0], h_gpu_results[1], h_gpu_results[2], h_gpu_results[total_elements-1]);

	compare_results(h_host_results, h_gpu_results, total_elements);

	printf("block_bcao: %.5fms\n", time_bscan_bcao);


	// *************************** FULL SETUP *****************************
	printf("\nResetting for 10,000,000 sized array.\n");

	// Resize arrays.
	total_elements = 10000000;
	size = total_elements * sizeof(int);
	int stride = 2048;
	int total_padded = total_elements + (stride - total_elements % stride);
	int padded_size = total_padded * sizeof(int);

	// Reallocate space.
	h_input_array = (int *)malloc(size);
	h_gpu_results = (int *)malloc(size);
	h_host_results = (int *)malloc(size);

	if(h_input_array == NULL || h_host_results == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	// Reinitialise arrays.
	for(int i = 0; i < total_elements; i++){
		h_input_array[i] = rand()%10;
	}
	h_host_results[0] = 0;
	for(int i = 1; i < total_elements; i++){
		h_host_results[i] = h_host_results[i-1] + h_input_array[i-1];
	}

	// Check host vectors are as expected
	printf("%d %d %d %d\n", h_input_array[0], h_input_array[1], h_input_array[2], h_input_array[total_elements-1]);
	printf("%d %d %d %d\n", h_host_results[0], h_host_results[1], h_host_results[2], h_host_results[total_elements-1]);

	// Initialise GPU arrays
	cudaFree(d_input_array);
	cudaFree(d_gpu_results);
	d_input_array = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_input_array, padded_size), "Failed to allocate d_input_array");
	d_gpu_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, padded_size), "Failed to allocate d_gpu_results");

	// Copy the host input vector to the device memory
	CUDA_ERROR(cudaMemset(d_input_array, 0, padded_size), "Failed to create padded array in GPU");
	CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice), "Failed to copy input vector from host to device");

	// Create the secondary and tertiary sum arrays
	int *d_sum1 = NULL;
	int *d_sum1_scanned = NULL;
	int sum1_length = total_elements/(BLOCK_SIZE*2);
	size_t sum1_size = sum1_length * sizeof(int);

	int *d_sum2 = NULL;
	int *d_sum2_scanned = NULL;
	int sum2_length = sum1_length/(BLOCK_SIZE*2);
	size_t sum2_size = sum2_length * sizeof(int);

	CUDA_ERROR(cudaMalloc((void **)&d_sum1, sum1_size), "Failed to allocate d_sum1");
	CUDA_ERROR(cudaMalloc((void **)&d_sum1_scanned, sum1_size), "Failed to allocate d_sum1_scanned");
	CUDA_ERROR(cudaMalloc((void **)&d_sum2, sum2_size), "Failed to allocate d_sum2");
	CUDA_ERROR(cudaMalloc((void **)&d_sum2_scanned, sum2_size), "Failed to allocate d_sum2_scanned");


	// *************************** FSCAN **********************************
	cudaEventRecord(start, 0);
	fuller<<<4882, 1024>>>(d_input_array, d_gpu_results, stride, total_elements, d_sum1);
	cudaDeviceSynchronize();
	fuller<<<3, 1024>>>(d_sum1, d_sum1_scanned, stride, sum1_length, d_sum2);
	cudaDeviceSynchronize();
	fuller<<<1, 1>>>(d_sum2, d_sum2_scanned, stride, sum2_length, d_sum1); //redundantly using d_sum1 so no null pointer
	cudaDeviceSynchronize();
	apply_block_ends<<<3, 1024>>>(d_sum2_scanned, d_sum1_scanned, sum1_length, stride);
	cudaDeviceSynchronize();
	apply_block_ends<<<9764, 1024>>>(d_sum1_scanned, d_gpu_results, total_elements, stride);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch hsh kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_fscan, start, stop), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");
	compare_results(h_host_results, h_gpu_results, total_elements);

	printf("full: %dms", time_fscan);


	// *************************** FSCAN BCAO******************************




	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
	return 0;
}
