/*
 * Theodore Gregory - 1453831
 *
 * Tasks:
 * Block scan
 * Full scan for large vectors
 * Bank Conflict Avoidance Optimisation
 *
 * Timings:
 * Block scan without BCAO - 0.07578ms
 * Block scan with BCAO - 0.06144ms
 * Full scan without BCAO - 11.38774ms
 * Full scan with BCAO - 11.56349ms
 *
 * Hardware:
 * CPU - Intel - Core i5-6600 3.3GHz Quad-Core Processor
 * GPU - Zotac - GeForce GTX 1080 8GB AMP! Edition Video Card
 *
 * Implementation details:
 * I spent over 30 total hours on this.
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
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define BLOCK_SIZE 1024

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> LOG_NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

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
full(int *g_odata, int *g_idata, int n, int stride, int *sums)
{
	__shared__ int temp[BLOCK_SIZE*2];  // allocated on invocation
	int thid = threadIdx.x;
	int real_index = thid + blockDim.x * blockIdx.x;
	if(real_index < n){
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
			sums[blockIdx.x] = temp[stride-1];
			temp[stride-1] = 0;
		} // clear the last element
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

__global__ void
full_bcao(int *g_odata, int *g_idata, int n, int stride, int *sums)
{
	__shared__ int temp[BLOCK_SIZE*2];  // allocated on invocation
		int thid = threadIdx.x;
		int real_index = thid + blockDim.x * blockIdx.x;
		if(real_index < n){
			int offset = 1;
			int ai = thid;
			int bi = thid + (stride/2);
			int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
			int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
			temp[ai + bankOffsetA] = g_idata[ai + blockIdx.x*stride];
			temp[bi + bankOffsetB] = g_idata[bi + blockIdx.x*stride];
			for (int d = stride>>1; d > 0; d >>= 1)  // build sum in place up the tree
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
			if (thid == 0) {
				sums[blockIdx.x] = temp[stride-1];
				temp[stride - 1 + CONFLICT_FREE_OFFSET(stride - 1)] = 0;
			} // clear the last element
			for (int d = 1; d < stride; d *= 2) // traverse down tree & build scan
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
			g_odata[ai + blockIdx.x * stride] = temp[ai + bankOffsetA];
			g_odata[bi + blockIdx.x * stride] = temp[bi + bankOffsetB];
		}
}

__global__ void
apply_block_ends(const int *block_ends, int *input, int in_length, int stride){
	int thid = blockDim.x * blockIdx.x + threadIdx.x;
	int input_i = thid + stride;
	if(input_i < in_length){
		input[input_i] += block_ends[input_i/stride];
	}
}

static int compare_results(const int *host, const int *device, int num_elements)
{
	for (int i = 0; i < num_elements; ++i){
		if (fabs(host[i] - device[i]) > 1e-5f){
			fprintf(stderr, "Result verification failed at element %d!  h%d : d%d\n", i, host[i], device[i]);
			return 1;
		}
	}
	return 0;
}

int block_and_block_bcao(){
	cudaError_t err = cudaSuccess;

	// ***************** Initial variable construction ******************
	cudaEvent_t start, stop;
	float time_bscan, time_bscan_bcao;
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

	compare_results(h_host_results, h_gpu_results, total_elements);

	printf("block_bcao: %.5fms\n", time_bscan_bcao);
	cudaDeviceSynchronize();

	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");

	return 0;
}

int full(void){
	cudaError_t err = cudaSuccess;

	cudaEvent_t start, stop;
	float time_fscan;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint total_elements = 10000000;
	size_t size = total_elements * sizeof(int);
	int stride = 2048;
	int total_padded = (total_elements % stride == 0) ? total_elements : total_elements + (stride - total_elements % stride);
	size_t size_padded = total_padded * sizeof(int);

	int *h_input_array = (int *) malloc(size_padded);
	int *h_gpu_results = (int *) malloc(size);
	int *h_host_results = (int *) malloc(size);

	if(h_input_array == NULL || h_host_results == NULL || h_gpu_results == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	srand(time(NULL));
	for(int i = 0; i < total_elements; i++){
		h_input_array[i] = rand()%10;
	}
	for(int i = total_elements; i < total_padded; i++){
		h_input_array[i] = 0;
	}
	h_host_results[0] = 0;
	for(int i = 1; i < total_elements; i++){
		h_host_results[i] = h_host_results[i-1] + h_input_array[i-1];
	}

	// Initialise GPU arrays
	int *d_input_array = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_input_array, size_padded), "Failed to allocate d_input_array");
	int *d_gpu_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size_padded), "Failed to allocate d_gpu_results");

	// Copy the host input vector to the device memory
	CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size_padded, cudaMemcpyHostToDevice), "Failed to copy input vector from host to device");

	// SCAN 1 VARS
	int *d_gpu_sums1 = NULL;
	int sums1_length = total_padded/stride;
	int sums1_length_padded = (sums1_length % 2048 == 0) ? sums1_length : sums1_length + (stride - sums1_length % stride);
	size_t sums1_size = sums1_length_padded * sizeof(int);
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums1, sums1_size), "Failed to allocate d_sum1");

	int scan1_grid = 1 + (total_padded-1)/stride;
	int scan1_block = stride/2;

	// SCAN 2 VARS
	int *d_gpu_sums2 = NULL;
	int sums2_length = sums1_length/stride;
	int sums2_length_padded = (sums2_length % 2048 == 0) ? sums2_length : sums2_length + (stride - sums2_length % stride);
	size_t sums2_size = sums2_length_padded * sizeof(int);
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums2, sums2_size), "Failed to allocate d_sum2");

	int scan2_grid = 1+ (sums1_length-1)/stride;
	int scan2_block = stride/2;

	int scan2_length = total_elements/stride;
	int scan2_padded = (scan2_length % stride == 0) ? scan2_length : scan2_length + (stride - scan2_length % stride);
	size_t size2_padded = scan2_padded * sizeof(int);

	int *d_scan2_results = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_scan2_results, size2_padded), "Failed to allocate d_sum2");

	// SCAN 3 VARS
	int *d_gpu_sums3 = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums3, 2048*sizeof(int)), "Failed to allocate d_gpu_sums3"); //Must be large enough to avoid memory access exception, this array is acutally redundant in the 3rd cycle.

	int scan3_grid = 1 + (sums2_length-1)/stride;
	int scan3_block = stride/2;

	int scan3_length = total_elements/stride;
	int scan3_padded = (scan3_length % stride == 0) ? scan3_length : scan3_length + (stride - scan3_length % stride);
	size_t size3_padded = scan3_padded * sizeof(int);

	int *d_scan3_results = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_scan3_results, size3_padded), "Failed to allocate d_sum3");

	// ADD 1 VARS
	int add1_grid = 1 + (sums1_length-1)/BLOCK_SIZE;
	int add1_block = stride/2;

	// ADD 2 VARS
	int add2_grid = 1 + (total_elements-1)/BLOCK_SIZE;
	int add2_block = BLOCK_SIZE;

	// DO IT!!
	cudaEventRecord(start, 0);
	full<<<scan1_grid, scan1_block>>>(d_gpu_results, d_input_array, total_padded, stride, d_gpu_sums1);
	full<<<scan2_grid, scan2_block>>>(d_scan2_results, d_gpu_sums1, sums1_length_padded, stride, d_gpu_sums2);
	full<<<scan3_grid, scan3_block>>>(d_scan3_results, d_gpu_sums2, sums2_length_padded, stride, d_gpu_sums3);
	apply_block_ends<<<add1_grid, add1_block>>>(d_scan3_results, d_scan2_results, scan2_length, stride);
	apply_block_ends<<<add2_grid, add2_block>>>(d_scan2_results, d_gpu_results, total_elements, stride);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_ERROR(cudaGetLastError(), "Failed to launch fscan kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_fscan, start, stop), "Failed to get elapsed time");

	// *********** Final test ********************
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy scan3 results to host.");
	cudaDeviceSynchronize();
	if (compare_results(h_host_results, h_gpu_results, 10000)){
		printf("Whole thing failed");
	} else {
		printf("full: %.5fms\n", time_fscan);
	}
	cudaDeviceSynchronize();

	// *********** CLEANUP ***********************
	cudaFree(d_input_array);
	cudaFree(d_gpu_results);
	cudaFree(d_gpu_sums1);

	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
	return 0;
}

int full_bcao(void){
	cudaError_t err = cudaSuccess;

	cudaEvent_t start, stop;
	float time_fscan;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint total_elements = 10000000;
	size_t size = total_elements * sizeof(int);
	int stride = 2048;
	int total_padded = (total_elements % stride == 0) ? total_elements : total_elements + (stride - total_elements % stride);
	size_t size_padded = total_padded * sizeof(int);

	int *h_input_array = (int *) malloc(size_padded);
	int *h_gpu_results = (int *) malloc(size);
	int *h_host_results = (int *) malloc(size);

	if(h_input_array == NULL || h_host_results == NULL || h_gpu_results == NULL){
		fprintf(stderr, "Failed to allocate host vectors!\n");
		exit(EXIT_FAILURE);
	}

	srand(time(NULL));
	for(int i = 0; i < total_elements; i++){
		h_input_array[i] = rand()%10;
	}
	for(int i = total_elements; i < total_padded; i++){
		h_input_array[i] = 0;
	}
	h_host_results[0] = 0;
	for(int i = 1; i < total_elements; i++){
		h_host_results[i] = h_host_results[i-1] + h_input_array[i-1];
	}

	// Initialise GPU arrays
	int *d_input_array = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_input_array, size_padded), "Failed to allocate d_input_array");
	int *d_gpu_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size_padded), "Failed to allocate d_gpu_results");

	// Copy the host input vector to the device memory
	CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size_padded, cudaMemcpyHostToDevice), "Failed to copy input vector from host to device");

	// SCAN 1 VARS
	int *d_gpu_sums1 = NULL;
	int sums1_length = total_padded/stride;
	int sums1_length_padded = (sums1_length % 2048 == 0) ? sums1_length : sums1_length + (stride - sums1_length % stride);
	size_t sums1_size = sums1_length_padded * sizeof(int);
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums1, sums1_size), "Failed to allocate d_sum1");

	int scan1_grid = 1 + (total_padded-1)/stride;
	int scan1_block = stride/2;

	// SCAN 2 VARS
	int *d_gpu_sums2 = NULL;
	int sums2_length = sums1_length/stride;
	int sums2_length_padded = (sums2_length % 2048 == 0) ? sums2_length : sums2_length + (stride - sums2_length % stride);
	size_t sums2_size = sums2_length_padded * sizeof(int);
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums2, sums2_size), "Failed to allocate d_sum2");

	int scan2_grid = 1+ (sums1_length-1)/stride;
	int scan2_block = stride/2;

	int scan2_length = total_elements/stride;
	int scan2_padded = (scan2_length % stride == 0) ? scan2_length : scan2_length + (stride - scan2_length % stride);
	size_t size2_padded = scan2_padded * sizeof(int);

	int *d_scan2_results = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_scan2_results, size2_padded), "Failed to allocate d_sum2");

	// SCAN 3 VARS
	int *d_gpu_sums3 = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_gpu_sums3, 2048*sizeof(int)), "Failed to allocate d_gpu_sums3"); //Must be large enough to avoid memory access exception, this array is acutally redundant in the 3rd cycle.

	int scan3_grid = 1 + (sums2_length-1)/stride;
	int scan3_block = stride/2;

	int scan3_length = total_elements/stride;
	int scan3_padded = (scan3_length % stride == 0) ? scan3_length : scan3_length + (stride - scan3_length % stride);
	size_t size3_padded = scan3_padded * sizeof(int);

	int *d_scan3_results = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_scan3_results, size3_padded), "Failed to allocate d_sum3");

	// ADD 1 VARS
	int add1_grid = 1 + (sums1_length-1)/BLOCK_SIZE;
	int add1_block = stride/2;

	// ADD 2 VARS
	int add2_grid = 1 + (total_elements-1)/BLOCK_SIZE;
	int add2_block = BLOCK_SIZE;

	// DO IT!!
	cudaEventRecord(start, 0);
	full_bcao<<<scan1_grid, scan1_block>>>(d_gpu_results, d_input_array, total_padded, stride, d_gpu_sums1);
	full_bcao<<<scan2_grid, scan2_block>>>(d_scan2_results, d_gpu_sums1, sums1_length_padded, stride, d_gpu_sums2);
	full_bcao<<<scan3_grid, scan3_block>>>(d_scan3_results, d_gpu_sums2, sums2_length_padded, stride, d_gpu_sums3);
	apply_block_ends<<<add1_grid, add1_block>>>(d_scan3_results, d_scan2_results, scan2_length, stride);
	apply_block_ends<<<add2_grid, add2_block>>>(d_scan2_results, d_gpu_results, total_elements, stride);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	CUDA_ERROR(cudaGetLastError(), "Failed to launch fscan_bcao kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_fscan, start, stop), "Failed to get elapsed time");

	// *********** Final test ********************
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy scan3 results to host.");
	cudaDeviceSynchronize();
	if (compare_results(h_host_results, h_gpu_results, 10000)){
		printf("Whole thing failed");
	} else {
		printf("full: %.5fms\n", time_fscan);
	}
	cudaDeviceSynchronize();

	// *********** CLEANUP ***********************
	cudaFree(d_input_array);
	cudaFree(d_gpu_results);
	cudaFree(d_gpu_sums1);

	// Clean up the Device timer event objects
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	// Reset the device and exit
	err = cudaDeviceReset();
	CUDA_ERROR(err, "Failed to reset the device");
	return 0;
}

int main(void){
	block_and_block_bcao();
	full();
	full_bcao();
}
