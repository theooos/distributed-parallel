
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


__global__ void
extract_final_sums(const float *A, float *B, int num_elements, int stride){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < num_elements/stride){
		B[i] = A[(i+1)*stride-1];
	}
}

__global__ void
add_extracts(const float *extracts_array, float *pre_sum_array, int num_elements){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (1023 < i < num_elements){
		pre_sum_array[i] = pre_sum_array[i] + extracts_array[i/BLOCK_SIZE-1];
	}
}

__host__ void
host(const float *A, float *B, int num_elements)
{
	int sum = 0;
    int i;
    for ( i = 0; i < num_elements ; i ++){
    	sum += A[i];
    	B[i] = sum;
    }
}

__global__ void
single_thread(const float *A, float *B, int num_elements)
{
	int sum = 0;
	int i;
	for ( i = 0; i < num_elements ; i ++){
		sum += A[i];
		B[i] = sum;
	}
}

__global__ void
hsh_nsm(const float *A, float *B, int num_elements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}

__global__ void
blelloch_nsm(const float *A, float *B, int num_elements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}

__global__ void
hsh(const float *A, float *B, int num_elements)
{
	__shared__ float XY[BLOCK_SIZE*2];
	int r_buf = 0; int w_buf = BLOCK_SIZE;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < num_elements){
		XY[w_buf + threadIdx.x] = A[i];
	}

	for(uint s=1; s < BLOCK_SIZE; s*= 2){
		__syncthreads();
        w_buf = BLOCK_SIZE - w_buf; r_buf = BLOCK_SIZE - r_buf;
		if(threadIdx.x >= s)
			XY[w_buf + threadIdx.x] = XY[r_buf + threadIdx.x - s] + XY[r_buf + threadIdx.x];
		else
			XY[w_buf + threadIdx.x] = XY[r_buf + threadIdx.x];
	}

	// Copy to global memory
	if (i < num_elements) B[i] = XY[w_buf + threadIdx.x];

}

__global__ void blelloch(float *A, float *y, int len) {
    __shared__ float XY[BLOCK_SIZE];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < len) XY[threadIdx.x] = A[i];

    // Reduction
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < blockDim.x) XY[index] += XY[index - stride];
    }

    // Distribution
    for(unsigned int stride = BLOCK_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index + stride < BLOCK_SIZE) XY[index + stride] += XY[index];
    }
    __syncthreads();

    // Copy to global memory
    if(i < len) y[i] = XY[threadIdx.x];
}

__global__ void
blelloch_dblock(const float *A, float *B, int num_elements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}


static void compare_results(const float *vector1, const float *vector2, int num_elements)
{
	for (int i = 0; i < num_elements; ++i){
		if (fabs(vector1[i] - vector2[i]) > 1e-5f){
			fprintf(stderr, "Result verification failed at element %d!  h%0.5f : d%0.5f\n", i, vector1[i], vector2[i]);
			exit (EXIT_FAILURE);
		}
	}
}



int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Create Host stopwatch timer
    StopWatchInterface * timer = NULL ;
    sdkCreateTimer (& timer );
    double time_host ;

    // Create Device timer event objects
    cudaEvent_t start , stop ;
    float time_single_gpu, time_hsh_nsm, time_blel_nsm, time_hsh, time_blel, time_blel_dblock;
    cudaEventCreate (&start);
    cudaEventCreate (&stop) ;


    uint num_elements = 1000000;
    size_t size = num_elements * sizeof(float);
    int grid_size = 1 + (num_elements - 1) / BLOCK_SIZE;

    // ******************************* HOST *******************************

    // Allocate the input and output vector
    float *h_input_array = (float *)malloc(size);
    float *h_gpu_results = (float *)malloc(size);
    float *h_host_results = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_input_array == NULL || h_host_results == NULL){
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialise the host input vectors
    for (int i = 0; i < num_elements; ++i){
        h_input_array[i] = 1.0f;
    }

    // ** Execute the vector addition on the Host and time it: **
    sdkStartTimer (& timer );
    host(h_input_array, h_host_results, num_elements);
    sdkStopTimer (& timer );
    time_host = sdkGetTimerValue (& timer );
    printf("host: %.5fms\n", num_elements, time_host);


    // ******************************* GPU SINGLE *******************************
    float *d_input_array = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_input_array, size), "Failed to allocate d_input_array");
    float *d_gpu_results = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size), "Failed to allocate d_gpu_results");

    // Copy the host input vector A in host memory to the device input vector in device memory
    CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice), "Failed to copy vector A from host to device");

    // Test
    cudaEventRecord( start, 0 );
    single_thread<<<1, 1>>>(d_input_array, d_gpu_results, num_elements);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // wait for device to finish
    cudaDeviceSynchronize();

    CUDA_ERROR(cudaGetLastError(), "Failed to launch gpu single kernel");
    CUDA_ERROR(cudaEventElapsedTime(&time_single_gpu, start, stop), "Failed to get elapsed time");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector d_gpu_results from device to host");
    compare_results(h_host_results, h_gpu_results, num_elements);

    printf("single_thread: %.5fms, speedup: %.5f\n", num_elements, time_single_gpu, time_host/time_single_gpu);


    // ******************************* LARGE VECTOR LAYER VARIABLE SETUP **********************

    int extract_length = num_elements/BLOCK_SIZE;
	size_t extract_size = extract_length * sizeof(float);
	int extract_grid_size = 1 + (extract_length - 1) / BLOCK_SIZE;

	float *d_last_elems = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_last_elems, extract_size), "Failed to create array for last elements of each block");
	float *d_last_elems_scanned = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_last_elems_scanned, extract_size), "Failed to create array for last summed elements");


//    // ******************************* HSH-NSM-BSCAN ******************************* TODO Work
//    cudaEventRecord( start, 0 );
//    hsh_nsm<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
//    cudaEventRecord( stop, 0 );
//    cudaEventSynchronize( stop );
//    cudaDeviceSynchronize();
//
//    CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
//    CUDA_ERROR(cudaEventElapsedTime( &time_hsh_nsm, start, stop ), "Failed to get elapsed time");
//    CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
//    compare_results(h_host_results, h_gpu_results, num_elements);
//
//    printf("hsh_nsm: %.5fms, speedup: %.5f\n", num_elements, time_hsh_nsm, time_host/time_hsh_nsm);
//
//
//    // ******************************* BLELLOCH-NSM-BSCAN ******************************* TODO Remove copy to/from host and where says XY just use x (leave out if struggling)
//	cudaEventRecord( start, 0 );
//	blelloch_nsm<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaDeviceSynchronize();
//
//	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
//	CUDA_ERROR(cudaEventElapsedTime( &time_blel_nsm, start, stop ), "Failed to get elapsed time");
//	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
//	compare_results(h_host_results, h_gpu_results, num_elements);
//
//	printf("blelloch_nsm: %.5fms, speedup: %.5f\n", num_elements, time_blel_nsm, time_host/time_blel_nsm);


	// ******************************* HSH-BSCAN *******************************

	cudaEventRecord( start, 0 );
	hsh<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
	extract_final_sums<<<extract_grid_size, BLOCK_SIZE>>>(d_gpu_results, d_last_elems, num_elements, BLOCK_SIZE);
	hsh<<<grid_size, BLOCK_SIZE>>>(d_last_elems, d_last_elems_scanned, extract_length);
	add_extracts<<<grid_size, BLOCK_SIZE>>>(d_last_elems_scanned, d_gpu_results, num_elements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch hsh kernel");
	CUDA_ERROR(cudaEventElapsedTime(&time_hsh, start, stop), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");
	compare_results(h_host_results, h_gpu_results, num_elements);

	printf("hsh: %.5fms, speedup: %.5f\n", num_elements, time_hsh, time_host/time_hsh);


	// ******************************* BLELLOCH-BSCAN *******************************
	cudaEventRecord( start, 0 );
	blelloch<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
	extract_final_sums<<<extract_grid_size, BLOCK_SIZE>>>(d_gpu_results, d_last_elems, num_elements, BLOCK_SIZE);
	blelloch<<<grid_size, BLOCK_SIZE>>>(d_last_elems, d_last_elems_scanned, extract_length);
	add_extracts<<<grid_size, BLOCK_SIZE>>>(d_last_elems_scanned, d_gpu_results, num_elements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &time_blel, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_host_results, h_gpu_results, num_elements);

	printf("blelloch: %.5fms, speedup: %.5f\n", num_elements, time_blel, time_host/time_blel);


//	// ******************************* BLELLOCH-DBLOCK-BSCAN ******************************* TODO Work
//	cudaEventRecord( start, 0 );
//	blelloch_dblock<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaDeviceSynchronize();
//
//	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
//	CUDA_ERROR(cudaEventElapsedTime( &time_blel_dblock, start, stop ), "Failed to get elapsed time");
//	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
//	compare_results(h_host_results, h_gpu_results, num_elements);
//
//	printf("blelloch_dblock: %.5fms, speedup: %.5f\n", num_elements, time_blel_dblock, time_host/time_blel_dblock);



    // ******************************* Cleanup *******************************

    // Free device global memory
    CUDA_ERROR(cudaFree(d_input_array), "Failed to free device vector A");
    CUDA_ERROR(cudaFree(d_gpu_results), "Failed to free device vector Scan");
    CUDA_ERROR(cudaFree(d_last_elems), "Failed to free device vector Scan");
    CUDA_ERROR(cudaFree(d_last_elems_scanned), "Failed to free device vector Scan");

    // Free host memory
    free(h_input_array);
    free(h_gpu_results);
    free(h_host_results);

    // Clean up the Host timer
    sdkDeleteTimer (& timer );

    // Clean up the Device timer event objects
    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );

    // Reset the device and exit
    err = cudaDeviceReset();
    CUDA_ERROR(err, "Failed to reset the device");
    return 0;
}

