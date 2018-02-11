/**
 * Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 *
 * Slightly modified to provide timing support
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
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
		B[i] = A[i*stride];
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
	if (i < num_elements) B[i] = XY[w_buf + threadIdx.x];

}

__global__ void
blelloch(const float *A, float *B, int num_elements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
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
    if (h_input_array == NULL || h_host_results == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialise the host input vectors
    for (int i = 0; i < num_elements; ++i)
    {
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
    CUDA_ERROR(cudaMalloc((void **)&d_input_array, size), "Failed to allocate device vector A");
    float *d_gpu_results = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_gpu_results, size), "Failed to allocate device output vector");

    // Copy the host input vector A in host memory to the device input vector in device memory
    CUDA_ERROR(cudaMemcpy(d_input_array, h_input_array, size, cudaMemcpyHostToDevice), "Failed to copy vector A from host to device");

    // Test
    cudaEventRecord( start, 0 );
    single_thread<<<1, 1>>>(d_input_array, d_gpu_results, num_elements);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // wait for device to finish
    cudaDeviceSynchronize();

    CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
    CUDA_ERROR(cudaEventElapsedTime(&time_single_gpu, start, stop), "Failed to get elapsed time");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector d_gpu_results from device to host");
    compare_results(h_host_results, h_gpu_results, num_elements);

    printf("single_thread: %.5fms, speedup: %.5f\n", num_elements, time_single_gpu, time_host/time_single_gpu);


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
//    printf("hsh_nsm: %.5fms, speedup: %.5f\n", num_elements, time_hsh_nsm, h_msecs/time_hsh_nsm);
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
//	printf("blelloch_nsm: %.5fms, speedup: %.5f\n", num_elements, time_blel_nsm, h_msecs/time_blel_nsm);


	// ******************************* HSH-BSCAN ******************************* TODO Copy
//    cudaEventRecord( start, 0 );
//	hsh<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaDeviceSynchronize();
//
//	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
//	CUDA_ERROR(cudaEventElapsedTime( &time_hsh, start, stop ), "Failed to get elapsed time");
//	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
//	compare_results(h_host_results, h_gpu_results, num_elements);
//
//	printf("hsh: %.5fms, speedup: %.5f\n", num_elements, time_hsh, h_msecs/time_hsh);

	int extract_length = num_elements/BLOCK_SIZE;
	size_t extract_size = extract_length * sizeof(float);
	int extract_grid_size = 1 + (extract_length - 1) / BLOCK_SIZE;

	float *d_last_elems = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_last_elems, extract_size), "Failed to create array for last elements of each block");
	float *d_last_elems_scanned = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_last_elems_scanned, extract_size), "Failed to create array for last elements of each block");

	cudaEventRecord( start, 0 );
	hsh<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
	extract_final_sums<<<extract_grid_size, BLOCK_SIZE>>>(d_gpu_results, d_last_elems, num_elements, BLOCK_SIZE);
	hsh<<<grid_size, BLOCK_SIZE>>>(d_last_elems, d_last_elems_scanned, extract_length);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	float *h_last_elems = (float *) malloc(extract_size);
	float *h_last_elems_scanned = (float *) malloc(extract_size);

	CUDA_ERROR(cudaMemcpy(h_last_elems, d_last_elems, extract_size, cudaMemcpyDeviceToHost), "LastElems copy failed.");
	CUDA_ERROR(cudaMemcpy(h_last_elems_scanned, d_last_elems_scanned, extract_size, cudaMemcpyDeviceToHost), "LastElemsScanned copy failed.");

	printf("%d\n", sizeof(h_last_elems));
	printf("%d\n", sizeof(*h_last_elems_scanned));
	printf("%d\n", extract_size);
	printf("%d\n", extract_length);

	for (int i=0;i < extract_length; i++) {
	    printf("%f,",h_last_elems_scanned[i]);
	}
	printf("\n");

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &time_hsh, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_host_results, h_gpu_results, num_elements);

//	printf("hsh: %.5fms, speedup: %.5f\n", num_elements, time_hsh, h_msecs/time_hsh);


//	// ******************************* BLELLOCH-BSCAN ******************************* TODO Copy
//	cudaEventRecord( start, 0 );
//	blelloch<<<grid_size, BLOCK_SIZE>>>(d_input_array, d_gpu_results, num_elements);
//	cudaEventRecord( stop, 0 );
//	cudaEventSynchronize( stop );
//	cudaDeviceSynchronize();
//
//	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
//	CUDA_ERROR(cudaEventElapsedTime( &time_blel, start, stop ), "Failed to get elapsed time");
//	CUDA_ERROR(cudaMemcpy(h_gpu_results, d_gpu_results, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
//	compare_results(h_host_results, h_gpu_results, num_elements);
//
//	printf("blelloch: %.5fms, speedup: %.5f\n", num_elements, time_blel, h_msecs/time_blel);
//
//
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
//	printf("blelloch_dblock: %.5fms, speedup: %.5f\n", num_elements, time_blel_dblock, h_msecs/time_blel_dblock);



    // ******************************* Cleanup *******************************

    // Free device global memory
    err = cudaFree(d_input_array);
    CUDA_ERROR(err, "Failed to free device vector A");
    err = cudaFree(d_gpu_results);
    CUDA_ERROR(err, "Failed to free device vector Scan");

    // Free host memory
    free(h_input_array);
    free(h_gpu_results);

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

