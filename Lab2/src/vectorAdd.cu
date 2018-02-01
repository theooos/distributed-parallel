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

int const numElements = 1000000;
int const block_size = 1024;
// Note this pattern, based on integer division, for rounding up
int grid_size = 1 + ((numElements - 1) / block_size);


__host__ void
host_bscan(const float *A, float *B, int numElements)
{
	int sum = 0;
    int i;
    for ( i = 0; i < numElements ; i ++){
    	sum += A[i];
    	B[i] = sum;
    }
}

__global__ void
single_thread_bscan(const float *A, float *B, int numElements)
{
	int sum = 0;
	int i;
	for ( i = 0; i < numElements ; i ++){
		sum += A[i];
		B[i] = sum;
	}
}

__global__ void
hsh_nsm_bscan(const float *A, float *B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}

__global__ void
blelloch_nsm_bscan(const float *A, float *B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}

__global__ void
hsh_bscan(const float *A, float *B, int numElements)
{
	__shared__ float XY[block_size*2];
	int r_buf = 0; int w_buf = block_size;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numElements){
		XY[w_buf + threadIdx.x] = A[i];
	}

	for(uint s=1; s < block_size; s*= 2){
		__syncthreads();
        w_buf = block_size - w_buf; r_buf = block_size - r_buf;
		if(threadIdx.x >= s)
			XY[w_buf + threadIdx.x] = XY[r_buf + threadIdx.x - s] + XY[r_buf + threadIdx.x];
		else
			XY[w_buf + threadIdx.x] = XY[r_buf + threadIdx.x];
    }

	if(i < numElements)
		B[i] = 3.0;
}

__global__ void
blelloch_bscan(const float *A, float *B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}

__global__ void
blelloch_dblock_bscan(const float *A, float *B, int numElements)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	B[i] = i+1;
}


static void compare_results(const float *vector1, const float *vector2, int numElements)
{
	for (int i = 0; i < numElements; ++i)
	{
		if (vector1[i] != vector2[i])
		{
			fprintf(stderr, "Result verification failed at element %d!  %0.5f : %0.5f\n", i, vector1[i], vector2[i]);
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
    double h_msecs ;

    // Create Device timer event objects
    cudaEvent_t start , stop ;
    float d_msecs1, d_msecs2, d_msecs3, d_msecs4, d_msecs5, d_msecs6;
    cudaEventCreate (&start);
    cudaEventCreate (&stop) ;


    // Print the vector length to be used, and compute its size
    size_t size = numElements * sizeof(float);

    // ******************************* HOST *******************************

    // Allocate the input and output vector
    float *h_A = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_SCAN = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_SCAN == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialise the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = 1.0f;
    }


    // ** Execute the vector addition on the Host and time it: **
    sdkStartTimer (& timer );
    host_bscan(h_A, h_SCAN, numElements);
    sdkStopTimer (& timer );
    h_msecs = sdkGetTimerValue (& timer );
    printf("host_bscan: %.5fms\n", numElements, h_msecs);


    // ******************************* GPU SINGLE *******************************

    // Allocate the device input vector A, B and C
    float *d_A = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_A, size), "Failed to allocate device vector A");
    float *d_Scan = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_Scan, size), "Failed to allocate device output vector");

    // Copy the host input vector A in host memory to the device input vector in device memory
    CUDA_ERROR(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "Failed to copy vector A from host to device");

    // Test
    cudaEventRecord( start, 0 );
    single_thread_bscan<<<1, 1>>>(d_A, d_Scan, numElements);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // wait for device to finish
    cudaDeviceSynchronize();

    CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
    CUDA_ERROR(cudaEventElapsedTime(&d_msecs1, start, stop), "Failed to get elapsed time");

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector d_Scan from device to host");
    compare_results(h_SCAN, h_C, numElements);

    printf("single_thread_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs1, h_msecs/d_msecs1);


    // ******************************* HSH-NSM-BSCAN ******************************* TODO Work
    cudaEventRecord( start, 0 );
    hsh_nsm_bscan<<<grid_size, block_size>>>(d_A, d_Scan, numElements);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaDeviceSynchronize();

    CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
    CUDA_ERROR(cudaEventElapsedTime( &d_msecs2, start, stop ), "Failed to get elapsed time");
    CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
    compare_results(h_SCAN, h_C, numElements);

    printf("hsh_nsm_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs2, h_msecs/d_msecs2);


    // ******************************* BLELLOCH-NSM-BSCAN ******************************* TODO Remove copy to/from host and where says XY just use x (leave out if struggling)
	cudaEventRecord( start, 0 );
	blelloch_nsm_bscan<<<grid_size, block_size>>>(d_A, d_Scan, numElements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &d_msecs3, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_SCAN, h_C, numElements);

	printf("blelloch_nsm_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs3, h_msecs/d_msecs3);


	// ******************************* HSH-BSCAN ******************************* TODO Copy
	cudaEventRecord( start, 0 );
	hsh_bscan<<<grid_size, block_size>>>(d_A, d_Scan, numElements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &d_msecs4, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_SCAN, h_C, numElements);

	printf("hsh_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs4, h_msecs/d_msecs4);


	// ******************************* BLELLOCH-BSCAN ******************************* TODO Copy
	cudaEventRecord( start, 0 );
	blelloch_bscan<<<grid_size, block_size>>>(d_A, d_Scan, numElements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &d_msecs5, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_SCAN, h_C, numElements);

	printf("blelloch_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs5, h_msecs/d_msecs5);


	// ******************************* BLELLOCH-DBLOCK-BSCAN ******************************* TODO Work
	cudaEventRecord( start, 0 );
	blelloch_dblock_bscan<<<grid_size, block_size>>>(d_A, d_Scan, numElements);
	cudaEventRecord( stop, 0 );
	cudaEventSynchronize( stop );
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
	CUDA_ERROR(cudaEventElapsedTime( &d_msecs6, start, stop ), "Failed to get elapsed time");
	CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");
	compare_results(h_SCAN, h_C, numElements);

	printf("blelloch_dblock_bscan: %.5fms, speedup: %.5f\n", numElements, d_msecs6, h_msecs/d_msecs6);



    // ******************************* Cleanup *******************************

    // Free device global memory
    err = cudaFree(d_A);
    CUDA_ERROR(err, "Failed to free device vector A");
    err = cudaFree(d_Scan);
    CUDA_ERROR(err, "Failed to free device vector Scan");

    // Free host memory
    free(h_A);
    free(h_C);

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

