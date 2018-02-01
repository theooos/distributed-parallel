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

}


static void compare_results(const float *vector1, const float *vector2, int numElements)
{
	for (int i = 0; i < numElements; ++i)
	{
		if (vector1[i] != vector2[i])
		{
			fprintf(stderr, "Result verification failed at element %d!\n", i);
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
    int numElements = 1000000;
    size_t size = numElements * sizeof(float);
    printf("[Scan of %d elements]\n", numElements);

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
    printf("Executed vector add of %d elements on the Host in = %.5fmSecs\n", numElements, h_msecs);


    // ******************************* GPU SINGLE *******************************

    // Allocate the device input vector A, B and C
    float *d_A = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_A, size), "Failed to allocate device vector A");
    float *d_Scan = NULL;
    CUDA_ERROR(cudaMalloc((void **)&d_Scan, size), "Failed to allocate device output vector");

    // Copy the host input vector A in host memory to the device input vector in device memory
    printf("Copy input data from the host memory to the CUDA device\n");
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

    printf("Executed scan on the Device in a SINGLE THREAD in = %.5fmSecs\n", numElements, d_msecs1);

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost);
    CUDA_ERROR(err, "Failed to copy vector d_Scan from device to host");

    // Verify that the result vector is correct
    compare_results(h_SCAN, h_C, numElements);
    printf("Test PASSED\n");


    // ******************************* HSH-NSM-BSCAN *******************************

    int threadsPerBlock = 1024;
    // Note this pattern, based on integer division, for rounding up
    int blocksPerGrid = 1 + ((numElements - 1) / threadsPerBlock);

    printf("Launching the CUDA kernel with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // Test
    cudaEventRecord( start, 0 );
    hsh_nsm_bscan<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_Scan, numElements);
    cudaEventRecord( stop, 0 );
    cudaEventSynchronize( stop );
    cudaDeviceSynchronize();

    CUDA_ERROR(cudaGetLastError(), "Failed to launch vectorAdd kernel");
    CUDA_ERROR(cudaEventElapsedTime( &d_msecs2, start, stop ), "Failed to get elapsed time");

    printf("Executed vector add of %d elements on the Device in %d blocks of %d threads in = %.5fmSecs\nSpeedup: %.5fms",
    		numElements, blocksPerGrid, threadsPerBlock, d_msecs2, h_msecs/d_msecs2);

    CUDA_ERROR(cudaMemcpy(h_C, d_Scan, size, cudaMemcpyDeviceToHost), "Failed to copy vector C from device to host");

    compare_results(h_SCAN, h_C, numElements);
    printf("Test PASSED\n");



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

    printf("Done\n");
    return 0;
}

