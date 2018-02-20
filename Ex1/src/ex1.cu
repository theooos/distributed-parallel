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

int main(void)
{
	cudaError_t err = cudaSuccess;

	// ***************** Initial variable construction ******************
	cudaEvent_t start, stop;
	float time_bscan, time_bscan_bcao, time_fscan, time_fscan_bcao;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	uint num_elements = 10000000;
	size_t size = num_elements * sizeof(float);
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

	printf("%d %d %d %d\n", h_input_array[0], h_input_array[1], h_input_array[2], h_input_array[num_elements-1]);
	printf("%d %d %d %d\n", h_host_results[0], h_host_results[1], h_host_results[2], h_host_results[num_elements-1]);

	// *************************** BSCAN **********************************



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
