/*
 ============================================================================
 Name        : Lab3.cu
 Author      : Theo Gregory
 Version     :
 Copyright   : 
 Description : CUDA Perform a full HSH scan, and convert image to grayscale.
 ============================================================================
 */

#include <iostream>
#include <numeric>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include "lodepng.h"

#define CUDA_ERROR( err, msg ) { \
if (err != cudaSuccess) {\
    printf( "%s: %s in %s at line %d\n", msg, cudaGetErrorString( err ), __FILE__, __LINE__);\
    exit( EXIT_FAILURE );\
}\
}
#define BLOCK_SIZE 1024

// ********************** FULL SCAN *************************

/**
 * Performs Hillis-Steele-Horne scan using global memory.
 */
__global__ void
hsh(const float *A, float *B, int numElements)
{
	__shared__ float XY[BLOCK_SIZE*2];
	int rBuf = 0; int wBuf = BLOCK_SIZE;
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < numElements){
		XY[wBuf + threadIdx.x] = A[i];
	}

	for(uint s=1; s < BLOCK_SIZE; s*= 2){
		__syncthreads();
        wBuf = BLOCK_SIZE - wBuf; rBuf = BLOCK_SIZE - rBuf;
		if(threadIdx.x >= s)
			XY[wBuf + threadIdx.x] = XY[rBuf + threadIdx.x - s] + XY[rBuf + threadIdx.x];
		else
			XY[wBuf + threadIdx.x] = XY[rBuf + threadIdx.x];
	}

	// Copy to global memory
	if (i < numElements) B[i] = XY[wBuf + threadIdx.x];
}

/**
 * Retrieves the last element of each block of A, and places them in B.
 */
__global__ void
extractFinalSums(const float *A, float *B, int numElements, int stride){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if(i < numElements/stride){
		B[i] = A[(i+1)*stride-1];
	}
}

/**
 * Adds the value of the correct block to each element in the subsequent block
 */
__global__ void
addExtracts(const float *extracts_array, float *preSumArray, int num_elements){
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (1023 < i < num_elements){
		preSumArray[i] = preSumArray[i] + extracts_array[i/BLOCK_SIZE-1];
	}
}

/**
 * Host function that copies the data and launches the work on GPU
 */
float *hshFullScan(float *data, unsigned length)
{
	float *h_results = new float[length];
	unsigned size = length * sizeof(float);
	int gridSize = 1 + (length - 1) / BLOCK_SIZE;

	float *d_inputArray = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_inputArray, size), "Failed to allocate d_input_array");
	float *d_results = NULL;
	CUDA_ERROR(cudaMalloc((void **)&d_results, size), "Failed to allocate d_results");
	CUDA_ERROR(cudaMemcpy(d_inputArray, data, size, cudaMemcpyHostToDevice), "Failed to copy input array.");

	int extractLength = length/BLOCK_SIZE;
	size_t extractSize = extractLength * sizeof(float);
	int extractGridSize = 1 + (extractLength - 1) / BLOCK_SIZE;

	float *d_lastElems = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_lastElems, extractSize), "Failed to create array for last elements of each block");
	float *d_lastElemsScanned = NULL;
	CUDA_ERROR(cudaMalloc((void **) &d_lastElemsScanned, extractSize), "Failed to create array for last summed elements");
	
	hsh<<<gridSize, BLOCK_SIZE>>>(d_inputArray, d_results, length);
	extractFinalSums<<<extractGridSize, BLOCK_SIZE>>>(d_results, d_lastElems, length, BLOCK_SIZE);
	hsh<<<gridSize, BLOCK_SIZE>>>(d_lastElems, d_lastElemsScanned, extractLength);
	addExtracts<<<gridSize, BLOCK_SIZE>>>(d_lastElemsScanned, d_results, length);
	cudaDeviceSynchronize();

	CUDA_ERROR(cudaGetLastError(), "Failed to launch hsh kernel");
	CUDA_ERROR(cudaMemcpy(h_results, d_results, size, cudaMemcpyDeviceToHost), "Failed to copy results from device to host");

	CUDA_ERROR(cudaFree(d_inputArray), "Failed to free input array on GPU");
	CUDA_ERROR(cudaFree(d_results), "Failed to free results array on GPU");
	CUDA_ERROR(cudaFree(d_lastElems), "Failed to free lastElems array on GPU");
	CUDA_ERROR(cudaFree(d_lastElemsScanned), "Failed to free lastElemsScanned array on GPU");

	return h_results;
}

float *cpuScan(float *data, unsigned size)
{
	float *result = new float[size];
	float sum = 0;
	int i;
	for (i = 0; i < size ; i++){
		sum += data[i];
		result[i] = sum;
	}
	return result;
}

void initialise(float *data, unsigned size)
{
	for (unsigned i = 0; i < size; i++)
		data[i] = 0.5f;
}

static void compareResults(const float *hostVector, const float *gpuVector, int num_elements)
{
	for (int i = 0; i < num_elements; ++i){
		if (fabs(hostVector[i] - gpuVector[i]) > 1e-5f){
			fprintf(stderr, "Result verification failed at element %d!  h%0.5f : d%0.5f\n", i, hostVector[i], gpuVector[i]);
			exit (EXIT_FAILURE);
		}
	}
}

void performFullScan(){
	int length = 1000000;
	float *data = new float[length];
	initialise(data, length);

	float *cpuResult = cpuScan(data, length);
	float *gpuResult = hshFullScan(data, length);

	compareResults(cpuResult, gpuResult, length);
	std::cout << "HSH full scan completed successfully.\n";

	delete data;
	delete cpuResult;
	delete gpuResult;
}

// ************************* WARP EXPERIMENT ****************************

void runWarpExperiment(){

}

// ************************* GRAY SCALING ****************************

__global__ void
grayscale(unsigned char* inputImage, unsigned char* outputImage, unsigned int width, unsigned int height){
	const unsigned int thread = blockDim.x * blockIdx.x + threadIdx.x;
	if(thread < width*height){
		float average = (inputImage[thread] + inputImage[thread+1] + inputImage[thread+2])/3;

		outputImage[thread] = average;
		outputImage[thread+1] = average;
		outputImage[thread+2] = average;
	}
}

void filter(unsigned char* h_input, unsigned char* h_output, unsigned int width, unsigned int height){
	int length = width * height * 3;
	int size = length * sizeof(unsigned char);

	unsigned char* d_input;
	unsigned char* d_output;
	CUDA_ERROR(cudaMalloc((void **) &d_input, size), "Failed to allocate image space.");
	CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice), "Failed to copy image to GPU.");
	CUDA_ERROR(cudaMalloc((void **) &d_output, size), "Failed to allocate result image space.");

	dim3 blockDims(512, 1, 1);
	dim3 gridDims((unsigned int) ceil((double)(length/blockDims.x)), 1, 1);
	grayscale<<<gridDims, blockDims>>>(d_input, d_output, width, height);

	CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost), "Failed to copy results from GPU.");
	CUDA_ERROR(cudaFree(d_input), "Failed to free input memory.");
	CUDA_ERROR(cudaFree(d_output), "Failed to free output memory.");
}

int performGrayscale(){
    // Read the arguments
    const char* inputFile = "colour.png";
    const char* outputFile = "grayscale.png";

    std::vector<unsigned char> in_image;
    unsigned int width, height;

    // Load the data
    unsigned error = lodepng::decode(in_image, width, height, inputFile);
    if(error) std::cout << "decoder error " << error << ": " << lodepng_error_text(error) << std::endl;

    // Remove the alpha channels
//    unsigned char* inputImage = new unsigned char[(in_image.size()*3)/4];
//    unsigned char* outputImage = new unsigned char[(in_image.size()*3)/4];
//    int where = 0;
//    for(int i = 0; i < in_image.size(); ++i) {
//       if((i+1) % 4 != 0) {
//           inputImage[where] = in_image.at(i);
//           outputImage[where] = 255;
//           where++;
//       }
    }

    // Run the filter on it
    filter(inputImage, outputImage, width, height);

    // Prepare data for output
    std::vector<unsigned char> outImage;
    for(int i = 0; i < in_image.size(); ++i) {
        outImage.push_back(outputImage[i]);
        if((i+1) % 3 == 0) {
            outImage.push_back(255);
        }
    }

    // Output the data
    error = lodepng::encode(outputFile, outImage, width, height);

    //if there's an error, display it
    if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;

    delete[] inputImage;
    delete[] outputImage;

    return 0;
}

int main(void)
{
	performFullScan();
	runWarpExperiment();
	performGrayscale();
	return 0;
}
