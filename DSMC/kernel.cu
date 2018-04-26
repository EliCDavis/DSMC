
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

/* Personal declarations */
#include "vect3d.h"
#include "particle.h"

#include <stdio.h>

cudaError_t inflowPotentialParticles(particle* particleList, int i, int j, int k, int meanParticlePerCell);

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

// Physical constant describing atom collision size
const float sigmak = 1e-28; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27; // number of particles per simulated particle

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void inflowKernel(particle *particles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	/*particles[idx] = particle(
		vect3d(blockIdx.x, blockDim.x, threadIdx.x), 
		vect3d(blockIdx.x, blockDim.x, threadIdx.x)
	);*/
	particles[idx].position.x = blockIdx.x;
	particles[idx].position.y = blockDim.x;
	particles[idx].position.z = threadIdx.x;

	particles[idx].velocity.x = blockIdx.x;
	particles[idx].velocity.y = blockDim.x;
	particles[idx].velocity.z = threadIdx.x;

	particles[idx].index = 1;

}

cudaError_t initializeCuda()
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	return cudaSetDevice(0);
}

int main()
{
	/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

	*/

	cudaError_t cudaStatus = initializeCuda();

	particle *particleList = 0;
	inflowPotentialParticles(particleList, 1, 2, 3, 4);

	printf("I think I completed?");
	printf("Maybe?");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/* 
 Fill an array with new random particles to be exeucuted on
 */
cudaError_t inflowPotentialParticles(particle *particleList, int i, int j, int k, int meanParticlePerCell) {
	
	int numOfPoints = i*j*k*meanParticlePerCell;
	int size = numOfPoints * sizeof(particle);
	particle *dev_a;
	particle *host_a = (particle*)malloc(size);

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	// Not sure if I have to do this, kinda just want them null right...
	// Copy empty particle list..
	/*cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}*/

	inflowKernel <<<1, numOfPoints>>>(dev_a);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	cudaStatus = cudaMemcpy(host_a, dev_a, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numOfPoints; ++i)
	{
		printf(
			"[%d] vel{ %f, %f, %f }\n", 
			i, 
			host_a[i].velocity.x, 
			host_a[i].velocity.y,
			host_a[i].velocity.z
		);
	}

	return cudaStatus;

}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
	int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
