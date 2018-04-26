#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
#include <curand_precalc.h>

/* Personal declarations */
#include "vect3d.h"
#include "particle.h"


// Physical constant describing atom collision size
const float sigmak = 1e-28; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27; // number of particles per simulated particle


cudaError_t initializeCuda(curandState_t* states, int blocks);

cudaError_t inflowPotentialParticles(curandState_t* randomStates, particle* particleList, int i, int j, int k, int meanParticlePerCell);


int main()
{
	int numberOfInflowParticlesEachStep = 1 * 2 * 3 * 4;

	curandState_t* dev_randomInflowStates = NULL;
	
	cudaError_t cudaStatus = initializeCuda(dev_randomInflowStates, numberOfInflowParticlesEachStep);

	particle *inflowParticleList = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));

	inflowPotentialParticles(dev_randomInflowStates, inflowParticleList, 1, 2, 3, 4);
	
	for (int i = 0; i < numberOfInflowParticlesEachStep ; ++i)
	{
		printf(
			"[%-2d] vel{ %.3f, %.3f, %.3f }; pos{ %.3f, %.3f, %.3f };\n",
			i,
			inflowParticleList[i].velocity.x,
			inflowParticleList[i].velocity.y,
			inflowParticleList[i].velocity.z,
			inflowParticleList[i].position.x,
			inflowParticleList[i].position.y,
			inflowParticleList[i].position.z
		);
	}

	//cudaFree(dev_randomInflowStates);

	printf("Complete");

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

/* ============================== INITIALIZE =============================== */

__global__ void initRandomStates(unsigned int seed, curandState_t* states) {
	//int idx = blockIdx.x * blockDim.x + threadIdx.x;
	///* we have to initialize the state */
	//curand_init(0, /* the seed can be the same for each core, here we pass the time in from the CPU */
	//	idx, /* the sequence number should be different for each core (unless you want all
	//				cores to get the same sequence of numbers for some reason - use thread id! */
	//	0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	//	&states[idx]);

	//curand(&states[idx]);

	curandState_t state;

	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		0, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	/* curand works like rand - except that it takes a state as a parameter */
	curand(&state);
}

cudaError_t initializeCuda(curandState_t *randomInflowStates, int blocks)
{
	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaError = cudaSetDevice(0);

	curandState_t *dev_states;

	//cudaMalloc((void**) &dev_states, blocks * sizeof(curandState_t));
	//initRandomStates <<<1, blocks>>>(2, dev_states);
	
	//printf("%d\n", time(0))

	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("init launch failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	// cudaDeviceSynchronize();

	return cudaError;
}

/* ================================ INFLOW ================================= */

__global__ void inflowKernel(particle *particles)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	curandState_t state;
	curand_init(0, /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	particles[idx].position.x = curand_uniform(&state);
	particles[idx].position.y = curand_uniform(&state);
	particles[idx].position.z = curand_uniform(&state);

	particles[idx].velocity.x = curand_uniform(&state);
	particles[idx].velocity.y = curand_uniform(&state);
	particles[idx].velocity.z = curand_uniform(&state);

	particles[idx].index = 1;
}

/* 
 Fill an array with new random particles to be exeucuted on

 Notes:
	Do I even need dev_a? Can I just use particle list?
 */
cudaError_t inflowPotentialParticles(curandState_t *randomStates, particle *particleList, int i, int j, int k, int meanParticlePerCell) {
	int numOfPoints = i*j*k*meanParticlePerCell;
	int size = numOfPoints * sizeof(particle);
	particle *dev_a;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	inflowKernel <<<1, numOfPoints>>>(dev_a);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("inflow launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}
	return cudaMemcpy(particleList, dev_a, size, cudaMemcpyDeviceToHost);
}
