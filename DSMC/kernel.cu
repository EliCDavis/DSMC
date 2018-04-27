#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

/* Personal declarations */
#include "vect3d.h"
#include "particle.h"
#include "cell.h"

// 970 Constraints
#define MAX_THREAD_PER_BLOCK 1024
#define MAX_THREAD_PER_PROCESOR 2048
#define NUM_OF_BLOCKS 64

#define CUDART_PI_F 3.141592654f

// Geometry
#define PLATE_X -0.25
#define PLATE_DY 0.25
#define PLATE_DZ 0.5

// Physical constant describing atom collision size
const float sigmak = 1e-28; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27; // number of particles per simulated particle


cudaError_t initializeCuda(curandState_t* states, int blocks);

cudaError_t inflowPotentialParticles(particle* particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp);

cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions);

particle* removeParticlesOutofBounds(particle* particles, int size, int* newSize);

cudaError_t clearCellInformation(cell* cells, int numCells);

cudaError_t sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells);


void printParticle(particle p)
{
	printf(
		"i %d; v{ %.3f, %.3f, %.3f }; p{ %.3f, %.3f, %.3f };\n",
		p.index,
		p.velocity.x,
		p.velocity.y,
		p.velocity.z,
		p.position.x,
		p.position.y,
		p.position.z
	);
}

int main()
{
	int meanParticlePerCell = 10;
	vect3d cellDimensions = vect3d(32, 32, 32);
	float vmean = 1;
	float Mach = 20;
	float vtemp = vmean / Mach;
	float deltax = 2. / float(fmax(fmax(cellDimensions.x, cellDimensions.y), cellDimensions.z));
	float deltaT = .1 * deltax / (vmean + vtemp);

	// simulate for 4 free-stream flow-through times
	float time = 8. / (vmean + vtemp);
	int numberOfTimesteps = 1 << int(ceil(log(time / deltaT) / log(2.0)));
	printf("Time: %.2f; Steps: %d\n", time, numberOfTimesteps);

	// re-sample 4 times during simulation
	const int sample_reset = numberOfTimesteps / 4;
	int nsample = 0;

	int numberOfInflowParticlesEachStep = cellDimensions.y * cellDimensions.z * meanParticlePerCell;

	int currentNumberOfParticles = numberOfInflowParticlesEachStep;

	const int numberOfCells = cellDimensions.x * cellDimensions.y * cellDimensions.z;
	cell* cellSamples = (cell*)malloc(numberOfCells * sizeof(cell));

	curandState_t* dev_randomInflowStates = NULL;
	
	cudaError_t cudaStatus = initializeCuda(dev_randomInflowStates, numberOfInflowParticlesEachStep);

	particle *inflowParticleList = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));

	inflowPotentialParticles(inflowParticleList, cellDimensions, meanParticlePerCell, vmean, vtemp);

	for (int t = 0; t < numberOfTimesteps; t++) {
		moveAndIndexParticles(inflowParticleList, currentNumberOfParticles, deltaT, cellDimensions);

		// Clean up list of particles out of bounds
		int newParticleListSize = 0;
		particle* cleanedParticles = removeParticlesOutofBounds(inflowParticleList, currentNumberOfParticles, &newParticleListSize);
		currentNumberOfParticles = newParticleListSize;
		free(inflowParticleList);
		inflowParticleList = cleanedParticles;

		if (t % sample_reset == 0)
		{
			clearCellInformation(cellSamples, numberOfCells);
			nsample = 0;
		}
		nsample++;
		
		sampleCellInformation(inflowParticleList, currentNumberOfParticles, cellSamples, numberOfCells);

		//printParticle(inflowParticleList[0]);
		printf("[%d] num particles: %d\n", t, currentNumberOfParticles);
	}

	for (int i = 0; i < currentNumberOfParticles; ++i)
	{
		printf("[%-2d] ", i);
		printParticle(inflowParticleList[i]);
	}

	// cudaFree(dev_randomInflowStates);

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

/* ================================ HELPERS ================================ */

/*
	Compute a unit vector with a random orientation and uniform distribution
*/
__device__ void randomDirection(curandState_t seed, vect3d* vel)
{
	vel[0].x = 2.0 * curand_uniform(&seed) - 1;
	double A = sqrt(1. - vel[0].x * vel[0].x);
	double theta = curand_uniform(&seed) * 2 * CUDART_PI_F;
	vel[0].y = A * cos(theta);
	vel[0].z = A * sin(theta);
}

/* =============================== 1. INFLOW =============================== */

__global__ void inflowKernel(particle *particles, vect3d cellDimensions, float vmean, float vtemp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	curandState_t state;
	curand_init(
		0,   /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0,   /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&state);

	int cellY = idx % int(cellDimensions.y);
	int cellZ = floor(cellDimensions.y);

	double dx = 2. / float(cellDimensions.x);
	double dy = 2. / float(cellDimensions.y);
	double dz = 2. / float(cellDimensions.z);

	double cx = -1 - dx;
	double cy = -1 + float(cellY) * dy;
	double cz = -1 + float(cellZ) * dz;

	particles[idx].position.x = cx + curand_uniform(&state) * dx;
	particles[idx].position.y = cy + curand_uniform(&state) * dy;
	particles[idx].position.z = cz + curand_uniform(&state) * dz;

	randomDirection(state, &particles[idx].velocity);

	double rndVel = sqrt(-log(fmax(double(sqrt(curand_uniform(&state))), 1e-200))) * vtemp;

	particles[idx].velocity.x = (particles[idx].velocity.x * rndVel) + vmean;
	particles[idx].velocity.y = particles[idx].velocity.y * rndVel;
	particles[idx].velocity.z = particles[idx].velocity.z * rndVel;

	particles[idx].index = 1;
}

/* 
 Fill an array with new random particles to be exeucuted on

 Notes:
	Do I even need dev_a? Can I just use particle list?
 
 TODO:
	figure out how to build random seeds in seperate block
	Stress test number of particles

 */
cudaError_t inflowPotentialParticles(particle *particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp) {
	
	int numOfPoints = cellDimensions.y * cellDimensions.z * meanParticlePerCell;
	int blockSize = numOfPoints / NUM_OF_BLOCKS;
	/*int numParticlesPerThread = 1;
	if (blockSize > MAX_THREAD_PER_BLOCK)
	{
		numParticlesPerThread = ceil(float(blockSize) / float(MAX_THREAD_PER_BLOCK));
	}*/

	int size = numOfPoints * sizeof(particle);
	particle *dev_a;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	inflowKernel <<<NUM_OF_BLOCKS, blockSize>>>(dev_a, cellDimensions, vmean, vtemp);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("inflow launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	/*cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}*/

	return cudaMemcpy(particleList, dev_a, size, cudaMemcpyDeviceToHost);
}


/* =========================== 2. MOVE PARTICLES =========================== */

__global__ void moveParticlesKernel(particle* particles, float deltaTime, vect3d cellDimensions, vect3d dividedCellDimensions) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	float newPosX = particles[idx].position.x + (particles[idx].velocity.x * deltaTime);
	float newPosY = particles[idx].position.y + (particles[idx].velocity.y * deltaTime);
	float newPosZ = particles[idx].position.z + (particles[idx].velocity.z * deltaTime);

	// May have passed through the plate..
	if ((particles[idx].position.x < PLATE_X && newPosX > PLATE_X) ||
		(particles[idx].position.x > PLATE_X && newPosX < PLATE_X)) {

		// Where did it actually pass through the plate..
		double t = (particles[idx].position.x - PLATE_X) / (particles[idx].position.x - newPosX);
		float pointOfCollisionY = (particles[idx].position.y * (1.0 - t)) + (newPosY * t);
		float pointOfCollisionZ = (particles[idx].position.z * (1.0 - t)) + (newPosZ * t);

		// Actually collided..
		if ((pointOfCollisionY < PLATE_DY && pointOfCollisionY > -PLATE_DY) &&
			(pointOfCollisionZ < PLATE_DZ && pointOfCollisionZ > -PLATE_DZ))
		{
			newPosX = newPosX - 2 * (newPosX - PLATE_X);
			particles[idx].velocity.x = -particles[idx].velocity.x;
		}

	}

	if (newPosY > 1) {
		newPosY -= 2.0;
	} else if (newPosY < -1) {
		newPosY += 2.0;
	}
	
	if (newPosZ > 1) {
		newPosZ -= 2.0;
	} else if (newPosZ < -1) {
		newPosZ += 2.0;
	}

	particles[idx].position.x = newPosX;

	// Particle moved out of bounds?
	if (particles[idx].position.x < -1 || particles[idx].position.x > 1) {
		// Mark for deletion
		particles[idx].status = -1;
	} else {
		// Assign particle positions and index
		particles[idx].position.y = newPosY;
		particles[idx].position.z = newPosZ;

		int i = int(fmin(floor((newPosX + 1.0) / dividedCellDimensions.x), double(cellDimensions.x - 1)));
		int j = int(fmin(floor((newPosY + 1.0) / dividedCellDimensions.y), double(cellDimensions.y - 1)));
		int k = int(fmin(floor((newPosZ + 1.0) / dividedCellDimensions.z), double(cellDimensions.z - 1)));
		particles[idx].index = i * cellDimensions.y * cellDimensions.z + j * cellDimensions.z + k;
	}
}

/*
	Move particles appropriately and marks those out of bounds with a flag for deletion.
	Reindexes particles not marked for deletion

	TODO:
		Some how parrallel sum how many particles now need to be deleted..
*/
cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions) {
	int blockSize = numOfParticles / NUM_OF_BLOCKS;
	particle *dev_a;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, numOfParticles*sizeof(particle));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaMemcpy(dev_a, particleList, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);

	vect3d dividedCellDimensions = vect3d(2. / cellDimensions.x, 2. / cellDimensions.y, 2. / cellDimensions.z);

	moveParticlesKernel <<<NUM_OF_BLOCKS, blockSize >>>(dev_a, deltaTime, cellDimensions, dividedCellDimensions);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(particleList, dev_a, numOfParticles * sizeof(particle), cudaMemcpyDeviceToHost);

	cudaFree(dev_a);

	return cudaStatus;
}

/* ========================== 3. REMOVE PARTICLES ========================== */

particle* removeParticlesOutofBounds(particle* particles, int size, int* newSize) {
	*newSize = size;
	for (int i = 0; i < size; i++) {
		if (particles[i].status == -1) {
			*newSize -= 1;
		}
	}

	particle* newParticleList = (particle*)malloc(*newSize * sizeof(particle));
	int added = 0;
	for (int i = 0; i < size; i++) {
		if (particles[i].status != -1) {
			newParticleList[added] = particle(particles[i]);
			added += 1;
		}
	}

	return newParticleList;
}

/* ============================== 4. SAMPLING ============================== */

__global__ void clearCellsKernel(cell* cells) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	cells[idx].numberOfParticles = 0;
	cells[idx].velocity.x = 0;
	cells[idx].velocity.y = 0;
	cells[idx].velocity.z = 0;
	cells[idx].energy = 0;
}

/*
	TODO:
		CHECK IF THIS IS ANY BETTER THAN DOING IT LINEARLY
*/
cudaError_t	clearCellInformation(cell* cells, int numCells) {
	int blockSize = numCells / NUM_OF_BLOCKS;
	cell *deviceCells;

	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCells, numCells * sizeof(cell));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaMemcpy(deviceCells, cells, numCells * sizeof(cell), cudaMemcpyHostToDevice);

	clearCellsKernel <<<NUM_OF_BLOCKS, blockSize>> >(deviceCells);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(cells, deviceCells, numCells * sizeof(cell), cudaMemcpyDeviceToHost);

	cudaFree(deviceCells);

	return cudaStatus;
}

__global__ void sampleCellsKernel(cell* cells, particle* particles, int particleSize) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	for (int particleIndex = 0; particleIndex < particleSize; particleIndex++)
	{
		if (particles[particleIndex].index == idx)
		{
			cells[idx].numberOfParticles = cells[idx].numberOfParticles + 1;
			cells[idx].velocity.x = particles[particleIndex].velocity.x + cells[idx].velocity.x;
			cells[idx].velocity.y = particles[particleIndex].velocity.y + cells[idx].velocity.y;
			cells[idx].velocity.z = particles[particleIndex].velocity.z + cells[idx].velocity.z;
			cells[idx].energy = cells[idx].energy + (
				.5 * 
				((particles[particleIndex].velocity.x * particles[particleIndex].velocity.x) +
				(particles[particleIndex].velocity.y * particles[particleIndex].velocity.y) +
				(particles[particleIndex].velocity.z * particles[particleIndex].velocity.z)));
		}
	}
}

cudaError_t sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells) {
	int blockSize = numCells / NUM_OF_BLOCKS;
	cell *deviceCells;
	particle* deviceParticles;

	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCells, numCells * sizeof(cell));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&deviceParticles, numOfParticles * sizeof(particle));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaMemcpy(deviceCells, cells, numCells * sizeof(cell), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceParticles, particles, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);


	sampleCellsKernel <<<NUM_OF_BLOCKS, blockSize>>>(deviceCells, particles, numOfParticles);

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(cells, deviceCells, numCells * sizeof(cell), cudaMemcpyDeviceToHost);

	cudaFree(deviceCells);
	cudaFree(deviceParticles);

	return cudaStatus;
}