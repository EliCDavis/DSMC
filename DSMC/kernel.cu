#include <stdio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#include "vect3d.h"
#include "particle.h"
#include "cell.h"
#include "collisionInfo.h"

/* From old program */
#include <list>
#include <vector>
#include <stdlib.h>
#include "vect3d.h"

#include <iostream>
#include <fstream>

// 970 Constraints
#define MAX_THREAD_PER_BLOCK 1024
#define MAX_THREAD_PER_PROCESOR 2048
#define NUM_OF_BLOCKS 2048

#define CUDART_PI_F 3.141592654f

// Geometry
#define PLATE_X -0.25
#define PLATE_DY 0.25
#define PLATE_DZ 0.5

// Physical constant describing atom collision size
const float sigmak = 1e-28; // collision cross section

// Note, pnum recomputed from mean particle per cell and density
float pnum = 1e27; // number of particles per simulated particle

using namespace std;


cudaError_t initializeCuda(curandState_t* states, int blocks);

cudaError_t inflowPotentialParticles(curandState_t* randomStates, particle* particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp);

cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions);

particle* removeParticlesOutofBounds(particle* particles, int size, int* newSize);

cudaError_t clearCellInformation(cell* cells, int numCells);

void sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells);

void collideParticles(
	particle* particleList,
	int particleListSize,
	collisionInfo* collisionData,
	cell* cellData,
	int cellDataSize,
	int nsample,
	float cellvol,
	float deltaT);

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

float ranf() {
	return rand() / 32767.0f;
}

void initializeCollision(collisionInfo *collisionData, int size ,float vtemp)
{
	for (int i = 0; i < size; ++i)
	{
		collisionData[i].maxCollisionRate = sigmak * vtemp;
		collisionData[i].collisionRemainder = ranf();
	}
}


void writeParticles(int step, particle* particles, int num) {
	ofstream myfile;
	myfile.open("output.txt");
	for (int p = 0; p < num; p++) {
		myfile << particles[p].position.x << " " << particles[p].position.y << " " << particles[p].position.z << " " << particles[p].status << endl;
	}
	myfile.close();
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
	float density = 1e30; // Number of molecules per unit cube of space

	// simulate for 4 free-stream flow-through times
	float time = 8. / (vmean + vtemp);
	int numberOfTimesteps = 1 << int(ceil(log(time / deltaT) / log(2.0)));
	printf("Time: %.2f; Steps: %d\n", time, numberOfTimesteps);


	// re-sample 4 times during simulation
	const int sample_reset = numberOfTimesteps / 4;
	int nsample = 0;

	int numberOfInflowParticlesEachStep = cellDimensions.y * cellDimensions.z * meanParticlePerCell;

	int currentNumberOfParticles = 0;

	const int numberOfCells = cellDimensions.x * cellDimensions.y * cellDimensions.z;
	pnum = density * numberOfCells / float(meanParticlePerCell);


	cell* cellSamples = (cell*)malloc(numberOfCells * sizeof(cell));

	collisionInfo* collisionData = (collisionInfo*)malloc(numberOfCells * sizeof(collisionInfo));
	initializeCollision(collisionData, numberOfCells, vtemp);

	curandState_t* randomInflowStates = (curandState_t*)malloc(numberOfInflowParticlesEachStep * sizeof(curandState_t));
	
	cudaError_t cudaStatus = initializeCuda(randomInflowStates, numberOfInflowParticlesEachStep);
	if (cudaStatus != cudaSuccess) {
		printf("init failed: %s\n", cudaGetErrorString(cudaStatus));
		return -1;
	}

	particle *allParticles = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));
	particle *inflowParticleList = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));

	for (int t = 0; t < 500; t++) {

		cudaStatus = inflowPotentialParticles(randomInflowStates, inflowParticleList, cellDimensions, meanParticlePerCell, vmean, vtemp);
		if (cudaStatus != cudaSuccess) {
			printf("inflow failed: %s\n", cudaGetErrorString(cudaStatus));
			return 1;
		}
		// Combine new particles with existing
		particle* newTotal = (particle*)malloc((numberOfInflowParticlesEachStep + currentNumberOfParticles) * sizeof(particle));
		memcpy(newTotal, inflowParticleList, numberOfInflowParticlesEachStep * sizeof(particle));
		if (currentNumberOfParticles != 0) {
			memcpy(newTotal + numberOfInflowParticlesEachStep, allParticles, currentNumberOfParticles * sizeof(particle));
		}
		free(allParticles);
		allParticles = newTotal;
		currentNumberOfParticles += numberOfInflowParticlesEachStep;

		moveAndIndexParticles(allParticles, currentNumberOfParticles, deltaT, cellDimensions);

		// Clean up list of particles out of bounds
		int newParticleListSize = 0;
		particle* cleanedParticles = removeParticlesOutofBounds(allParticles, currentNumberOfParticles, &newParticleListSize);
		currentNumberOfParticles = newParticleListSize;
		free(allParticles);
		allParticles = cleanedParticles;

		if (t % sample_reset == 0)
		{
			clearCellInformation(cellSamples, numberOfCells);
			nsample = 0;
		}
		nsample++;
		
		sampleCellInformation(allParticles, currentNumberOfParticles, cellSamples, numberOfCells);

		collideParticles(allParticles,
			currentNumberOfParticles,
			collisionData,
			cellSamples,
			numberOfCells,
			nsample,
			numberOfCells,
			deltaT);

		//printParticle(allParticles[0]);
		printf("[%d] num particles: %d\n", t, currentNumberOfParticles);
		
	}

	writeParticles(0, allParticles, currentNumberOfParticles);

	/*for (int i = 0; i < currentNumberOfParticles; ++i)
	{
		printf("[%-2d] ", i);
		printParticle(allParticles[i]);
	}*/

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
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	///* we have to initialize the state */
	//curand_init(0, /* the seed can be the same for each core, here we pass the time in from the CPU */
	//	idx, /* the sequence number should be different for each core (unless you want all
	//				cores to get the same sequence of numbers for some reason - use thread id! */
	//	0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
	//	&states[idx]);

	//curand(&states[idx]);

	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[idx]);

	/* curand works like rand - except that it takes a state as a parameter */
	curand(&states[idx]);
}

cudaError_t initializeCuda(curandState_t *randomInflowStates, int numOfStates)
{
	int blockSize = numOfStates / NUM_OF_BLOCKS;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaError_t cudaError = cudaSetDevice(0);

	curandState_t *dev_states;

	cudaError = cudaMalloc((void**) &dev_states, numOfStates * sizeof(curandState_t));
	if (cudaError != cudaSuccess) {
		printf("init malloc failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	initRandomStates <<<NUM_OF_BLOCKS, blockSize >>>(2, dev_states);
	
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("init launch failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	cudaError = cudaMemcpy(randomInflowStates, dev_states, numOfStates * sizeof(curandState_t), cudaMemcpyDeviceToHost);
	if (cudaError != cudaSuccess) {
		printf("inti memcpy failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	return cudaFree(dev_states);
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

__global__ void inflowKernel(curandState_t *randState, particle *particles, int dimX, int dimY, int dimZ, float vmean, float vtemp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	int cellY = idx % int(dimY);
	int cellZ = int(floorf(float(idx) / float(dimY)));

	double dx = 2. / float(dimX);
	double dy = 2. / float(dimY);
	double dz = 2. / float(dimZ);

	double cx = -1 - dx;
	double cy = -1 + float(cellY) * dy;
	double cz = -1 + float(cellZ) * dz;

	particles[idx].position.x = cx + curand_uniform(&randState[idx]) * dx;
	particles[idx].position.y = cy + curand_uniform(&randState[idx]) * dy;
	particles[idx].position.z = cz + curand_uniform(&randState[idx]) * dz;

	randomDirection(randState[idx], &particles[idx].velocity);

	double rndVel = sqrt(-log(fmax(double(sqrt(curand_uniform(&randState[idx]))), 1e-200))) * vtemp;

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
cudaError_t inflowPotentialParticles(curandState_t* randomStates, particle *particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp) {
	
	int numOfPoints = cellDimensions.y * cellDimensions.z * meanParticlePerCell;
	int blockSize = numOfPoints / NUM_OF_BLOCKS;
	/*int numParticlesPerThread = 1;
	if (blockSize > MAX_THREAD_PER_BLOCK)
	{
		numParticlesPerThread = ceil(float(blockSize) / float(MAX_THREAD_PER_BLOCK));
	}*/

	int size = numOfPoints * sizeof(particle);
	
	curandState_t *dev_randomStates;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_randomStates, numOfPoints * sizeof(curandState_t));
	if (cudaStatus != cudaSuccess) {
		printf("inflow malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaMemcpy(dev_randomStates, randomStates, numOfPoints * sizeof(curandState_t), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("inflow memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	particle *dev_a;
	cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		printf("inflow malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	inflowKernel <<<NUM_OF_BLOCKS, blockSize>>>(dev_randomStates, dev_a, cellDimensions.x, cellDimensions.y, cellDimensions.z, vmean, vtemp);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("inflow launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(randomStates, dev_randomStates, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("inflow memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(particleList, dev_a, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("inflow memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaFree(dev_randomStates);
	if (cudaStatus != cudaSuccess) {
		printf("inflow free rnd failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaFree(dev_a);
}



/* =========================== 2. MOVE PARTICLES =========================== */

/*
	TODO:
		MOVE BRANCHIGN COLLISION LOGIC OUTSIDE OF KERNEL
*/
__global__ void moveParticlesKernel(particle* particles, float deltaTime, int dimX, int dimY, int dimZ, float divX, float divY, float divZ) {
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
			particles[idx].status = 1;
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

	// Assign particle positions and index
	particles[idx].position.x = newPosX;
	particles[idx].position.y = newPosY;
	particles[idx].position.z = newPosZ;

	int i = int(fmin(floor((newPosX + 1.0) / divX), double(dimX - 1)));
	int j = int(fmin(floor((newPosY + 1.0) / divY), double(dimY - 1)));
	int k = int(fmin(floor((newPosZ + 1.0) / divZ), double(dimZ - 1)));
	particles[idx].index = i * dimY * dimZ + j * dimZ + k;
}

/*
	Move particles appropriately and marks those out of bounds with a flag for deletion.
	Reindexes particles not marked for deletion

	TODO:
		Some how parrallel sum how many particles now need to be deleted..
*/
cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions) {
	int blockSize = ceil(numOfParticles / NUM_OF_BLOCKS);
	particle *dev_a;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, numOfParticles*sizeof(particle));
	if (cudaStatus != cudaSuccess) {
		printf("move&index malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_a, particleList, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("move&index memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	vect3d dividedCellDimensions = vect3d(2. / cellDimensions.x, 2. / cellDimensions.y, 2. / cellDimensions.z);

	moveParticlesKernel <<<NUM_OF_BLOCKS, blockSize >>>(dev_a, deltaTime, cellDimensions.x, cellDimensions.y, cellDimensions.z, dividedCellDimensions.x, dividedCellDimensions.y, dividedCellDimensions.z);

	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("move&index launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(particleList, dev_a, numOfParticles * sizeof(particle), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("move&index memcpy to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaFree(dev_a);
}

/* ========================== 3. REMOVE PARTICLES ========================== */

particle* removeParticlesOutofBounds(particle* particles, int size, int* newSize) {
	*newSize = size;
	for (int i = 0; i < size; i++) {
		if (particles[i].position.x < -1 || particles[i].position.x > 1) {
			*newSize -= 1;
		}
	}

	particle* newParticleList = (particle*)malloc(*newSize * sizeof(particle));
	int added = 0;
	for (int i = 0; i < size; i++) {
		if (particles[i].position.x >= -1 && particles[i].position.x <= 1) {
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

void sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells) {
	for (int p = 0; p < numOfParticles; p++) {
		int cellIndex = particles[p].index;
		cells[cellIndex].numberOfParticles++;
		cells[cellIndex].velocity += particles[p].velocity;
		cells[cellIndex].energy += .5* dot(particles[p].velocity, particles[p].velocity);

	}
}

//cudaError_t sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells) {
//	int blockSize = numCells / NUM_OF_BLOCKS;
//	cell *deviceCells;
//	particle* deviceParticles;
//
//	const int maxParticles = 10000;
//	int iterations = ceil(double(numOfParticles) / double(maxParticles));
//
//	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCells, numCells * sizeof(cell));
//	if (cudaStatus != cudaSuccess) {
//		printf("sample cell malloc failed: %s\n", cudaGetErrorString(cudaStatus));
//		return cudaStatus;
//	}
//
//	cudaStatus = cudaMemcpy(deviceCells, cells, numCells * sizeof(cell), cudaMemcpyHostToDevice);
//	if (cudaStatus != cudaSuccess) {
//		printf("sample cell memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
//		return cudaStatus;
//	}
//
//	cudaStatus = cudaMalloc((void**)&deviceParticles, maxParticles * sizeof(particle));
//	if (cudaStatus != cudaSuccess) {
//		printf("sample particle malloc failed: %s\n", cudaGetErrorString(cudaStatus));
//		return cudaStatus;
//	}
//
//	//if (numOfParticles > 55205) {
//	//	printf("scream");
//	//}
//
//	for (int i = 0; i < iterations; i++) {
//
//		int numParticlesThisIteration = iterations == i + 1 ? numOfParticles % maxParticles : maxParticles;
//
//		cudaStatus = cudaMemcpy(deviceParticles, particles+(i*maxParticles), numParticlesThisIteration * sizeof(particle), cudaMemcpyHostToDevice);
//		if (cudaStatus != cudaSuccess) {
//			printf("sample particle memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
//			return cudaStatus;
//		}
//
//		sampleCellsKernel <<<NUM_OF_BLOCKS, blockSize >>>(deviceCells, deviceParticles, numParticlesThisIteration);
//
//		cudaStatus = cudaGetLastError();
//		if (cudaStatus != cudaSuccess) {
//			printf("sample launch failed: %s\n", cudaGetErrorString(cudaStatus));
//			return cudaStatus;
//		}
//
//		
//
//	}
//
//	cudaStatus = cudaMemcpy(cells, deviceCells, numCells * sizeof(cell), cudaMemcpyDeviceToHost);
//	if (cudaStatus != cudaSuccess) {
//		printf("sample mcmcpy back to host failed: %s\n", cudaGetErrorString(cudaStatus));
//		return cudaStatus;
//	}
//
//	cudaStatus = cudaFree(deviceParticles);
//
//	cudaStatus = cudaFree(deviceCells);
//
//	return cudaStatus;
//}

/* ============================ Cell Collision ============================= */



// Computes a unit vector with a random orientation and uniform distribution
inline vect3d randomDir()
{
	double B = 2. * ranf() - 1;
	double A = sqrt(1. - B * B);
	double theta = ranf() * 2 * CUDART_PI_F;
	return vect3d(B, A * cos(theta), A * sin(theta));
}

void collideParticles(
	particle* particleList,
	int particleListSize,
	collisionInfo* collisionData,
	cell* cellData,
	int cellDataSize,
	int nsample, 
	float cellvol, 
	float deltaT)
{

	// Compute number of particles per cell and compute a set of pointers
	// from each cell to the corresponding particles
	vector<int> np(cellDataSize), cnt(cellDataSize);
	for (int i = 0; i < cellDataSize; ++i)
	{
		np[i] = 0;
		cnt[i] = 0;
	}

	for (int p = 0; p < particleListSize; p++)
	{
		np[particleList[p].index]++;
	}

	// Offsets will contain the index in the pmap data structure where
	// the pointers to particles for the given cell will begin
	vector<int> offsets(cellDataSize + 1);
	offsets[0] = 0;
	for (int i = 0; i < cellDataSize; ++i)
	{
		offsets[i + 1] = offsets[i] + np[i];
	}

	// pmap is a structure of pointers from cells to particles, note
	// since there may be many particles per cell, the offsets need to
	// be used to access particles from this data structure.
	vector<particle *> pmap(offsets[cellDataSize]);
	for (int p = 0; p < particleListSize; p++)
	{
		int i = particleList[p].index;
		pmap[cnt[i] + offsets[i]] = &(particleList[p]);
		cnt[i]++;
	}
	
	// Loop over cells and select particles to perform collisions
	for (int i = 0; i < cellDataSize; ++i)
	{
		// Compute mean and instantaneous particle numbers for the cell
		float n_mean = float(cellData[i].numberOfParticles) / float(nsample);
		float n_instant = np[i];

		// Compute a number of particles that need to be selected for
		// collision tests
		float select = n_instant * n_mean * pnum * collisionData[i].maxCollisionRate * deltaT / cellvol + collisionData[i].collisionRemainder;
		// We can only check an integer number of collisions in any timestep
		int nselect = int(select);
		// The remainder collision fraction is saved for next timestep
		collisionData[i].collisionRemainder = select - float(nselect);
		if (nselect > 0)
		{ // selected particles for collision
			if (np[i] < 2)
			{ // if not enough particles for collision, wait until
			  // we have enough
				collisionData[i].collisionRemainder += nselect;
			}
			else
			{
				// Select nselect particles for possible collision
				float cmax = collisionData[i].maxCollisionRate;
				for (int c = 0; c < nselect; ++c)
				{
					// select two points in the cell
					int pt1 = min(int(floor(ranf() * n_instant)), np[i] - 1);
					int pt2 = min(int(floor(ranf() * n_instant)), np[i] - 1);

					// Make sure they are unique points
					while (pt1 == pt2)
						pt2 = min(int(floor(ranf() * n_instant)), np[i] - 1);

					// Compute the relative velocity of two particles
					vect3d v1 = pmap[offsets[i] + pt1]->velocity;
					vect3d v2 = pmap[offsets[i] + pt2]->velocity;
					vect3d vr = v1 - v2;
					float vrm = norm(vr);
					// Compute collision  rate for hard sphere model
					float crate = sigmak * vrm;
					if (crate > cmax) {
						cmax = crate;
					}
					// Check if these particles actually collide
					if (ranf() < crate / collisionData[i].maxCollisionRate)
					{
						// Collision Accepted, adjust particle velocities
						// Compute center of mass velocity, vcm
						vect3d vcm = .5 * (v1 + v2);
						// Compute random perturbation that conserves momentum
						vect3d vp = randomDir() * vrm;

						// Adjust particle velocities to reflect collision
						pmap[offsets[i] + pt1]->velocity = vcm + 0.5 * vp;
						pmap[offsets[i] + pt2]->velocity = vcm - 0.5 * vp;

						pmap[offsets[i] + pt1]->status = 2;
						pmap[offsets[i] + pt2]->status = 2;
					}
				}
				// Update the maximum collision rate to be used in future timesteps
				// for determining number of particles to select.
				collisionData[i].maxCollisionRate = cmax;
			}
		}
	}
}