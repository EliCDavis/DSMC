#include <stdio.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>

#include "vect3d.h"
#include "particle.h"
#include "cell.h"
#include "collisionInfo.h"


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

cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions, vect3d dividedCellDimensions);

particle* removeParticlesOutofBounds(particle* particles, int size, int* newSize, cell* cells, int numberOfCells, particle*** outMap);

cudaError_t clearCellInformation(cell* cells, int numCells);

void sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells, particle*** cellToParticleMap);

void collideParticles(
	particle* particleList,
	int particleListSize,
	collisionInfo* collisionData,
	cell* cellData,
	int cellDataSize,
	int nsample,
	float cellvol,
	float deltaT,
	particle*** cellToParticleMap);

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
	vect3d cellDimensions = vect3d(128, 128, 128);
	vect3d dividedCellDimensions = vect3d(2. / cellDimensions.x, 2. / cellDimensions.y, 2. / cellDimensions.z);
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
	
	// TODO TRY IN KERNEL AND MONITOR FOR PERFORMANCE IMPROVEMENT
	//for (int c = 0; c < numberOfCells; c++) {
	//	cellSamples[c].currentNumberOfParticles = 0;
	//}

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

	particle*** cellToParticlesMap = NULL;

	clock_t clockTime;

	for (int t = 0; t < numberOfTimesteps/4; t++) {

		if (t == 11) {
			printf("im about to die");
		}

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

		moveAndIndexParticles(allParticles, currentNumberOfParticles, deltaT, cellDimensions, dividedCellDimensions);

		// Clean up list of particles out of bounds and recompute cell map
		particle*** newCellMap = (particle***)malloc(numberOfCells * sizeof(particle**));
		int newParticleListSize = 0;
		particle* cleanedParticles = removeParticlesOutofBounds(allParticles, currentNumberOfParticles, &newParticleListSize, cellSamples, numberOfCells, newCellMap);
		currentNumberOfParticles = newParticleListSize;
		free(allParticles);
		allParticles = cleanedParticles;

		if (cellToParticlesMap != NULL) {
			for (int c = 0; c < numberOfCells; c++) {
				free(cellToParticlesMap[c]);
			}
			free(cellToParticlesMap);
		}
		cellToParticlesMap = newCellMap;

		if (t % sample_reset == 0)
		{
			clearCellInformation(cellSamples, numberOfCells);
			nsample = 0;
		}
		nsample++;
		
		sampleCellInformation(allParticles, currentNumberOfParticles, cellSamples, numberOfCells, cellToParticlesMap);

		clockTime = clock();

		collideParticles(allParticles,
			currentNumberOfParticles,
			collisionData,
			cellSamples,
			numberOfCells,
			nsample,
			numberOfCells,
			deltaT,
			cellToParticlesMap);

		clockTime = clock() - clockTime;
		printf("[%d] time: %f to collide ", t, ((double)clockTime) / CLOCKS_PER_SEC);
		printf("num particles: %d\n", currentNumberOfParticles);
	}

	writeParticles(0, allParticles, currentNumberOfParticles);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

	printf("Complete");

    return 0;
}

/* ============================== INITIALIZE =============================== */

__global__ void initRandomStates(unsigned int seed, curandState_t* states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[idx]);
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

__global__ void inflowKernel(int numOfParticles, curandState_t *randState, particle *particles, int dimX, int dimY, int dimZ, float vmean, float vtemp)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numOfParticles) {
		return;
	}

	int k = idx % (dimX* dimY);
	int cellY = k % int(dimY);
	int cellZ = int(floorf(float(k) / float(dimY)));

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

	int numberOfBlocks = ceil(double(numOfPoints) / double(MAX_THREAD_PER_BLOCK));

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

	inflowKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(numOfPoints, dev_randomStates, dev_a, cellDimensions.x, cellDimensions.y, cellDimensions.z, vmean, vtemp);
	
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
__global__ void moveParticlesKernel(particle* particles, int numParticles, float deltaTime, int dimX, int dimY, int dimZ, float divX, float divY, float divZ) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numParticles) {
		return;
	}

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
cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions, vect3d dividedCellDimensions) {
	int numberOfBlocks = ceil(double(numOfParticles) / double(MAX_THREAD_PER_BLOCK));
	
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

	moveParticlesKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK>>>(dev_a, numOfParticles, deltaTime, cellDimensions.x, cellDimensions.y, cellDimensions.z, dividedCellDimensions.x, dividedCellDimensions.y, dividedCellDimensions.z);

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

particle* removeParticlesOutofBounds(particle* particles, int originalParticleSize, int* newSize, cell* cells, int numberOfCells, particle*** outMap) {
	
	int* cellParticleCount = (int*)malloc(numberOfCells * sizeof(int));
	for (int i = 0; i < numberOfCells; i ++) {
		cellParticleCount[i] = 0;
 	}
	
	*newSize = originalParticleSize;

	for (int i = 0; i < originalParticleSize; i++) {
		if (particles[i].position.x < -1 || particles[i].position.x > 1) {
			*newSize -= 1;
		}
		else {
			cellParticleCount[particles[i].index] += 1;
		}
	}

	for (int i = 0; i < numberOfCells; i ++) {
		cells[i].currentNumberOfParticles = 0;
		outMap[i] = (particle**)malloc(cellParticleCount[i] * sizeof(particle*));
	}

	particle* newParticleList = (particle*)malloc(*newSize * sizeof(particle));
	int added = 0;
	for (int i = 0; i < originalParticleSize; i++) {
		if (particles[i].position.x >= -1 && particles[i].position.x <= 1) {
			newParticleList[added] = particle(particles[i]);
			int cellIndex = newParticleList[added].index;
			outMap[cellIndex][cells[cellIndex].currentNumberOfParticles] = &newParticleList[added];
			//outMap[newParticleList[added].index][cells[i].currentNumberOfParticles];
			added += 1;
			cells[cellIndex].currentNumberOfParticles += 1;
		}
	}

	free(cellParticleCount);

	return newParticleList;
}

/* ============================== 4. SAMPLING ============================== */

__global__ void clearCellsKernel(cell* cells, int numCells) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numCells) {
		return;
	}
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
	int numberOfBlocks = ceil(double(numCells) / double(MAX_THREAD_PER_BLOCK));
	cell *deviceCells;

	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCells, numCells * sizeof(cell));
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaMemcpy(deviceCells, cells, numCells * sizeof(cell), cudaMemcpyHostToDevice);

	clearCellsKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(deviceCells, numCells);

	cudaStatus = cudaMemcpy(cells, deviceCells, numCells * sizeof(cell), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		return cudaStatus;
	}

	cudaFree(deviceCells);

	return cudaStatus;
}

__global__ void sampleCellsKernel(cell* cells, int cellSize, particle* particles, int particleSize, particle*** cellToParticleMap) {
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

void sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells, particle*** cellToParticleMap) {
	
	for (int p = 0; p < numOfParticles; p++) {
		int cellIndex = particles[p].index;
		cells[cellIndex].numberOfParticles++;
		cells[cellIndex].velocity += particles[p].velocity;
		cells[cellIndex].energy += .5* dot(particles[p].velocity, particles[p].velocity);
	}

	//int numberOfBlocks = ceil(double(numCells) / double(MAX_THREAD_PER_BLOCK));
	//
	//cell *deviceCells;
	//particle *deviceParticles;
	//particle*** deviceCellToParticleMap;

	//cudaError_t cudaStatus = cudaMalloc((void**)&deviceCells, numCells * sizeof(cell));
	//if (cudaStatus != cudaSuccess) {
	//	printf("sample malloc cells failed: %s\n", cudaGetErrorString(cudaStatus));
	//	return cudaStatus;
	//}

	//cudaError_t cudaStatus = cudaMalloc((void**)&deviceParticles, numOfParticles * sizeof(particle));
	//if (cudaStatus != cudaSuccess) {
	//	printf("sample malloc particles failed: %s\n", cudaGetErrorString(cudaStatus));
	//	return cudaStatus;
	//}

	//cudaError_t cudaStatus = cudaMemcpy(deviceCells, cells, numCells * sizeof(cell), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	printf("sample memcpy cells failed: %s\n", cudaGetErrorString(cudaStatus));
	//	return cudaStatus;
	//}

	//cudaError_t cudaStatus = cudaMemcpy(deviceParticles, particles, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);
	//if (cudaStatus != cudaSuccess) {
	//	printf("sample memcpy particles failed: %s\n", cudaGetErrorString(cudaStatus));
	//	return cudaStatus;
	//}

	//sampleCellsKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(deviceCells, numCells, deviceParticles, particleSize, deviceCellToParticleMap);

	//cudaStatus = cudaGetLastError();
	//if (cudaStatus != cudaSuccess) {
	//	printf("sample memcpy to host failed: %s\n", cudaGetErrorString(cudaStatus));
	//	return cudaStatus;
	//}

	//cudaStatus = cudaMemcpy(cells, deviceCells, numCells * sizeof(cell), cudaMemcpyDeviceToHost);

	//cudaFree(deviceCells);

	//cudaFree(deviceParticles);

	//return cudaStatus;

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

__global__ void preCollisionKernel(cell* cells, collisionInfo* collisionData, int cellSize, float nSample, float deltaT, float cellVol, float moleculesPerParticle) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= cellSize) {
		return;
	}

	// Compute mean and instantaneous particle numbers for the cell
	float n_mean = float(cells[idx].numberOfParticles) / float(nSample);
	float n_instant = cells[idx].currentNumberOfParticles;

	// Compute a number of particles that need to be selected for collision tests
	float select = n_instant * n_mean * moleculesPerParticle * collisionData[idx].maxCollisionRate * deltaT / cellVol + collisionData[idx].collisionRemainder;
	
	// We can only check an integer number of collisions in any timestep
	collisionData[idx].nSelect = int(select);

	// The remainder collision fraction is saved for next timestep
	collisionData[idx].collisionRemainder = select - float(collisionData[idx].nSelect);

}


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
	float deltaT,
	particle*** cellToParticleMap)
{

	cell* deviceCellData;
	collisionInfo* deviceCollisionData;

	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCellData, cellDataSize * sizeof(cell));
	if (cudaStatus != cudaSuccess) {
		printf("Pcol malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaMalloc((void**)&deviceCollisionData, cellDataSize * sizeof(collisionInfo));
	if (cudaStatus != cudaSuccess) {
		printf("Pcol malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaMemcpy(deviceCellData, cellData, cellDataSize * sizeof(cell), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaMemcpy(deviceCollisionData, collisionData, cellDataSize * sizeof(collisionInfo), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	int numberOfBlocks = ceil(double(cellDataSize) / double(MAX_THREAD_PER_BLOCK));

	preCollisionKernel<<<numberOfBlocks , MAX_THREAD_PER_BLOCK >>>(deviceCellData, deviceCollisionData, cellDataSize, nsample, deltaT, cellvol, pnum);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Pcol launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaMemcpy(collisionData, deviceCollisionData, cellDataSize * sizeof(collisionInfo), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol memcpy to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	// Loop over cells and select particles to perform collisions
	for (int cellIndex = 0; cellIndex < cellDataSize; ++cellIndex)
	{
		if (collisionData[cellIndex].nSelect > 0)
		{ // selected particles for collision
			if (cellData[cellIndex].currentNumberOfParticles < 2)
			{ // if not enough particles for collision, wait until
			  // we have enough
				collisionData[cellIndex].collisionRemainder += collisionData[cellIndex].nSelect;
			}
			else
			{
				// Select nselect particles for possible collision
				float cmax = collisionData[cellIndex].maxCollisionRate;
				for (int c = 0; c < collisionData[cellIndex].nSelect; ++c)
				{
					// select two points in the cell
					int pt1 = min(int(floor(ranf() * cellData[cellIndex].currentNumberOfParticles)), cellData[cellIndex].currentNumberOfParticles - 1);
					int pt2 = min(int(floor(ranf() * cellData[cellIndex].currentNumberOfParticles)), cellData[cellIndex].currentNumberOfParticles - 1);

					// Make sure they are unique points
					while (pt1 == pt2) {
						pt2 = min(int(floor(ranf() * cellData[cellIndex].currentNumberOfParticles)), cellData[cellIndex].currentNumberOfParticles - 1);
					}
					// Compute the relative velocity of two particles
					vect3d v1 = cellToParticleMap[cellIndex][pt1]->velocity;
					vect3d v2 = cellToParticleMap[cellIndex][pt2]->velocity;
					vect3d vr = v1 - v2;
					float vrm = norm(vr);

					// Compute collision  rate for hard sphere model
					float crate = sigmak * vrm;
					if (crate > cmax) {
						cmax = crate;
					}
					
					// Check if these particles actually collide
					if (ranf() < crate / collisionData[cellIndex].maxCollisionRate)
					{
						// Collision Accepted, adjust particle velocities
						// Compute center of mass velocity, vcm
						vect3d vcm = .5 * (v1 + v2);
						// Compute random perturbation that conserves momentum
						vect3d vp = randomDir() * vrm;

						// Adjust particle velocities to reflect collision
						cellToParticleMap[cellIndex][pt1]->velocity = vcm + 0.5 * vp;
						cellToParticleMap[cellIndex][pt2]->velocity = vcm - 0.5 * vp;

						cellToParticleMap[cellIndex][pt1]->status = 2;
						cellToParticleMap[cellIndex][pt2]->status = 2;
					}
				}
				// Update the maximum collision rate to be used in future timesteps
				// for determining number of particles to select.
				collisionData[cellIndex].maxCollisionRate = cmax;
			}
		}
	
	}


	cudaStatus = cudaFree(deviceCellData);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol free cell failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaFree(deviceCollisionData);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol free cell failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

}