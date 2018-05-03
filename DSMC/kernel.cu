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


cudaError_t initializeCuda(curandState_t** states, int blocks, cell** deviceCells, int numCells);

cudaError_t inflowPotentialParticles(curandState_t* randomStates, particle* particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp);

cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions, vect3d dividedCellDimensions, int* numToDelete);

particle* removeParticlesOutofBounds(particle* particles, int size, int newSize);

cudaError_t clearCellInformation(cell* cells, int numCells);

void sampleCellInformation(particle* particles, int numOfParticles, cell* cells, int numCells, int* cellSteal, bool* thingsHaveChanged);

__device__ void swapParticles(particle* p1, particle* p2) {
	int tempIndex = p1->index;
	bool tempHitParticle = p1->hitParticle;
	bool tempHitPlate = p1->hitPlate;
	bool tempDelete = p1->deleteMe;
	
	float tempVelX = p1->velocity.x;
	float tempVelY = p1->velocity.y;
	float tempVelZ = p1->velocity.z;
	
	float tempPosX = p1->position.x;
	float tempPosY = p1->position.y;
	float tempPosZ = p1->position.z;
	
	p1->index = p2->index;
	p1->hitParticle = p2->hitParticle;
	p1->hitPlate = p2->hitPlate;
	p1->deleteMe = p2->deleteMe;

	p1->velocity.x = p2->velocity.x;
	p1->velocity.y = p2->velocity.y;
	p1->velocity.z = p2->velocity.z;

	p1->position.x = p2->position.x;
	p1->position.y = p2->position.y;
	p1->position.z = p2->position.z;

	p2->index = tempIndex;
	p2->hitParticle = tempHitParticle;
	p2->hitPlate = tempHitPlate;
	p2->deleteMe = tempDelete;

	p2->velocity.x = tempVelX;
	p2->velocity.y = tempVelY;
	p2->velocity.z = tempVelZ;

	p2->position.x = tempPosX;
	p2->position.y = tempPosY;
	p2->position.z = tempPosZ;
}

__global__ void bitonic_sort_step(particle *dev_values, int j, int k, int numParticles)
{
	unsigned int i, ixj; /* Sorting partners: i and ixj */
	i = blockDim.x * blockIdx.x + threadIdx.x;
	ixj = i^j;
	
	if (i >= numParticles || ixj >= numParticles) {
		return;
	}

	/* The threads with the lowest ids sort the array. */
	if ((ixj)>i) {
		if (((i&k) == 0 && dev_values[i].index > dev_values[ixj].index) || (((i&k) != 0) && dev_values[i].index < dev_values[ixj].index)) {
			swapParticles(&dev_values[i], &dev_values[ixj]);
		}
	}
}

/**
* Inplace bitonic sort using CUDA.
*/
void bitonic_sort_particles(particle *values, int numParticles)
{
	cudaError_t status;
	int numberOfBlocks = ceil(double(numParticles) / double(MAX_THREAD_PER_BLOCK));

	particle *dev_values;
	size_t size = numParticles * sizeof(particle);

	status = cudaMalloc((void**)&dev_values, size);
	status = cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

	int j, k;
	/* Major step */
	for (k = 2; k <= numParticles; k <<= 1) {
		/* Minor step */
		for (j = k >> 1; j>0; j = j >> 1) {
			bitonic_sort_step <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(dev_values, j, k, numParticles);
			status = cudaGetLastError();
		}
	}
	status = cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
	status = cudaFree(dev_values);
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
	curandGenerator_t cudaRandomHostGenerator);

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
		int status = particles[p].hitPlate;
		myfile << particles[p].position.x << " " << particles[p].position.y << " " << particles[p].position.z << " " << status << endl;
	}
	myfile.close();
}

int main()
{
	int meanParticlePerCell = 10;
	vect3d cellDimensions = vect3d(32, 32, 32);
	vect3d dividedCellDimensions = vect3d(2. / cellDimensions.x, 2. / cellDimensions.y, 2. / cellDimensions.z);
	float vmean = 1;
	float Mach = 20;
	float vtemp = vmean / Mach;
	float deltax = 2. / float(fmax(fmax(cellDimensions.x, cellDimensions.y), cellDimensions.z));
	float deltaT = .1 * deltax / (vmean + vtemp);
	float density = 1e30; // Number of molecules per unit cube of space

	// Initialize cuda random generator so host can create random numbers
	curandGenerator_t cudaRandomHostGenerator;
	curandCreateGenerator(&cudaRandomHostGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(cudaRandomHostGenerator, 1234ULL);

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

	cell* deviceCellSamples;
	

	collisionInfo* collisionData = (collisionInfo*)malloc(numberOfCells * sizeof(collisionInfo));
	initializeCollision(collisionData, numberOfCells, vtemp);

	curandState_t* deviceRandomInflowStates = NULL;
	
	cudaError_t cudaStatus = initializeCuda(&deviceRandomInflowStates, numberOfInflowParticlesEachStep, &deviceCellSamples, numberOfCells);
	if (cudaStatus != cudaSuccess) {
		printf("init kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return 1;
	}

	particle *allParticles = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));
	particle *inflowParticleList = (particle*)malloc(numberOfInflowParticlesEachStep * sizeof(particle));

	clock_t totalTime;

	clock_t clockTime;

	int* deviceCellSteals;
	cudaStatus = cudaMalloc((void**)&deviceCellSteals, numberOfCells * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("sample malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	bool* deviceContentsChanged;
	cudaStatus = cudaMalloc((void**)&deviceContentsChanged, sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		printf("sample malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}
	totalTime = clock();
	for (int t = 0; t < numberOfTimesteps; t++) {

		clockTime = clock();

		cudaStatus = inflowPotentialParticles(deviceRandomInflowStates, inflowParticleList, cellDimensions, meanParticlePerCell, vmean, vtemp);
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

		int numToDelete = 0;
		moveAndIndexParticles(allParticles, currentNumberOfParticles, deltaT, cellDimensions, dividedCellDimensions, &numToDelete);

		// Clean up list of particles out of bounds and recompute cell map
		particle* cleanedParticles = removeParticlesOutofBounds(allParticles, currentNumberOfParticles, currentNumberOfParticles - numToDelete);
		currentNumberOfParticles -= numToDelete;
		free(allParticles);
		allParticles = cleanedParticles;

		if (t % sample_reset == 0)
		{
			clearCellInformation(deviceCellSamples, numberOfCells);
			nsample = 0;
		}
		nsample++;
		
		if (t % 37 == 0) {
			bitonic_sort_particles(allParticles, currentNumberOfParticles);
		}

		sampleCellInformation(allParticles, currentNumberOfParticles, deviceCellSamples, numberOfCells, deviceCellSteals, deviceContentsChanged);

		collideParticles(allParticles,
			currentNumberOfParticles,
			collisionData,
			deviceCellSamples,
			numberOfCells,
			nsample,
			numberOfCells,
			deltaT,
			cudaRandomHostGenerator);

		clockTime = clock() - clockTime;
		printf("%d %f %d\n", t, ((double)clockTime) / CLOCKS_PER_SEC, currentNumberOfParticles);
	}

	printf("Total Time: %f", ((double)clock() - totalTime) / CLOCKS_PER_SEC);

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

__global__ void initRandomStatesKernel(unsigned int seed, curandState_t* states) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	/* we have to initialize the state */
	curand_init(seed, /* the seed controls the sequence of random values that are produced */
		idx, /* the sequence number is only important with multiple cores */
		0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
		&states[idx]);
}

__global__ void initCellsKenel(cell* cells, int numCells) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numCells) {
		return;
	}
	cells[idx].currentNumberOfParticles = 0;
	cells[idx].energy= 0;
	cells[idx].numberOfParticles = 0;
	cells[idx].velocity.x = 0;
	cells[idx].velocity.y = 0;
	cells[idx].velocity.z = 0;
}

cudaError_t initializeCuda(curandState_t **randomInflowStates, int numOfStates, cell** deviceCells, int numCells)
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

	initRandomStatesKernel <<<NUM_OF_BLOCKS, blockSize >>>(2, dev_states);
	cudaError = cudaGetLastError();
	if (cudaError != cudaSuccess) {
		printf("init kernel launch failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	*randomInflowStates = dev_states;

	cell* devcells;

	cudaError = cudaMalloc((void**)&devcells, numCells * sizeof(cell));
	if (cudaError != cudaSuccess) {
		printf("init malloc failed: %s\n", cudaGetErrorString(cudaError));
		return cudaError;
	}

	int numberOfBlocks = ceil(double(numCells) / double(MAX_THREAD_PER_BLOCK));

	initCellsKenel<<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(devcells, numCells);

	*deviceCells = devcells;

	return cudaGetLastError();
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
	particles[idx].deleteMe = false;
	particles[idx].hitPlate = false;
	particles[idx].hitParticle= false;
}

/* 
 Fill an array with new random particles to be exeucuted on
 */
cudaError_t inflowPotentialParticles(curandState_t* deviceRandomStates, particle *particleList, vect3d cellDimensions, int meanParticlePerCell, float vmean, float vtemp) {
	
	int numOfPoints = cellDimensions.y * cellDimensions.z * meanParticlePerCell;

	int numberOfBlocks = ceil(double(numOfPoints) / double(MAX_THREAD_PER_BLOCK));

	int size = numOfPoints * sizeof(particle);

	particle *dev_a;
	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, size);
	if (cudaStatus != cudaSuccess) {
		printf("inflow malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	inflowKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(numOfPoints, deviceRandomStates, dev_a, cellDimensions.x, cellDimensions.y, cellDimensions.z, vmean, vtemp);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("inflow launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(particleList, dev_a, size, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("inflow memcpy failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaFree(dev_a);
}



/* =========================== 2. MOVE PARTICLES =========================== */

__global__ void moveParticlesKernel(int* deviceDeletionCount, particle* particles, int numParticles, float deltaTime, int dimX, int dimY, int dimZ, float divX, float divY, float divZ) {
	
	extern __shared__ int sdata[];
	sdata[threadIdx.x] = 0;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numParticles) {
		return;
	}

	float newPosX = particles[idx].position.x + (particles[idx].velocity.x * deltaTime);
	float newPosY = particles[idx].position.y + (particles[idx].velocity.y * deltaTime);
	float newPosZ = particles[idx].position.z + (particles[idx].velocity.z * deltaTime);


	// Where did it actually pass through the plate..
	double t = (particles[idx].position.x - PLATE_X) / (particles[idx].position.x - newPosX);
	float pointOfCollisionY = (particles[idx].position.y * (1.0 - t)) + (newPosY * t);
	float pointOfCollisionZ = (particles[idx].position.z * (1.0 - t)) + (newPosZ * t);

	// Actually collided..
	if (((particles[idx].position.x < PLATE_X && newPosX > PLATE_X) ||
		(particles[idx].position.x > PLATE_X && newPosX < PLATE_X)) && (pointOfCollisionY < PLATE_DY && pointOfCollisionY > -PLATE_DY) &&
		(pointOfCollisionZ < PLATE_DZ && pointOfCollisionZ > -PLATE_DZ))
	{
		newPosX = newPosX - 2 * (newPosX - PLATE_X);
		particles[idx].velocity.x = -particles[idx].velocity.x;
		particles[idx].hitPlate = true;
	}


	newPosY = newPosY + (2.0 * ((newPosY < -1) - (newPosY > 1)));
	newPosZ = newPosZ + (2.0 * ((newPosZ < -1) - (newPosZ > 1)));

	// Assign particle positions and index
	particles[idx].position.x = newPosX;
	particles[idx].position.y = newPosY;
	particles[idx].position.z = newPosZ;

	int i = int(fmin(floor((newPosX + 1.0) / divX), double(dimX - 1)));
	int j = int(fmin(floor((newPosY + 1.0) / divY), double(dimY - 1)));
	int k = int(fmin(floor((newPosZ + 1.0) / divZ), double(dimZ - 1)));
	particles[idx].index = i * dimY * dimZ + j * dimZ + k;

	int deletion = 0;
	if (newPosX > 1.0 || newPosX < -1.0) {
		deletion = 1;
		particles[idx].deleteMe = true;
	}

	sdata[threadIdx.x] = deletion;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		deviceDeletionCount[blockIdx.x] = sdata[0];
	}
}

/*
	Move particles appropriately and marks those out of bounds with a flag for deletion.
	Reindexes particles not marked for deletion
*/
cudaError_t moveAndIndexParticles(particle* particleList, int numOfParticles, float deltaTime, vect3d cellDimensions, vect3d dividedCellDimensions, int* numToDelete) {
	int numberOfBlocks = ceil(double(numOfParticles) / double(MAX_THREAD_PER_BLOCK));
	
	particle *dev_a;

	int *deviceDeletionCount;

	cudaError_t cudaStatus = cudaMalloc((void**)&dev_a, numOfParticles*sizeof(particle));
	if (cudaStatus != cudaSuccess) {
		printf("move&index malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMalloc((void**)&deviceDeletionCount, numberOfBlocks * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("move&index malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(dev_a, particleList, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("move&index memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	moveParticlesKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK * sizeof(int) >>>(deviceDeletionCount, dev_a, numOfParticles, deltaTime, cellDimensions.x, cellDimensions.y, cellDimensions.z, dividedCellDimensions.x, dividedCellDimensions.y, dividedCellDimensions.z);

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

	int* deletionCount = (int*)malloc(numberOfBlocks * sizeof(int));
	cudaStatus = cudaMemcpy(deletionCount, deviceDeletionCount, numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("move&index memcpy to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	for (int i = 0; i < numberOfBlocks; i++) {
		*numToDelete += deletionCount[i];
	}
	free(deletionCount);

	cudaStatus = cudaFree(deviceDeletionCount);
	if (cudaStatus != cudaSuccess) {
		printf("move&index free count failed: %s\n", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	return cudaFree(dev_a);
}

/* ========================== 3. REMOVE PARTICLES ========================== */

particle* removeParticlesOutofBounds(particle* particles, int originalParticleSize, int newSize) {
	particle* newParticleList = (particle*)malloc(newSize * sizeof(particle));
	int added = 0;
	for (int i = 0; i < originalParticleSize; i++) {
		if (!particles[i].deleteMe) {
			newParticleList[added] = particle(particles[i]);
			added += 1;
		}
	}
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

cudaError_t	clearCellInformation(cell* deviceCells, int numCells) {
	int numberOfBlocks = ceil(double(numCells) / double(MAX_THREAD_PER_BLOCK));

	clearCellsKernel <<<numberOfBlocks, MAX_THREAD_PER_BLOCK >>>(deviceCells, numCells);

	return cudaGetLastError();
}

__global__ void initializeStolenKernel(bool* stolenBefore, int numStolen) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numStolen) {
		return;
	}
	stolenBefore[idx] = false;
}

__global__ void sameplCellClearKernel(cell* cells, int numCells) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numCells) {
		return;
	}
	cells[idx].currentNumberOfParticles = 0;
}

__global__ void sameplCellStealKernel(particle* particles, int particleSize, int* cellSteals, bool* stolenBefore, bool* changed) {
	*changed = false;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= particleSize || stolenBefore[idx]) {
		return;
	}
	cellSteals[particles[idx].index] = idx;
}

__global__ void sameplCellRunKernel(particle* particles, int particleSize, int* cellSteals, cell* cells, bool* stolenBefore, bool* changed) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= particleSize) {
		return;
	}
	int cellIndex = particles[idx].index;

	// We didn't succesfully grab the cell, try again next time
	if (cellSteals[cellIndex] != idx) {
		return;
	}

	cells[cellIndex].currentNumberOfParticles++;
	cells[cellIndex].numberOfParticles++;
	cells[cellIndex].velocity.x += particles[idx].velocity.x;
	cells[cellIndex].velocity.y += particles[idx].velocity.y;
	cells[cellIndex].velocity.z += particles[idx].velocity.z;
	cells[cellIndex].energy  += (.5 *
		((particles[idx].velocity.x * particles[idx].velocity.x) +
		 (particles[idx].velocity.y * particles[idx].velocity.y) +
		 (particles[idx].velocity.z * particles[idx].velocity.z)));

	cellSteals[cellIndex] = -1;
	stolenBefore[idx] = true;
	*changed = true;
}


void sampleCellInformation(particle* particles, int numOfParticles, cell* deviceCells, int numCells, int* deviceCellsStolen, bool* deviceThingsChanged) {

	int numberOfBlocksForParticles = ceil(double(numOfParticles) / double(MAX_THREAD_PER_BLOCK));
	int numberOfBlocksForCells = ceil(double(numCells) / double(MAX_THREAD_PER_BLOCK));

	particle *deviceParticles;
	cudaError_t cudaStatus = cudaMalloc((void**)&deviceParticles, numOfParticles * sizeof(particle));
	if (cudaStatus != cudaSuccess) {
		printf("sample malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return ;
	}

	bool* deviceParticleSuccessStolen;
	cudaStatus = cudaMalloc((void**)&deviceParticleSuccessStolen, numOfParticles * sizeof(bool));
	if (cudaStatus != cudaSuccess) {
		printf("sample malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	initializeStolenKernel <<<numberOfBlocksForParticles, MAX_THREAD_PER_BLOCK>>> (deviceParticleSuccessStolen, numOfParticles);

	cudaStatus = cudaMemcpy(deviceParticles, particles, numOfParticles * sizeof(particle), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("sample memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	sameplCellClearKernel <<<numberOfBlocksForCells, MAX_THREAD_PER_BLOCK>>> (deviceCells, numCells);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("sample kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	bool thingsChanged = true;
	int times = 0;
	while (thingsChanged) {
		sameplCellStealKernel <<<numberOfBlocksForParticles, MAX_THREAD_PER_BLOCK>>>(deviceParticles, numOfParticles, deviceCellsStolen, deviceParticleSuccessStolen, deviceThingsChanged);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("sample kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		sameplCellRunKernel <<<numberOfBlocksForParticles, MAX_THREAD_PER_BLOCK>>>(deviceParticles, numOfParticles, deviceCellsStolen, deviceCells, deviceParticleSuccessStolen, deviceThingsChanged);
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			printf("sample kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return;
		}

		if (times >= 3) {
			cudaStatus = cudaMemcpy(&thingsChanged, deviceThingsChanged, sizeof(bool), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				printf("sample copy failed: %s\n", cudaGetErrorString(cudaStatus));
				return;
			}
			times = 0;
		}
		times++;
	}

	cudaStatus = cudaFree(deviceParticles);
	if (cudaStatus != cudaSuccess) {
		printf("sample free failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaFree(deviceParticleSuccessStolen);
	if (cudaStatus != cudaSuccess) {
		printf("sample free failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

}


/* ============================ Cell Collision ============================= */

__global__ void preCollisionKernel(cell* cells, collisionInfo* collisionData, int cellSize, float nSample, float deltaT, float cellVol, float moleculesPerParticle, int *numOfRandomNumbers) {

	extern __shared__ int sdata[];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[threadIdx.x] = 0;

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

	sdata[threadIdx.x] = collisionData[idx].nSelect * (n_instant >= 2) * 5;
	__syncthreads();

	for (unsigned int s = blockDim.x / 2; s>0; s >>= 1) {
		if (threadIdx.x < s) {
			sdata[threadIdx.x] += sdata[threadIdx.x + s];
		}
		__syncthreads();
	}
	if (threadIdx.x == 0) {
		numOfRandomNumbers[blockIdx.x] = sdata[0];
	}

}


// Computes a unit vector with a random orientation and uniform distribution
inline vect3d randomDir(float rndOne, float rndTwo)
{
	double B = 2. * rndOne - 1;
	double A = sqrt(1. - B * B);
	double theta = rndTwo * 2 * CUDART_PI_F;
	return vect3d(B, A * cos(theta), A * sin(theta));
}

void collideParticles(
	particle* particleList,
	int particleListSize,
	collisionInfo* collisionData,
	cell* deviceCellData,
	int cellDataSize,
	int nsample, 
	float cellvol, 
	float deltaT,
	curandGenerator_t cudaRandomHostGenerator)
{

	// Compute number of particles per cell and compute a set of pointers
	// from each cell to the corresponding particles
	int* np = (int*)malloc(cellDataSize * sizeof(int));
	int* cnt = (int*)malloc(cellDataSize * sizeof(int));
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
	int* offsets = (int*)malloc((cellDataSize + 1) * sizeof(int));
	offsets[0] = 0;
	for (int i = 0; i < cellDataSize; ++i)
	{
		offsets[i + 1] = offsets[i] + np[i];
	}

	// pmap is a structure of pointers from cells to particles, note
	// since there may be many particles per cell, the offsets need to
	// be used to access particles from this data structure.
	particle** pmap = (particle**)malloc(offsets[cellDataSize] * sizeof(particle*));
	for (int p = 0; p < particleListSize; p++)
	{
		int i = particleList[p].index;
		pmap[cnt[i] + offsets[i]] = &(particleList[p]);
		cnt[i]++;
	}

	free(cnt);
	
	collisionInfo* deviceCollisionData;
	cudaError_t cudaStatus = cudaMalloc((void**)&deviceCollisionData, cellDataSize * sizeof(collisionInfo));
	if (cudaStatus != cudaSuccess) {
		printf("Pcol malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	int* deviceRandomNumbersNeeded;
	cudaStatus = cudaMalloc((void**)&deviceRandomNumbersNeeded, cellDataSize * sizeof(int));
	if (cudaStatus != cudaSuccess) {
		printf("Pcol malloc failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	cudaStatus = cudaMemcpy(deviceCollisionData, collisionData, cellDataSize * sizeof(collisionInfo), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol memcpy to device failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	int numberOfBlocks = ceil(double(cellDataSize) / double(MAX_THREAD_PER_BLOCK));

	preCollisionKernel<<<numberOfBlocks, MAX_THREAD_PER_BLOCK, MAX_THREAD_PER_BLOCK * sizeof(int)>>>(deviceCellData, deviceCollisionData, cellDataSize, nsample, deltaT, cellvol, pnum, deviceRandomNumbersNeeded);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		printf("Pcol launch failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	int* blockAggRandomNumbersNeeded = (int*)malloc(numberOfBlocks * sizeof(int));
	cudaStatus = cudaMemcpy(blockAggRandomNumbersNeeded, deviceRandomNumbersNeeded, numberOfBlocks * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol memcpy to host failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

	int randomNumbersNeeded = 0;
	for (int i = 0; i < numberOfBlocks; i++) {
		randomNumbersNeeded += blockAggRandomNumbersNeeded[i];
	}
	free(blockAggRandomNumbersNeeded);

	float* randomNumbers = (float *)malloc(randomNumbersNeeded * sizeof(float));
	float* deviceRandomNumbers;
	cudaStatus = cudaMalloc((void **)&deviceRandomNumbers, randomNumbersNeeded * sizeof(float));
	
	curandGenerateUniform(cudaRandomHostGenerator, deviceRandomNumbers, randomNumbersNeeded);
	cudaStatus = cudaMemcpy(randomNumbers, deviceRandomNumbers, randomNumbersNeeded * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaFree(deviceRandomNumbers);
	int randomNumbersUsed = 0;

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
			if (np[cellIndex] < 2)
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
					int pt1 = min(int(floor(randomNumbers[randomNumbersUsed] * np[cellIndex])), np[cellIndex] - 1);
					int pt2 = min(int(floor(randomNumbers[randomNumbersUsed + 1] * np[cellIndex])), np[cellIndex]- 1);

					// Make sure they are unique points
					while (pt1 == pt2) {
						pt2 = min(int(floor(ranf() * np[cellIndex])), np[cellIndex]- 1);
					}
					// Compute the relative velocity of two particles
					vect3d v1 = pmap[offsets[cellIndex] + pt1]->velocity;
					vect3d v2 = pmap[offsets[cellIndex] + pt2]->velocity;
					vect3d vr = v1 - v2;
					float vrm = norm(vr);

					// Compute collision  rate for hard sphere model
					float crate = sigmak * vrm;
					if (crate > cmax) {
						cmax = crate;
					}
					
					// Check if these particles actually collide
					if (randomNumbers[randomNumbersUsed + 2] < crate / collisionData[cellIndex].maxCollisionRate)
					{
						// Collision Accepted, adjust particle velocities
						// Compute center of mass velocity, vcm
						vect3d vcm = .5 * (v1 + v2);
						// Compute random perturbation that conserves momentum
						vect3d vp = randomDir(randomNumbers[randomNumbersUsed + 3], randomNumbers[randomNumbersUsed + 4]) * vrm;

						// Adjust particle velocities to reflect collision
						pmap[offsets[cellIndex] + pt1]->velocity = vcm + 0.5 * vp;
						pmap[offsets[cellIndex] + pt2]->velocity = vcm - 0.5 * vp;

						pmap[offsets[cellIndex] + pt1]->hitParticle = true;
						pmap[offsets[cellIndex] + pt2]->hitParticle = true;
					}
					randomNumbersUsed += 5;
				}
				// Update the maximum collision rate to be used in future timesteps
				// for determining number of particles to select.
				collisionData[cellIndex].maxCollisionRate = cmax;
			}
		}
	
	}

	free(pmap);
	free(np);
	free(offsets);
	free(randomNumbers);

	cudaStatus = cudaFree(deviceCollisionData);
	if (cudaStatus != cudaSuccess) {
		printf("Pcol free cell failed: %s\n", cudaGetErrorString(cudaStatus));
		return;
	}

}