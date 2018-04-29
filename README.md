# Direct Simulation Monte Carlo

*For CSE-8163 Parallel and Distributed Scientific Computing*

Parrallelized version of DSMC program provided By Ed Luke scaled to operate on millions of particles.

## Definitions

**molecule** - something like air. We don't deal with stuff on a per molecule basis

**particle** - A collection of molecules, deal with particle interactions inorder to simulate millions of molecules

**cell** - For computing collisions between particles we grab all particles within a cell and randomly sample

**device** - GPU

**host** - CPU

**Kernel** - function that runs on the device

## Stats

Dealing with 406,736 particles

## Findings / Issues Man

### Balance
Too many blocks is bad. Maxes out graphics card. Find right amount...
Max out amount of allowed threads per block, minimize number of blocks needed

### Sampling Cells
invalid configuration (9)
Have to break into smaller bits or number of particles is going with cell data is going to max out the card.

### Init random seeds in begining

Super increase

### Control logic is badddd

Note:High Priority: Avoid different execution paths within the same warp.

Flow control instructions (if, switch, do, for, while) can significantly affect the instruction throughput by causing threads of the same warp to diverge; that is, to follow different execution paths. If this happens, the different execution paths must be executed separately; this increases the total number of instructions executed for this warp.

https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#control-flow

### ALWAYS CHECK AFTER EVERY KERNEL EXECUTION

A failed kernel 30 steps back might be causing your issue

## Steps

### inflowPotentialParticles

* Create potential particles to be kept for the simulation
* Don't have to create array but once, since it's always same size, just overwrite stuff.

## Notes

### Don't create arrays in the kernel
https://stackoverflow.com/questions/2187189/creating-arrays-in-nvidia-cuda-kernel

### Introduction
https://www.nvidia.com/content/cudazone/download/Getting_Started_w_CUDA_Training_NVISION08.pdf

Lots of great graphs here

Kernel launches grid of **thread blocks**

threads within a block can synchronize, cooperate via shared memory
Shared memory per block is fast.

All __global__ and __device__ functions have
access to these automatically defined variables
dim3 gridDim;
	Dimensions of the grid in blocks (at most 2D)
dim3 blockDim;
	Dimensions of the block in threads
dim3 blockIdx;
	Block index within the grid
dim3 threadIdx;
	Thread index within the block

kernel<<<grid, block>>>(...);

unique thread index = (blockIdx.x * blockDim.x) + threadIdx.x

#### Hosts
Can read and write global memory but not shared memory
Host code manages device memory, applies to global device memory (DRAM)

Synchronization
	All kernel launches are asynchronous where control returns to CPU immediately
	cudaMemcpy() is syncrhonous
	cudaThreadSynchronize() blocks until all pervious CUDA calls complete

#### Function Qualifiers

__global__ - caled from host, executes on device, returns void

__device__ - called from device, runs on device, can't be called from host

### Random Numbers

http://cs.umw.edu/~finlayson/class/fall16/cpsc425/notes/cuda-random.html

https://developer.nvidia.com/curand

### Reductions

http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf


## Things to look into....

cudaMemset

# Next Steps...
[x]. Get main function to obtain generated particles

[x]. Get particles velocity and position to be randomized

[x]. Take into account temperature and average velocity

[x]. Move particles

[5]. Remove them