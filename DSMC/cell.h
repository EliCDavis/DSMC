#pragma once
#include "vect3d.h"

struct cell
{
	// total number of particles sampled
	int numberOfParticles; 
	
	// total velocity vector
	vect3d velocity;     
	
	// total kinetic energy of particles
	float energy;   
};
