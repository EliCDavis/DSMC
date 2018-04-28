#pragma once
#include "vect3d.h"

// Data structure for holding particle information
struct particle
{
	// particle position and velocity
	vect3d position, velocity; 
	
	// index of containing cell
	int index;

	/*
		-1 - marked for deletion
		 0 - fine
		 1 - hit plate
		 2 - hit particle
	*/
	int status;

	particle(vect3d pos, vect3d vel) {
		position = pos;
		velocity = vel;
		index = 0;
		status = 0;
	}
};