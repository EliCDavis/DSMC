#pragma once

// Information that is used to control the collision probability code
struct collisionInfo
{
	// Maximum collision rate seen for this cell so far in the simulation
	float maxCollisionRate;

	// Non-integral fraction of collisions that remain to be performed
	// and are carried over into the next timestep
	float collisionRemainder;

	// Number of collisions at current time step
	int nSelect;
};