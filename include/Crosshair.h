#ifndef CROSSHAIR_H
#define CROSSHAIR_H

#include "Helper.cuh"

struct Crosshair {
	float walk_radius;
	float run_radius;
	glm::vec4 color;
};

#endif
