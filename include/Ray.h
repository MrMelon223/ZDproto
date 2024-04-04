#ifndef RAY_H
#define RAY_H

#include "Helper.h"
#include "Model.h"

struct Payload {
	glm::vec2 uv;
	glm::vec3 intersection;
	glm::vec4 color;
	d_Model* model;
};

struct Ray {
	glm::vec3 position;
	glm::vec3 direction;
	bool intersected;
	Payload payload;
};

#endif
