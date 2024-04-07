#ifndef PRIMITIVES_H
#define PRIMITIVES_H

#include "Helper.cuh"

struct Vertex {
	glm::vec3 position;
	glm::vec3 normal;
	glm::vec4 color;
	glm::vec2 uv;
};

struct Tri {
	uint32_t a, b, c;
	glm::vec3 normal;
};

#endif
