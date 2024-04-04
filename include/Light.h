#ifndef LIGHT_H
#define LIGHT_H

#include "Helper.h"

struct d_PointLight {
	glm::vec3 position;
	glm::vec4 diffuse_color;
	glm::vec4 specular_color;
	float intensity, falloff_distance, range;
};

struct d_AmbientLight {
	glm::vec4 diffuse_color;
	glm::vec4 specular_color;
	float intensity;
};

#endif
