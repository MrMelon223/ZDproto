#ifndef RENDERING_CUH
#define RENDERING_CUH

#include "Helper.cuh"
#include "Light.h"
#include "Ray.h"
#include "Crosshair.h"


__device__
void transform();

__device__
bool ray_intersects_box(glm::vec3, glm::vec3, glm::vec3&, glm::vec3&, int&);

__device__
glm::vec4 sample_texture(glm::vec4*, glm::ivec2, float, float);

__device__
glm::vec4 interpolateColor3D(const glm::vec4& c1, const glm::vec4& c2, const glm::vec4& c3,
    float alpha, float beta, float gamma);

__global__
void setup_rays(glm::vec3, glm::vec3, float, Ray*, glm::ivec2);

__global__
void set_visible_tris(glm::vec3, glm::vec3, glm::vec2, d_Model*, d_ModelInstance*, uint32_t);

__global__
void capture_with_rays(glm::vec3, glm::vec3, float, d_ModelInstance*, uint32_t, Ray*, glm::ivec2, d_Model*);

__global__
void calculate_lighting(d_AmbientLight*, d_PointLight*, uint32_t, Ray*, glm::ivec2, glm::vec4*);

__global__
void texture_map(Ray*, glm::ivec2, glm::vec4*);

__global__
void draw_crosshair(glm::ivec2, glm::vec4*, Crosshair, bool, bool);

#endif