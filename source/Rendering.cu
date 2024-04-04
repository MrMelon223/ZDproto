	// Rendering.cu
#include "../include/Rendering.cuh"
#include "../include/Camera.h"

__device__
glm::vec4 interpolateColor3D(const glm::vec4& c1, const glm::vec4& c2, const glm::vec4& c3,
	float alpha, float beta, float gamma) {
	float r = alpha * c1.x + beta * c2.x + gamma * c3.x;
	float g = alpha * c1.y + beta * c2.y + gamma * c3.y;
	float b = alpha * c1.z + beta * c2.z + gamma * c3.z;
	return glm::vec4(r, g, b, 1.0f);
}

__device__
glm::vec4 sample_texture(glm::vec4* texture, glm::ivec2 dims, float x, float y) {
	if (y >= 0 && x >= 0) {
		return texture[static_cast<int>(y * dims.y) * dims.x + static_cast<int>(x * dims.x)];
	}
	return glm::vec4(0.0f);
}

void Camera::capture(d_ModelInstance* instances, uint32_t instance_count, d_Model* models, glm::vec4* buffer) {
	setup_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->fov.x, this->d_ray_matrix[0], this->dims);
	for (uint8_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		capture_with_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->fov.x, instances, instance_count, this->d_ray_matrix[i], this->dims, models);
		cudaDeviceSynchronize();
	}

	texture_map << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->d_ray_matrix[0], this->dims, buffer);
	cudaDeviceSynchronize();
}

__global__
void setup_rays(glm::vec3 position, glm::vec3 direction, float horizontal_fov, Ray* rays, glm::ivec2 dims) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	if (!(x >= dims.x && x < 0) && !(x >= dims.y && y < 0)) {
		Ray* ray = &rays[idx];

		//ray->position = position;

		float ratio = static_cast<float>(dims.x) / static_cast<float>(dims.y);
		float norm_x = (x - (static_cast<float>(dims.x) * 0.5f)) / (static_cast<float>(dims.x) * 0.5f);
		float norm_y = (y - (static_cast<float>(dims.y) * 0.5f)) / (static_cast<float>(dims.y) * 0.5f);
		float fov_rad = horizontal_fov * (PI / 180.0f);
		float half_fov = fov_rad * 0.5f;

		glm::vec3 right = glm::cross(direction, glm::vec3(0.0f, 1.0f, 0.0f));

		right = glm::normalize(right);

		glm::vec3 up = glm::cross(right, direction);

		up = glm::normalize(up);

		ray->direction = direction + norm_x * half_fov * ratio * right + norm_y * half_fov * up;
	}
}

__global__
void capture_with_rays(glm::vec3 position, glm::vec3 direction, float horizontal_fov, d_ModelInstance* instances, uint32_t instance_count, Ray* rays, glm::ivec2 dims, d_Model* models) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	if (!(x >= dims.x && x < 0) && !(x >= dims.y && y < 0)) {
		Ray* ray = &rays[idx];

		ray->position = position;
		bool intersected = false, tried = false;
		float last_leng = 10.0f;

		for (int j1 = 0; j1 < instance_count; j1++) {
			d_ModelInstance* g = &instances[j1];
			//printf("ModelIndex = %i\n", g->model_index);
			uint32_t c = *(models[g->model_index - 1].triangle_count);
			for (int k = 0; k < c; k++) {
				tried = true;
				//if (instances[j1].model->tri_visible[k]) {
				Tri* t = &models[instances[j1].model_index - 1].triangles[k];
				Vertex* vs = models[instances[j1].model_index - 1].vertices;
				glm::vec3 offset = instances[j1].position;

				glm::vec2 intersection;
				float d;
				bool intersection_detection = glm::intersectRayTriangle(ray->position, ray->direction, vs[t->a].position + offset, vs[t->b].position + offset, vs[t->c].position + offset, intersection, d);
				if (intersection_detection) {
					glm::vec3 intersect = (d * direction) + position;
					float tr = d;// (intersect - position).length();
					if (tr < last_leng) {
						//printf("Intersection true!\n");

						intersect = ray->position + tr * ray->direction;
						glm::vec3 diff = intersect - position;

						glm::vec3 bayo_coord = glm::inverse(glm::mat3(vs[t->b].position + offset - vs[t->a].position + offset, vs[t->c].position + offset - vs[t->a].position + offset, intersect - vs[t->a].position + offset)) * (intersect - vs[t->a].position + offset);

						glm::vec4 fin_color = interpolateColor3D(vs[t->a].color, vs[t->b].color, vs[t->c].color, bayo_coord.x, bayo_coord.y, bayo_coord.z);
						//bayo_coord = glm::clamp(bayo_coord, 0.0f, 1.0f);

						glm::vec2 uv = bayo_coord.x * vs[t->a].uv + bayo_coord.y * vs[t->b].uv + bayo_coord.z * vs[t->c].uv;

						uv = glm::clamp(uv, 0.0f, 1.0f);

						//printf("UV Coords: {%.2f, %.2f}\n", uv.x, uv.y);
						ray->payload.color = fin_color;
						ray->payload.intersection = intersect;
						ray->payload.uv = uv;
						ray->intersected = true;
						ray->payload.model = &models[g->model_index - 1];
						intersected = true;
						last_leng = tr;
					}
				}
			cont:;
				continue;
			}
		}
		if (!intersected && tried) {
			ray->payload.color = glm::vec4(0.1f, 0.5f, 1.0f, 1.0f);
			ray->intersected = false;
		}
	}
}

__global__
void texture_map(Ray* rays, glm::ivec2 dims, glm::vec4* out) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	Ray* ray = &rays[idx];
	if (ray->intersected) {
		out[idx] = sample_texture(ray->payload.model->color_map->data, ray->payload.model->color_map->dims, ray->payload.uv.x, ray->payload.uv.y);
	}
	else {
		out[idx] = ray->payload.color;
	}
}