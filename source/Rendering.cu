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
glm::vec3 calculateBarycentric(const glm::vec2& point, const glm::vec2& vertex0, const glm::vec2& vertex1, const glm::vec2& vertex2) {
	glm::vec2 v0 = vertex1 - vertex0;
	glm::vec2 v1 = vertex2 - vertex0;
	glm::vec2 v2 = point - vertex0;

	float dot00 = glm::dot(v0, v0);
	float dot01 = glm::dot(v0, v1);
	float dot02 = glm::dot(v0, v2);
	float dot11 = glm::dot(v1, v1);
	float dot12 = glm::dot(v1, v2);

	float denom = dot00 * dot11 - dot01 * dot01;
	float barycentricY = (dot11 * dot02 - dot01 * dot12) / denom;
	float barycentricZ = (dot00 * dot12 - dot01 * dot02) / denom;
	float barycentricX = 1.0f - barycentricY - barycentricZ;

	return glm::vec3(barycentricX, barycentricY, barycentricZ);
}

__device__
glm::vec4 sample_texture(glm::vec4* texture, glm::ivec2 dims, float x, float y) {
	if (y >= 0.0f && x >= 0.0f) {
		return texture[static_cast<int>(y * dims.y) * dims.x + static_cast<int>(x * dims.x)];
	}
	return glm::vec4(0.0f);
}

void Camera::capture(d_ModelInstance* instances, uint32_t instance_count, d_Model* models, d_AmbientLight* amb_light, d_PointLight* point_lights, uint32_t point_lights_size, glm::vec4* buffer) {
	setup_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->fov.x, this->d_ray_matrix[0], this->dims);
	for (uint8_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		capture_with_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->fov.x, instances, instance_count, this->d_ray_matrix[i], this->dims, models);
		cudaDeviceSynchronize();
	}

	//texture_map << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->d_ray_matrix[0], this->dims, buffer);
	calculate_lighting << < (this->dims.y * this->dims.x) / 128, 128 >> > (amb_light, point_lights, point_lights_size, this->d_ray_matrix[0], this->dims, buffer);
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

	if (!(x >= dims.x || x < 0) && !(y >= dims.y || y < 0)) {
		//printf("ray index = %i\n", idx);
		Ray* ray = &rays[idx];

		ray->position = position;
		bool intersected = false, tried = false;
		float last_leng = 1000.0f;
		//printf("%i # Instances\n", instance_count);
		for (int j1 = 0; j1 < instance_count; j1++) {
			d_ModelInstance* g = &instances[j1];
			//printf("ModelIndex = %i\n", g->model_index);
			//printf("Model Index %i\n", g->model_index);
			uint32_t c = *(models[g->model_index].triangle_count);
			for (int k = 0; k < c; k++) {
				tried = true;
				//if (instances[j1].model->tri_visible[k]) {
				Tri* t = &models[instances[j1].model_index].triangles[k];
				Vertex* vs = models[instances[j1].model_index].vertices;
				glm::vec3 offset = instances[j1].position;

				glm::vec2 intersection;
				float d;
				glm::vec2 uv;
				bool intersection_detection = glm::intersectRayTriangle(ray->position, ray->direction, vs[t->a].position + offset, vs[t->b].position + offset, vs[t->c].position + offset, uv, d);
				if (intersection_detection) {
					glm::vec3 intersect = (d * direction) + position;
					float tr = d;//(intersect - position).length();
					if (tr < last_leng && tr >= 0.1f) {
						//printf("Intersection true!\n");

						intersect = ray->position + tr * ray->direction;
						glm::vec3 diff = intersect - position;

						//bayo_coord = glm::clamp(bayo_coord, 0.0f, 1.0f);

						glm::mat3 a = glm::mat3(glm::vec3(vs[t->a].uv, 1.0f), glm::vec3(vs[t->b].uv, 1.0f), glm::vec3(vs[t->c].uv, 1.0f));

						glm::mat3 a_inv = glm::inverse(a);
						glm::vec3 barycentric = calculateBarycentric(uv, vs[t->a].uv, vs[t->b].uv, vs[t->c].uv);

						//printf("Barycentric Coords: {%.2f, %.2f, %.2f}\n", barycentric.x, barycentric.y, barycentric.z);

						//uv = (barycentric.x * vs[t->a].uv + barycentric.y * vs[t->b].uv + barycentric.z * vs[t->c].uv) * uv;
						uv = uv;
						//uv = glm::vec2(barycentric.x, barycentric.y);

						//uv /= 10.0f;

						uv = glm::clamp(uv, 0.0f, 1.0f);

						//printf("UV Coords: {%.2f, %.2f}\n", uv.x, uv.y);
						ray->payload.color = glm::vec4(1.0f);
						ray->payload.intersection = intersect;
						ray->payload.uv = uv;
						ray->intersected = true;
						ray->payload.model = &models[g->model_index];
						ray->payload.triangle = t;
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
void calculate_lighting(d_AmbientLight* amb, d_PointLight* lights, uint32_t lights_size, Ray* rays, glm::ivec2 dims, glm::vec4* out) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	out[idx] = glm::vec4(0.0f);

	Ray* ray = &rays[idx];
	if (ray->intersected) {
		glm::vec3 result = glm::vec3(0.0f);
		glm::vec4 diffuse_color = sample_texture(ray->payload.model->color_map->data, ray->payload.model->color_map->dims, ray->payload.uv.x, ray->payload.uv.y);

		glm::vec4 lighting = glm::vec4(0.0f);
		bool lit = false;
		float cummulative_intensity = 0.0f;
		for (int l = 0; l < lights_size; l++) {

			float amb_strength = 0.1f;
			glm::vec3 ambient = amb->intensity * amb->diffuse_color;
			glm::vec3 light_direction = lights[l].position - ray->payload.intersection;
			float distance = light_direction.length();
			float diff = glm::max(glm::dot(ray->payload.triangle->normal, light_direction), 0.0f);
			float intensity = lights[l].intensity / (distance * distance);
			glm::vec3 diffuse = intensity * lights[l].diffuse_color * diff;

			result += (ambient + diffuse) * glm::vec3(diffuse_color.x, diffuse_color.y, diffuse_color.z);

		}

		out[idx] += glm::vec4(result, 1.0f);

	}
	else {
		out[idx] = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
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