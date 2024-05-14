	// Rendering.cu
#include "../include/Rendering.cuh"
#include "../include/Camera.h"
#include "../include/Runtime.h"

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
bool ray_intersects_box(glm::vec3 position, glm::vec3 direction, glm::vec3& box_min, glm::vec3& box_max, int& towards) {

	glm::vec3 invDirection = 1.0f / direction;

	// Calculate intersection distances
	glm::vec3 t1 = (box_min - position) * invDirection;
	glm::vec3 t2 = (box_max - position) * invDirection;

	// Find the maximum and minimum of these intersection distances
	glm::vec3 tMin = glm::min(t1, t2);
	glm::vec3 tMax = glm::max(t1, t2);

	float tNear = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
	float tFar = glm::min(glm::min(tMax.x, tMax.y), tMax.z);

	if (tFar >= tNear) {
		if (tNear >= 0.0f) {
			towards = 1;
		}
		else {
			towards = -1;
		}
	}
	else {
		towards = 0;
	}
	
	return tFar >= tNear;
}

__device__
glm::vec4 sample_texture(glm::vec4* texture, glm::ivec2 dims, float x, float y) {
	if (y >= 0.0f && x >= 0.0f && x <= 1.0f && y <= 1.0f) {
		return texture[static_cast<int>(y * dims.y) * dims.x + static_cast<int>(x * dims.x)];
	}
	return glm::vec4(0.0f);
}

void Camera::capture(d_ModelInstance* instances, uint32_t instance_count, d_Model* models, d_Material* materials, d_AmbientLight* amb_light, d_PointLight* point_lights, uint32_t point_lights_size, glm::vec4* buffer, glm::vec4* buffer_post) {

	setup_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->current_fov.x, this->d_ray_matrix[0], this->dims);
	set_visible_tris << < (instance_count / 128) + 1, 128 >> > (this->position, this->direction, this->current_fov, models, instances, instance_count);
	cudaDeviceSynchronize();
	for (uint8_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		capture_with_rays << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->position, this->direction, this->current_fov.x, instances, instance_count, this->d_ray_matrix[i], this->dims, models);
		cudaDeviceSynchronize();
	}
	calculate_lighting << < (this->dims.y * this->dims.x) / 128, 128 >> > (amb_light, point_lights, materials, point_lights_size, this->d_ray_matrix[0], this->dims, buffer);
	cudaDeviceSynchronize();

	//fxaa_pass << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->dims, buffer, buffer_post);
	//error_check(cudaGetLastError(), "fxaa pass");
	//cudaDeviceSynchronize();

	//texture_map << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->d_ray_matrix[0], this->dims, buffer);

	//copy_frame_buffer << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->dims, buffer, fin_buffer);
	//error_check(cudaGetLastError(), "copy_frame_buffer");


	if (Runtime::PLAYER_OBJECT->get_player_state() == PlayerState::RunF) {
		draw_crosshair << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->dims, buffer, Runtime::PLAYER_OBJECT->get_current_weapon()->get_crosshair(), true, false);
		error_check(cudaGetLastError(), "draw_crosshair RUNNING");
	}
	else {
		draw_crosshair << < (this->dims.y * this->dims.x) / 128, 128 >> > (this->dims, buffer, Runtime::PLAYER_OBJECT->get_current_weapon()->get_crosshair(), false, true);
		error_check(cudaGetLastError(), "draw_crosshair IDLE");
	}
	cudaDeviceSynchronize();
}

__global__
void setup_rays(glm::vec3 position, glm::vec3 direction, float horizontal_fov, Ray* rays, glm::ivec2 dims) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	if (!(x >= dims.x && x < 0) && !(y >= dims.y && y < 0)) {
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
void set_visible_tris(glm::vec3 position, glm::vec3 direction, glm::vec2 fov, d_Model* models, d_ModelInstance* instances, uint32_t instance_count) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i);
	uint32_t idx = x;

	if (x < instance_count) {
		if (!instances[idx].is_hitbox) {
			glm::vec3 min_direction_x = glm::vec3((((-fov.x * 0.5f) / fov.x) * (PI / 180.0f)) * direction.x, direction.y, direction.z);
			glm::vec3 max_direction_x = glm::vec3((((fov.x * 0.5f) / fov.x) * (PI / 180.0f)) * direction.x, direction.y, direction.z);
			glm::vec3 min_direction_y = glm::vec3(direction.x, (((-fov.y * 0.5f) / fov.y) * (PI / 180.0f)) * direction.y, direction.z);
			glm::vec3 max_direction_y = glm::vec3(direction.x, (((-fov.y * 0.5f) / fov.y) * (PI / 180.0f)) * direction.y, direction.z);

			d_ModelInstance* instance = &instances[idx];
			d_Model* model = &models[instance->model_index];
			for (uint32_t k = 0; k < *model->triangle_count; k++) {

				float answers[5];
				answers[0] = glm::dot(model->triangles[k].normal, min_direction_x);
				answers[1] = glm::dot(model->triangles[k].normal, max_direction_x);
				answers[2] = glm::dot(model->triangles[k].normal, min_direction_y);
				answers[3] = glm::dot(model->triangles[k].normal, max_direction_y);
				answers[4] = glm::dot(model->triangles[k].normal, direction);

				bool vis = false;
				for (uint8_t m = 0; m < 5; m++) {
					if (answers[m] <= 0.0f) {
						vis = true;
						break;
					}
				}

				if (vis) {
					instance->visible_triangles[k] = true;
				}
				else {
					instance->visible_triangles[k] = false;
				}
				instance->visible_triangles[k] = true;
			}
		}
		else {
			return;
		}
	}
	else {
		return;
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
		uint32_t history[16];
		uint32_t history_idx = 0;
		for (int j1 = 0; j1 < instance_count; j1++) {
			d_ModelInstance* g = &instances[j1];
			if (!g->is_hitbox) {
				float scale = g->scale;
				//printf("ModelIndex = %i\n", g->model_index);
				//printf("Model Index %i\n", g->model_index);
				uint32_t c = *(models[g->model_index].triangle_count);
				d_Model* model = &models[g->model_index];
				uint32_t next = model->bvh.initial;;
				bool went_back_and_checked = false;
				BVHNode* node = &model->bvh.nodes[model->bvh.initial];
				for (uint32_t n = 0; n < models[g->model_index].bvh.layers; n++) {
					bool cont = false;
					if (n == 0 && !model->bvh.nodes[model->bvh.initial].volume.is_base && model->bvh.node_size > 1) {
						//printf("BVH initial = %i for size %i\n", model->bvh.initial, model->bvh.node_size);
						glm::vec3 min = node->volume.vertices[0] + g->position, max = node->volume.vertices[1] + g->position;
						int towards = 0;
						bool vol_intersect = ray_intersects_box(ray->position, ray->direction, min, max, towards);

						if (vol_intersect) {
							cont = true;
						}
						/*if (vol_intersect && !node->volume.is_base) {
							if (towards < 0) {
								next = model->bvh.nodes[model->bvh.initial].left_child_index;
								goto contin;
								//printf("Going left to %i\n", next);
							}
							else if (towards > 0) {
								next = model->bvh.nodes[model->bvh.initial].right_child_index;
								goto contin;
								//printf("Going right to %i\n", next);
							}
						}
						else if (node->volume.is_base) {
							cont = true;

						}*/
					}
					else if (model->bvh.nodes[model->bvh.initial].volume.is_base){
						cont = true;
					}

					if (cont) {
						//printf("Next Node = %i out of %i nodes\n", next, model->bvh.node_size);
						//BVHNode* node = &model->bvh.nodes[next];

						bool intersect_test = false;
						if (!node->volume.is_base) {
							//printf("Left idx = %i, Right idx = %i\n", node->left_child_index, node->right_child_index);
							BVHNode* left = &model->bvh.nodes[node->left_child_index],
								* right = &model->bvh.nodes[node->right_child_index];
							glm::vec3 min = left->volume.vertices[0] + g->position, max = left->volume.vertices[1] + g->position;
							glm::vec3 min_right = right->volume.vertices[0] + g->position, max_right = right->volume.vertices[1] + g->position;
							int towards = 0, towards_right = 0;
							bool vol_intersect_a = ray_intersects_box(ray->position, ray->direction, min, max, towards);
							bool vol_intersect_b = ray_intersects_box(ray->position, ray->direction, min_right, max_right, towards_right);
							bool corrected = false;
							if (vol_intersect_a && vol_intersect_b) {

								if (towards > 0) {
									vol_intersect_a = true;
									vol_intersect_b = false;
									next = left->right_child_index;
									corrected = true;
								}
								else if (towards < 0) {
									vol_intersect_a = true;
									vol_intersect_b = false;
									next = left->left_child_index;
									corrected = true;
								}
								if (towards_right > 0) {
									vol_intersect_a = false;
									vol_intersect_b = true;
									next = right->right_child_index;
									corrected = true;
								}
								else if (towards_right < 0) {
									vol_intersect_a = false;
									vol_intersect_b = true;
									next = right->left_child_index;
									corrected = true;
								}

								if (corrected) {
									n++;
								}
							}

							if (vol_intersect_a && !corrected) {
								next = node->left_child_index;
							}
							else if (vol_intersect_b && !corrected) {
								next = node->right_child_index;
							}
							if (corrected) {
								continue;
							}
							if (!vol_intersect_a && !vol_intersect_b) {
								/*if (next == node->left_child_index) {
									next = node->right_child_index;
								}
								else if (next == node->right_child_index) {
									next = node->left_child_index;
								}*/
								//printf("BVH going nowhere\n");
								goto contin;
								//printf("BVH going nowhere\n");
								//continue;
							}
						}
						if (node->volume.is_base) {
							//printf("Node is base w/ %i triangles!\n", node->volume.triangle_count);

							for (uint32_t p = 0; p < static_cast<uint32_t>(node->volume.triangle_count); p++) {
								tried = true;
								//printf("Checking Triangle %i\n", node->volume.triangles[p]);
								//printf("Before Grabbing Triangle %i!\n", node->volume.triangles[p]);
								Tri* t = &model->triangles[node->volume.triangles[p]];
								//printf("After grabbing triangle!\n");
								Vertex* vs = model->vertices;
								glm::vec3 offset = instances[j1].position;
								glm::vec3 dir = instances[j1].rotation;

								glm::vec2 intersection;
								float d;
								glm::vec2 uv;

								glm::vec3 rotation_axis = glm::vec3(0.0f, 0.0f, 1.0f);
								float angle = glm::acos(glm::dot(rotation_axis, rotation_axis));
								glm::vec3 axis = glm::cross(rotation_axis, dir);
								glm::mat4 rotation_matrix = glm::rotate(glm::mat4(1.0f), angle, dir);

								glm::vec4 verta4 = glm::vec4(scale * vs[t->a].position + offset, 1.0f) * rotation_matrix;
								glm::vec3 verta = glm::vec3(verta4.x, verta4.y, verta4.z);
								glm::vec4 vertb4 = glm::vec4(scale * vs[t->b].position + offset, 1.0f) * rotation_matrix;
								glm::vec3 vertb = glm::vec3(vertb4.x, vertb4.y, vertb4.z);
								glm::vec4 vertc4 = glm::vec4(scale * vs[t->c].position + offset, 1.0f) * rotation_matrix;
								glm::vec3 vertc = glm::vec3(vertc4.x, vertc4.y, vertc4.z);



								bool intersection_detection = glm::intersectRayTriangle(ray->position, ray->direction, verta, vertb, vertc, uv, d);
								if (intersection_detection) {
									glm::vec3 intersect = (d * direction) + position;
									float tr = d;//(intersect - position).length();
									if (tr < last_leng && tr >= 0.01f) {
										//printf("Intersection true for model %i!\n", g->model_index);

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
										ray->payload.model = model;
										ray->payload.triangle = t;
										intersected = true;
										last_leng = tr;
									}
									//break;
								}
							}
						}
					}
				contin:;
					node = &model->bvh.nodes[next];
					continue;
				}
			}
			else {
				continue;
			}
			if (!intersected) {
				ray->payload.color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
				ray->intersected = false;
			}
		}
	}
}

__global__
void calculate_lighting(d_AmbientLight* amb, d_PointLight* lights, d_Material* materials, uint32_t lights_size, Ray* rays, glm::ivec2 dims, glm::vec4* out) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	uint32_t idx = y * dims.x + x;

	out[idx] = glm::vec4(0.0f);

	Ray* ray = &rays[idx];
	if (ray->intersected) {
		glm::vec3 result = glm::vec3(0.0f);
		glm::vec4 diffuse_color = sample_texture(materials[ray->payload.model->material_index].color_map.data, materials[ray->payload.model->material_index].color_map.dims, ray->payload.uv.x, ray->payload.uv.y);

		glm::vec4 lighting = glm::vec4(0.0f);
		bool lit = false;
		float cummulative_intensity = 0.0f;
		//printf("Light Count = %i\n", static_cast<int>(lights_size));
		for (int l = 0; l < static_cast<int>(lights_size); l++) {

			float amb_strength = 1.0f;
			glm::vec3 ambient = amb->intensity * amb->diffuse_color * amb_strength;
			glm::vec3 light_direction = lights[l].position - ray->payload.intersection;
			float distance = glm::length(light_direction);
			if (distance <= lights[l].range) {
				float diff = glm::max(glm::dot(ray->payload.triangle->normal, light_direction), 0.0f);
				float intensity = lights[l].intensity / (distance * distance);
				glm::vec3 diffuse = intensity * lights[l].diffuse_color * diff;

				result += (ambient + diffuse) + diffuse_color.w * glm::vec3(diffuse_color.x, diffuse_color.y, diffuse_color.z);

				cummulative_intensity += intensity;
			}

		}

		out[idx] = glm::vec4(result, 1.0f);
		out[idx] = glm::clamp(out[idx], 0.0f, 1.0f);
	}
	else {
		out[idx] = glm::vec4(0.32f, 0.32f, 0.1f, 1.0f);
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
		//out[idx] = sample_texture(ray->payload.model->color_map->data, ray->payload.model->color_map->dims, ray->payload.uv.x, ray->payload.uv.y);
	}
	else {
		out[idx] = ray->payload.color;
	}
}

__global__
void draw_crosshair(glm::ivec2 dims, glm::vec4* buffer, Crosshair cross, bool is_run, bool is_walk) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	int32_t idx = y * dims.x + x;

	x -= static_cast<int32_t>(static_cast<float>(dims.x) * 0.5f);
	y -= static_cast<int32_t>(static_cast<float>(dims.y) * 0.5f);

	float f_x = static_cast<float>(x) / static_cast<float>(dims.x), f_y = static_cast<float>(y) / static_cast<float>(dims.y);

	if (is_walk) {
		if (sqrtf(f_x * f_x + f_y * f_y) <= cross.walk_radius && sqrtf(f_x * f_x + f_y * f_y) > cross.walk_radius * 0.5f) {
			buffer[idx] = glm::vec4(1.0f);
		}
	}
	else if (is_run) {
		if (sqrtf(f_x * f_x + f_y * f_y) <= cross.run_radius && sqrtf(f_x * f_x + f_y * f_y) > cross.walk_radius * 0.5f) {
			buffer[idx] = glm::vec4(1.0f);
		}
	}
	else {
		buffer[idx] = buffer[idx];
	}
}

__global__
void fxaa_pass(glm::ivec2 dims, glm::vec4* buffer, glm::vec4* buffer_out) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	int32_t idx = y * dims.x + x;

	if (x > 0 && x < dims.x - 1 && y > 0 && y < dims.y - 1) {
		glm::vec4 pixels[8], current = glm::clamp(buffer[idx], 0.001f, 1.0f);
		float* out_matrix = new float[8];
		pixels[0] = buffer[idx - (dims.x) - 1];
		pixels[1] = buffer[idx - dims.x];
		pixels[2] = buffer[idx - dims.x + 1];

		pixels[3] = buffer[idx - 1];
		pixels[4] = buffer[idx + 1];

		pixels[5] = buffer[idx + dims.x - 1];
		pixels[6] = buffer[idx + dims.x];
		pixels[7] = buffer[idx + dims.x + 1];
		
		float total = 0.0f;
		for (uint8_t k = 0; k < 8; k++) {
			pixels[k] = glm::clamp(pixels[k], 0.001f, 1.0f);

			pixels[k] = pixels[k] / current;

			const float comp = 1.089f;

			if (pixels[k].x >= comp || pixels[k].y >= comp || pixels[k].z >= comp) {
				out_matrix[k] = glm::clamp(glm::dot(pixels[k], current), 0.0f, 1.0f);
				total += out_matrix[k];
			}
			else {
				out_matrix[k] = 1.0f;
			}
		}

		total /= 8.0f;

		//glm::mat3 matrix = glm::mat3(glm::vec3(out_matrix[0], out_matrix[3], out_matrix[5]), glm::vec3(out_matrix[1], 1.0f, out_matrix[7]), glm::vec3(out_matrix[2], out_matrix[4], out_matrix[7]));
		buffer_out[idx] = glm::clamp(total * buffer[idx], 0.0f, 1.0f);
	}
}

__global__
void copy_frame_buffer(glm::ivec2 dims, glm::vec4* dest, glm::vec4* from) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i) % dims.x,
		y = ((j * 128 + i) - x) / dims.x;
	int32_t idx = y * dims.x + x;

	if (x >= 0 && x < dims.x && y >= 0 && y < dims.y) {
		dest[idx] = from[idx];
	}
}