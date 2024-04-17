
#include "../include/Level.cuh"

__global__
void test_intersection(glm::vec3 position, glm::vec3 direction, Object* objects, uint32_t object_count, d_ModelInstance* instances, uint32_t instance_count, d_Model* models, bool* intersected) {
	int j = blockDim.y * blockIdx.y + threadIdx.y,
		i = blockDim.x * blockIdx.x + threadIdx.x,
		x = (j * 128 + i);
	uint32_t idx = x;
	
	if (idx < object_count ) {
		//printf("Processing object %i\n", idx);
		if (objects[idx].get_object_type() != ObjectType::Player) {
			//printf("Index accessing = %i out of %i\n", objects[idx].get_instance_index(), instance_count);
			d_ModelInstance* instance = &instances[objects[idx].get_instance_index()];
			//printf("Index = %i from %p @ %p\n", idx, &objects[idx], &models[instance->model_index]);
			//printf("Model Index = %i\n", instance->model_index);
			d_Model model = models[instance->model_index];
			glm::vec3 offset = instance->position;
			Vertex* verts = model.vertices;
			uint32_t vert_count = *model.vertex_count;
			Tri* tris = model.triangles;
			float closest = 100.0f;
			int32_t r = -1;
			//printf("Model index = %i\n", instance->model_index);
			//printf("Triangle count = %i\n", *model.triangle_count);
			//printf("Vertex count = %i\n", *model.vertex_count);
			for (uint32_t k = 0; k < *model.triangle_count; k++) {
				glm::vec2 uv;
				float dist;
				Tri* t = &tris[k];
				//printf("Triangle %i = {%i, %i, %i} out of vertex %i, out of triangle %i\n", k, t->a, t->b, t->c, vert_count, *model.triangle_count);
				Vertex* a = &verts[tris[k].a], * b = &verts[tris[k].b], * c = &verts[tris[k].c];

				if (glm::intersectRayTriangle(position, direction, a->position + offset, b->position + offset, c->position + offset, uv, dist)) {
					//printf("Ray Intersected Triangle %i from object %i\n", k, idx);
					if (dist <= closest) {
						r = static_cast<int32_t>(k);
						//intersected[idx] = true;
						//printf("r set to %i\n", r);
						closest = dist;
						break;
					}
				}
			}
			if (r != -1) {
				intersected[idx] = true;
				//printf("Intersection on object %i!\n", idx);
			}
			else {
				//printf("No intersection from object %i\n", idx);
				intersected[idx] = false;
			}
		}
		else {
			intersected[idx] = false;
		}
	}
	else {
		//printf("Index %i out of range\n", idx);
	}
}

Level::Level() {
}


Level::Level(std::string path, Camera* cam) {
	std::cout << "Initializing level from: " << path << std::endl;
	this->camera_ptr = cam;
	this->d_objects = nullptr;
	this->load_from(path);
}

void Level::add_object(Object obj) {
	this->objects.push_back(obj);
}

void Level::clean_d_objects() {
	error_check(cudaFree(this->d_objects));
	error_check(cudaFree(this->d_model_instances));
}

void Level::upload_objects() {
	error_check(cudaMalloc((void**)&this->d_objects, sizeof(Object) * this->objects.size()), "Level::upload_objects cudaMalloc");
	error_check(cudaMemcpy(this->d_objects, this->objects.data(), sizeof(Object) * this->objects.size(), cudaMemcpyHostToDevice), "Level::upload_objects cudaMemcpy");
	cudaDeviceSynchronize();
	this->d_object_count = static_cast<uint32_t>(this->objects.size());
}



uint32_t get_instance_index(d_ModelInstance);