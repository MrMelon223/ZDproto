#ifndef OBJECT_H
#define OBJECT_H

#include "Model.h"
#include "Camera.h"

enum ObjectType {
	AI,
	Physics,
	Player
};

class Object {
protected:
	glm::vec3 spawn_point;

	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 direction;

	ObjectType object_type;

	uint32_t model_index;
	uint32_t instance_index;

		// Physics object variables

		// AI object variables
	glm::vec3 target_position;
	float current_health;
		// Player Variables
	Camera* camera_ptr;

public:
	Object(ObjectType, glm::vec3);
	Object(ObjectType, glm::vec3, glm::vec3, uint32_t, uint32_t);

	void bind_model(uint32_t model) { this->model_index = model;}

	uint32_t get_model_index() { return this->model_index; }

	glm::vec3 get_direction() { return this->direction; }
	__device__ __host__
	glm::vec3 get_position() { return this->position; }

	void set_direction(glm::vec3 dir) { this->direction = dir; }
	void set_position(glm::vec3 pos) { this->position = pos; }

	void set_instance_index(uint32_t idx) { this->instance_index = idx; }
	__device__ __host__
	uint32_t get_instance_index() { return this->instance_index; }

	void update(d_ModelInstance*, Camera*, float, GLFWwindow*);

	void set_health(float h) { this->current_health = h; }
	float get_health() { return this->current_health; }
	__device__ __host__
	ObjectType get_object_type() { return this->object_type; }

	void attach_camera(Camera* c) { this->camera_ptr = c; this->camera_ptr->set_position(this->position); }

	glm::vec3 get_spawn_point() { return this->spawn_point; }

};

#endif
