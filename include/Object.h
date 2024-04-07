#ifndef OBJECT_H
#define OBJECT_H

#include "Model.h"

enum ObjectType {
	AI,
	Physics,
	Player
};

class Object {
protected:
	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 direction;

	ObjectType object_type;

	uint32_t model_index;
	uint32_t instance_index;

		// Physics object variables


public:
	Object(ObjectType, glm::vec3);
	Object(ObjectType, glm::vec3, glm::vec3, uint32_t, uint32_t);

	void bind_model(uint32_t model) { this->model_index = model;}

	uint32_t get_model_index() { return this->model_index; }

	glm::vec3 get_direction() { return this->direction; }
	glm::vec3 get_position() { return this->position; }

	void set_direction(glm::vec3 dir) { this->direction = dir; }
	void set_position(glm::vec3 pos) { this->position = pos; }

	void set_instance_index(uint32_t idx) { this->instance_index = idx; }
	uint32_t get_instance_index() { return this->instance_index; }

	void update();

};

#endif
