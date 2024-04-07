	// Object.cpp
#include "../include/Object.h"

Object::Object(ObjectType type, glm::vec3 position) {
	this->position = position;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	this->direction = glm::vec3(0.0f);
}

Object::Object(ObjectType type, glm::vec3 position, glm::vec3 direction, uint32_t model, uint32_t instance) {
	this->position = position;
	this->direction = direction;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);

	this->model_index = model;
	this->instance_index = instance;
}

void Object::update() {

}