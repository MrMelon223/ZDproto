	// Object.cpp
#include "../include/Object.h"

Object::Object() {

}

Object::Object(ObjectType type, glm::vec3 position) {
	this->position = position;
	this->spawn_point = position;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	this->direction = glm::vec3(0.0f);
	this->creation_time = glfwGetTime();
}