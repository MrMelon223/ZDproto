	// Object.cpp
#include "../include/Object.h"

Object::Object(ObjectType type, glm::vec3 position) {
	this->position = position;
	this->spawn_point = position;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);
	this->direction = glm::vec3(0.0f);
}

Object::Object(ObjectType type, glm::vec3 position, glm::vec3 direction, uint32_t model, uint32_t instance) {
	this->position = position;
	this->spawn_point = position;
	this->direction = direction;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);

	this->model_index = model;
	this->instance_index = instance;
}

void Object::update(d_ModelInstance* instances, Camera* cam, float t, GLFWwindow* win) {
	std::cout << "updating objecting" << std::endl;
	switch (this->object_type) {
		case ObjectType::AI:
			std::cout << "	updating as AI" << std::endl;
			this->target_position = cam->get_position();
			this->direction = glm::normalize(this->target_position - this->position);
			this->position += this->direction * t * 1.0f;
			instances->position = this->position;
			break;
		case ObjectType::Physics:
			break;
		case ObjectType::Player:
			double x, y;
			glfwGetCursorPos(win, &x, &y);
			cam->add_to_euler_direction(glm::vec2(static_cast<float>(x), static_cast<float>(y)));
			glfwSetCursorPos(win, cam->get_dims().x * 0.5f, cam->get_dims().y * 0.5f);

			break;
	}
}