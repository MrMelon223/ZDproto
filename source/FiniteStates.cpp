	// FiniteStates.cpp
#include "../include/FiniteStates.h"

FiniteStateMachine::FiniteStateMachine(glm::vec3 spawn_point, Level* level, d_ModelInstance* model) {
	this->position = spawn_point;

	this->level = level;

	//this->target_position = this->level->get_camera().get_position();

	this->ai_model = model;
}

void FiniteStateMachine::update_target_position() {
	//this->target_position = this->level->get_camera().get_position();
}

void FiniteStateMachine::update_position(float t) {
	this->direction = glm::normalize(this->target_position - this->position);

	this->position += t * Runtime::BASE_SPEED * this->direction;
}