#ifndef FINITESTATES_H
#define FINITESTATES_H

#include "Application.h"

enum class AIstate {
	FIND_PLAYER,
	ATTACK,
	IDLE
};

class FiniteStateMachine{
protected:
	AIstate state;

	glm::vec3 position;
	glm::vec3 direction;

	glm::vec3 target_position;

	d_ModelInstance* ai_model;

	Level* level;
public:
	FiniteStateMachine(glm::vec3, Level*, d_ModelInstance*);

	void update_target_position();
	void update_position(float);

	glm::vec3 get_position() { return this->position; }

	d_ModelInstance* get_ai_model() { return this->ai_model; }

};

#endif