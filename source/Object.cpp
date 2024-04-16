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
}

Object::Object(ObjectType type, glm::vec3 position, glm::vec3 direction, uint32_t model, uint32_t instance, uint32_t hitbox_instance) {
	this->position = position;
	this->spawn_point = position;
	this->direction = direction;
	this->object_type = type;
	this->rotation = glm::vec3(0.0f, 0.0f, 0.0f);

	this->model_index = model;
	this->instance_index = instance;
	this->hitbox_instance_index = hitbox_instance;

	if (this->object_type == ObjectType::AI) {
		this->current_health = 100.0f;
		this->attack_range = 1.0f;
		this->current_damage = 25.0f;
		this->attack_cooldown = 0.1f;
		this->last_attack = glfwGetTime();
	}
	else if (this->object_type == ObjectType::Player) {
		this->current_health = 100.0f;
		this->current_damage = 50.0f;
		this->attack_cooldown = 0.0f;
		this->last_attack = glfwGetTime();
	}
	else if (this->object_type == ObjectType::Weapon) {
		this->walk_speed = 90.0f;
		this->run_speed = 100.0f;
		this->weapon_offset = glm::vec3(0.25f, 0.25f, 0.0f);
	}
}

void Object::update(d_ModelInstance* instances, d_ModelInstance* hitbox_instance, Camera* cam, float t, GLFWwindow* win, Object* player) {
	//std::cout << "updating object" << std::endl;
	float to_player, time;
	if (this->object_type == ObjectType::AI) {
		//std::cout << "	updating as AI" << std::endl;
		this->target_position = player->get_position();
		this->direction = glm::normalize(this->target_position - this->position);
		this->position += this->direction * t * 5.0f;
		instances->position = this->position;
		instances->rotation = this->direction;

		glm::vec3 dist = this->target_position - this->position;
		to_player = dist.length();
		printf("Distance to player = %.2f, %.2f, %.2f\n", dist.x, dist.y, dist.z);

		hitbox_instance->position = this->position;
		hitbox_instance->rotation = this->direction;

		time = glfwGetTime();
		if (to_player <= this->attack_range) {
			std::cout << std::setw(10) << "AI Within Attack Range" << std::endl;
			if (time - this->last_attack >= this->attack_cooldown) {
				player->set_health(player->get_health() - this->current_damage);
				this->last_attack = time;
				std::cout << "AI Attacking Player" << std::endl;
			}
		}
	}

	else if (this->object_type == ObjectType::Player) {
		//std::cout << "	updating as Player" << std::endl;
		double x, y;
		glfwGetCursorPos(win, &x, &y);
		cam->add_to_euler_direction(glm::vec2(static_cast<float>(x), static_cast<float>(y)));
		glfwSetCursorPos(win, cam->get_dims().x * 0.5f, cam->get_dims().y * 0.5f);
		this->direction = cam->get_direction();
		this->position = cam->get_position();
		player = this;

		hitbox_instance->position = this->position;
		hitbox_instance->rotation = this->direction;
	}

	else if (this->object_type == ObjectType::Weapon) {
		//std::cout << "	updating as Weapon" << std::endl;
		if (in_inventory) {
			if (in_use) {
				d_ModelInstance* m = &instances[this->instance_index];
				m->position = player->get_position() + this->weapon_offset;
				m->rotation = player->get_direction();
			}
		}
		else {
			glm::vec3 player_direction = glm::normalize(this->target_position - this->position);
			to_player = player_direction.length();

			if (to_player <= 0.5f) {
				std::cout << "Press F to Pickup Weapon" << std::endl;
			}
		}
	}
}