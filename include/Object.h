#ifndef OBJECT_H
#define OBJECT_H

#include "Model.h"
#include "Camera.h"

enum ObjectType {
	AI,
	Physics,
	Player,
	Weapon
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

	uint32_t hitbox_instance_index;

		// Physics object variables

		// AI object variables
	glm::vec3 target_position;
	float current_health;
	float attack_range;
	float current_damage;

	float last_attack;
	float attack_cooldown;
		// Player Variable(s)
	Camera* camera_ptr;
		// Weapon Variables
	float walk_speed;
	float run_speed;
	glm::vec3 weapon_offset;
	bool in_inventory, in_use;
	std::string weapon_name;


public:
	Object();
	Object(ObjectType, glm::vec3);
	Object(ObjectType, glm::vec3, glm::vec3, uint32_t, uint32_t, uint32_t);

	void bind_model(uint32_t model) { this->model_index = model;}

	uint32_t get_model_index() { return this->model_index; }
	uint32_t get_hitbox_instance_index() { return this->hitbox_instance_index; }

	glm::vec3 get_direction() { return this->direction; }
	__device__ __host__
	glm::vec3 get_position() { return this->position; }

	void set_direction(glm::vec3 dir) { this->direction = dir; }
	void set_position(glm::vec3 pos) { this->position = pos; }

	void set_instance_index(uint32_t idx) { this->instance_index = idx; }
	__device__ __host__
	uint32_t get_instance_index() { return this->instance_index; }

	void update(d_ModelInstance*, d_ModelInstance*, Camera*, float, GLFWwindow*, Object*);

	void set_health(float h) { this->current_health = h; }
	float get_health() { return this->current_health; }
	__device__ __host__
	ObjectType get_object_type() { return this->object_type; }

	void attach_camera(Camera* c) { this->camera_ptr = c; this->camera_ptr->set_position(this->position); }

	float get_current_damage() { return this->current_damage; }
	float get_attack_range() { return this->attack_range; }

	glm::vec3 get_spawn_point() { return this->spawn_point; }

	float get_last_attack_time() { return this->last_attack; }
	void set_last_attack_time(float t) { this->last_attack = t; }

	float get_attack_cooldown() { return this->attack_cooldown; }

	void set_walk_speed(float s) { this->walk_speed = s; }
	float get_walk_speed() { return this->walk_speed; }

	void set_run_speed(float s) { this->run_speed = s; }
	float get_run_speed() { return this->run_speed; }

	void set_weapon_offset(glm::vec3 p) { this->weapon_offset = p; }
	glm::vec3 get_weapon_offset() { return this->weapon_offset; }

	void set_in_inventory(bool b) { this->in_inventory = b; }
	bool get_in_inventory() { return this->in_inventory; }

	void set_in_use(bool b) { this->in_use = b; }
	bool get_in_use() { return this->in_use; }

	void set_weapon_name(std::string s) { this->weapon_name = s; }
	std::string get_weapon_name() { return this->weapon_name; }

};

#endif
