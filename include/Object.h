#ifndef OBJECT_H
#define OBJECT_H

#include "Model.h"
#include "Camera.h"
#include "Weapon.h"

enum ObjectType {
	AI,
	Physics,
	Player,
	Weapon,
	Static
};

union ObjIndexs {
	uint32_t physics_object_index;
	uint32_t player_index;
	uint32_t weapon_index;
	uint32_t static_index;
};

class Object {
protected:
	glm::vec3 spawn_point;

	glm::vec3 position;
	glm::vec3 rotation;
	glm::vec3 direction;
	glm::vec3 velocity;
	float mass_kg;

	ObjectType object_type;

	uint32_t model_index;
	uint32_t instance_index;

	uint32_t hitbox_instance_index;

		// Physics object variables

		// AI object variables
	glm::vec3 target_position;
	float current_health, max_health;
	float attack_range;
	float current_damage;

	float last_attack;
	float attack_cooldown;

	float last_time;	// Physics
		// Player Variable(s)
	Camera* camera_ptr;

	BulletWeapon* primary;

	PlayerState player_state;

	std::string name;

	ObjIndexs obj_indices;

	float creation_time;

public:
	Object();
	Object(ObjectType, glm::vec3);
	Object(ObjectType, ObjIndexs, std::string, glm::vec3, glm::vec3, uint32_t, uint32_t, uint32_t);

	void set_primary_weapon(BulletWeapon* w) { this->primary = w; }
	BulletWeapon* get_current_weapon() { return this->primary; }

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

	void set_max_health(float h) { this->max_health = h; }
	void set_health(float h) { this->current_health = h; }
	float get_health() { return this->current_health; }
	__device__ __host__
	ObjectType get_object_type() { return this->object_type; }

	void attach_camera(Camera* c) { this->camera_ptr = c; this->camera_ptr->set_position(this->position); }

	float get_current_damage() { return this->current_damage; }
	float get_attack_range() { return this->attack_range; }

	void set_spawn_point(glm::vec3 p) { this->spawn_point = p; }
	glm::vec3 get_spawn_point() { return this->spawn_point; }

	float get_last_attack_time() { return this->last_attack; }
	void set_last_attack_time(float t) { this->last_attack = t; }

	float get_attack_cooldown() { return this->attack_cooldown; }

	void set_name(std::string n) { this->name = n; }
	std::string get_name() { return this->name; }

	void set_mass(float m) { this->mass_kg = m; }
	float get_mass() { return this->mass_kg; }

	void set_player_state(PlayerState s) { this->player_state = s; }
	PlayerState get_player_state() { return this->player_state; }

	void set_velocity(glm::vec3 v) { this->velocity = v; }
	glm::vec3 get_velocity() { return this->velocity; }
};

#endif
