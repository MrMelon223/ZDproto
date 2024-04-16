#ifndef WEAPON_H
#define WEAPON_H

#include "Model.h"

enum class WeaponType {
	SemiAutomatic,
	FullyAutomatic,
	Burst,
	BoltAction,
	SingleShot,
	Melee
};

class BulletWeapon {
protected:
	WeaponType weapon_type;

	std::string name;
	glm::vec3 offset;

	float walk_speed, run_speed, base_damage, fire_delay;

	uint32_t instance_index;

public:
	BulletWeapon();
	BulletWeapon(WeaponType, std::string, glm::vec3, float, float, float, float, uint32_t);

	WeaponType get_weapon_type() { return this->weapon_type; }

	void set_name(std::string s) { this->name = s; }
	std::string get_name() { return this->name; }

	float get_walk_speed() { return this->walk_speed; }
	float get_run_speed() { return this->run_speed; }
	float get_base_damage() { return this->base_damage; }
	float get_fire_delay() { return this->fire_delay; }

	uint32_t get_instance_index() { return this->instance_index; }

};

#endif
