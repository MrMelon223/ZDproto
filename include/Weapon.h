#ifndef WEAPON_H
#define WEAPON_H

#include "Model.h"
#include "Crosshair.h"

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

	bool in_use, in_inventory;

	float last_fire;

	float mass_kg;

	Crosshair crosshair;

public:
	BulletWeapon();
	BulletWeapon(WeaponType, std::string, glm::vec3, float, float, float, float, uint32_t);

	void set_weapon_type(WeaponType t) { this->weapon_type = t; }

	WeaponType get_weapon_type() { return this->weapon_type; }

	void set_name(std::string s) { this->name = s; }
	std::string get_name() { return this->name; }

	void set_walk_speed(float s) { this->walk_speed = s; }
	void set_run_speed(float s) { this->run_speed = s; }
	void set_base_damage(float d) { this->base_damage = d; }
	void set_fire_delay(float d) { this->fire_delay = d; }
	void set_instance_index(uint32_t i) { this->instance_index = i; }

	float get_walk_speed() { return this->walk_speed; }
	float get_run_speed() { return this->run_speed; }
	float get_base_damage() { return this->base_damage; }
	float get_fire_delay() { return this->fire_delay; }

	uint32_t get_instance_index() { return this->instance_index; }

	void set_offset(glm::vec3 o) { this->offset = o; }
	glm::vec3 get_offset() { return this->offset; }

	float get_last_fire() { return this->last_fire; }
	void set_last_fire(float f) { this->last_fire = f; }

	void set_mass_kg(float m) { this->mass_kg = m; }
	float get_mass_kg() { return this->mass_kg; }

	Crosshair get_crosshair() { return this->crosshair; }

};

#endif
