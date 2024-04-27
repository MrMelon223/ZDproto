	// Weapon.cpp
#include "../include/Weapon.h"

BulletWeapon::BulletWeapon() {
	this->crosshair = Crosshair{ 0.010f, 0.015f, glm::vec4(1.0f) };
}

BulletWeapon::BulletWeapon(WeaponType w, std::string name, glm::vec3 offset, float walk, float run, float damage, float fire_delay, uint32_t instance) {
	this->weapon_type = w;

	this->name = name;
	this->offset = offset;

	this->walk_speed = walk;
	this->run_speed = run;
	this->base_damage = damage;
	this->fire_delay = fire_delay;

	this->instance_index = instance;

	this->last_fire = glfwGetTime();

	this->crosshair = Crosshair{ 0.010f, 0.015f, glm::vec4(1.0f)};
}