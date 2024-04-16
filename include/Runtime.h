#ifndef RUNTIME_H
#define RUNTIME_H

#include "Application.h"
#include "Weapon.h"
#include "Object.h"
#include "Model.h"

extern std::vector<HostModel> HOST_MODELS;
extern std::vector<d_Model> DEVICE_MODELS;


extern Object* PLAYER_OBJECT;

namespace Runtime {
	extern std::vector<Object> OBJECTS;
	extern std::vector<BulletWeapon> WEAPONS;

	void runtime_load(sqlite3*);
	void load_objects(sqlite3*);
	void load_weapons(sqlite3*);

	bool host_models_contains(std::string);
	HostModel* find_host_model(std::string);
	uint32_t find_host_model_index(std::string);

		// Controls
	static bool KEY_USED = false;
	static int CURRENT_KEY = -1;
	static int CURRENT_SCANCODE = -1;
	static int CURRENT_ACTION = -1;
	static int CURRENT_MODS = -1;

	static bool DEV_NEW = false;
	static bool DEV_NEW_LISTENER = false;
	static bool DEV_NEW_VERT = false;

	static bool MOUSE_USED = false;
	static int CURRENT_MOUSE = -1;

	static float BASE_SPEED = 100.0f;

	static float X_SENSITIVITY = 1.0f;
	static float Y_SENSITIVITY = 1.0f;

	namespace control {
		static void reset_key() {
			KEY_USED = false;
			CURRENT_KEY = -1;
			CURRENT_SCANCODE = -1;
			CURRENT_ACTION = -1;
			CURRENT_MODS = -1;

			DEV_NEW_LISTENER = false;
			DEV_NEW_VERT = false;

		}

		static void reset_mouse() {
			MOUSE_USED = false;
			CURRENT_MOUSE = -1;
			CURRENT_SCANCODE = -1;
			CURRENT_ACTION = -1;
			CURRENT_MODS = -1;
		}
	}
}

#endif
