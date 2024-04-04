#ifndef RUNTIME_H
#define RUNTIME_H

#include "Model.h"

namespace Runtime {
	static std::vector<HostModel> HOST_MODELS;
	static std::vector<d_Model> DEVICE_MODELS;

	bool host_models_contains(std::string);
	HostModel* find_host_model(std::string);

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
