#ifndef APPLICATION_H
#define APPLICATION_H

#include "Camera.h"
#include "Level.cuh"

static void keyboard_callback(GLFWwindow*, int, int, int, int);
static void mouse_callback(GLFWwindow*, int, int, int);

struct KeyboardButtonUse {
	int key, scancode, action, mods;
};
struct MouseButtonUse {
	int button, action, mods;
};

extern std::queue<KeyboardButtonUse> keyboard_button_uses;
extern std::queue<MouseButtonUse> mouse_button_uses;

class Application {
protected:
	glm::ivec2 dims;
	GLFWwindow* win;

	Level* level;
	Camera* camera;

	//std::vector<FiniteStateMachine> ai;

	bool loop, is_walking, trying_sprint, is_sprinting;

	glm::vec4* frame_buffer;

	sqlite3* database_connection;

	void input_handle(KeyboardButtonUse&);
	void mouse_handle(MouseButtonUse&);

	void empty_input_queues();

public:
	Application();
	Application(int32_t, int32_t);

	void zero_frame_buffer_sse();

	void main_loop();
};

#endif
