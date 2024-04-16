#ifndef APPLICATION_H
#define APPLICATION_H

#include "Camera.h"
#include "Level.cuh"

static void keyboard_callback(GLFWwindow*, int, int, int, int);
static void mouse_callback(GLFWwindow*, int, int, int);

class Application {
protected:
	glm::ivec2 dims;
	GLFWwindow* win;

	Level* level;
	Camera* camera;

	//std::vector<FiniteStateMachine> ai;

	bool loop;

	glm::vec4* frame_buffer;

	sqlite3* database_connection;

	void input_handle();
	void mouse_handle();

public:
	Application();
	Application(int32_t, int32_t);

	void zero_frame_buffer_sse();

	void main_loop();
};

#endif
