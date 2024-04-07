	// main.cpp
#include "../include/Application.h"

int main() {

	Runtime::runtime_load();

	Application* app = new Application(1280, 720);

	app->main_loop();

	return 0;
}