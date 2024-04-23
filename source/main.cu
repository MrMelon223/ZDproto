	// main.cpp
#include "../include/Application.h"

int main() {

	Application* app = new Application(1920, 1080);

	app->main_loop();

	return 0;
}