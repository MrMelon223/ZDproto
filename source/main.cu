	// main.cpp
#include "../include/Application.h"

int main() {

	Application* app = new Application(640, 360);

	app->main_loop();

	return 0;
}