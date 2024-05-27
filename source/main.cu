	// main.cpp
#include "../include/Application.h"

int main() {

	Application* app = new Application(960, 540);

	app->main_loop();

	return 0;
}