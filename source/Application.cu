	// Application.cpp
#include "../include/Application.h"

std::queue<KeyboardButtonUse> keyboard_button_uses;
std::queue<MouseButtonUse> mouse_button_uses;

Application::Application() {

}

void Application::zero_frame_buffer_sse() {
	__m128 to_set = _mm_set_ps(0.0f, 0.0f, 0.0f, 0.0f);

	glm::ivec2 dims = this->camera->get_dims();
	for (uint32_t y = 0; y < dims.y; y++) {
		for (uint32_t x = 0; x < dims.x; x++) {
			_mm_store_ps(&this->frame_buffer[y * dims.x + x].x, to_set);
			//this->frame_buffer[y * dims.x + x] = glm::vec4(0.0f);
		}
	}
}

void Application::empty_input_queues() {
	while (!keyboard_button_uses.empty()) {
		this->input_handle(keyboard_button_uses.front());
		keyboard_button_uses.pop();
	}

	while (!mouse_button_uses.empty()) {
		this->mouse_handle(mouse_button_uses.front());
		mouse_button_uses.pop();
	}
}
