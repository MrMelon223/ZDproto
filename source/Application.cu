	// Application.cpp
#include "../include/Application.h"

static void keyboard_callback(GLFWwindow* win, int key, int scancode, int action, int mods) {
	Runtime::KEY_USED = true;

	//std::cout << "Key Calledback!" << std::endl;

	Runtime::CURRENT_KEY = key;
	Runtime::CURRENT_SCANCODE = scancode;
	Runtime::CURRENT_ACTION = action;
	Runtime::CURRENT_MODS = mods;
}

static void mouse_callback(GLFWwindow* window, int button, int action, int mods) {
	Runtime::MOUSE_USED = true;

	Runtime::CURRENT_MOUSE = button;
	Runtime::CURRENT_ACTION = action;
	Runtime::CURRENT_MODS = mods;
}

Application::Application() {

}

Application::Application(int32_t dimx, int32_t dimy) {
	this->dims = glm::ivec2(dimx, dimy);

	glfwInit();
	glfwSwapInterval(0);
	glfwWindowHint(GLFW_CENTER_CURSOR, GLFW_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	this->win = glfwCreateWindow(this->dims.x, this->dims.y, "ZDproto-demo v0.00", NULL, NULL);
	this->loop = true;

	glm::vec3 init_position = glm::vec3(10.0f, 10.0f, 0.0f);
	glm::vec3 init_direction = glm::vec3(0.0f, 0.0f, 0.0f);

	this->camera = new Camera(this->dims, 120.0f, init_position, init_direction);

	this->level = new Level("resources/levels/test_level.txt", this->camera);

	//this->ai.push_back(FiniteStateMachine(glm::vec3(10.0f, 0.0f, 0.0f), this->level, ));

	glfwMakeContextCurrent(this->win);
}

void Application::input_handle() {
	std::cout << Runtime::CURRENT_KEY << std::endl;
	switch (Runtime::CURRENT_KEY) {
	case GLFW_KEY_W:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT || Runtime::CURRENT_ACTION == GLFW_RELEASE) {
			this->camera->forward(glfwGetTime() - this->camera->get_last_time());
		}
		break;
	case GLFW_KEY_S:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			this->camera->backward(glfwGetTime() - this->camera->get_last_time());
		}
		break;
	case GLFW_KEY_A:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			this->camera->left(glfwGetTime() - this->camera->get_last_time());
		}
		break;
	case GLFW_KEY_D:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			this->camera->right(glfwGetTime() - this->camera->get_last_time());
		}
		break;
	case GLFW_KEY_R:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			//this->cam->set_capture_mode(RT);
		}
		break;
	case GLFW_KEY_F:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			//this->camera->set_capture_mode(FULLBRIGHT);
		}
		break;
	case GLFW_KEY_ESCAPE:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			this->loop = false;
		}
		break;
	case GLFW_KEY_P:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			//this->cam->get_rays(0)->debug_stats();
		}
		break;
	case GLFW_MOUSE_BUTTON_LEFT:
		if (Runtime::CURRENT_ACTION == GLFW_REPEAT) {

		}
		break;
	}
}

void Application::mouse_handle() {
	switch (Runtime::CURRENT_MOUSE) {
	case GLFW_MOUSE_BUTTON_LEFT:
		if (Runtime::CURRENT_ACTION == GLFW_PRESS || Runtime::CURRENT_ACTION == GLFW_REPEAT) {
			glm::vec3 fire_direction = this->camera->get_direction();

			bool* intersection_tests = new bool[this->level->get_object_count()],* d_intersection_tests;
			error_check(cudaMalloc((void**)&d_intersection_tests, this->level->get_object_count() * sizeof(bool)), "Application::mouse_handle cudaMalloc");
			//std::cout << this->level->get_d_object_count() << " Objects going in @ " << this->level->get_d_objects() << std::endl;
			test_intersection << <(this->level->get_d_object_count() / 128) + 1, 128 >> > (this->camera->get_position(), fire_direction, this->level->get_d_objects(), this->level->get_d_object_count(), this->level->get_d_model_instances(), this->level->get_d_model_instance_count(), this->level->get_d_device_models(), d_intersection_tests);
			cudaDeviceSynchronize();
			error_check(cudaGetLastError(), "Application::mouse_handle kernel call");

			error_check(cudaMemcpy(intersection_tests, d_intersection_tests, this->level->get_object_count() * sizeof(bool), cudaMemcpyDeviceToHost), "Application::mouse_handle cudaMemcpy");
			
			for (size_t i = 0; i < this->level->get_object_count(); i++) {
				if (intersection_tests[i]) {
					Object objs = this->level->get_objects_ptr()[i];
					objs.set_health(objs.get_health() - 25.0f);
					if (objs.get_health() <= 0.0f) {
						objs.set_position(objs.get_spawn_point());
						objs.set_health(50.0f);
					}
					this->level->update_object(static_cast<uint32_t>(i), objs);
				}
			}

			error_check(cudaFree(d_intersection_tests));
		}
		break;

	}
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

void Application::main_loop() {
	this->camera->set_last_time(glfwGetTime());
	glfwSetKeyCallback(this->win, keyboard_callback);
	glfwSetMouseButtonCallback(this->win, mouse_callback);
	glfwMakeContextCurrent(this->win);

	int frame_count = 0;
	while (this->loop && !glfwWindowShouldClose(this->win)) {

		glfwPollEvents();
		this->frame_buffer = new glm::vec4[this->dims.x * this->dims.y];
		glm::vec4* d_frame_buffer;

		error_check(cudaMalloc((void**)&d_frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x));

		this->camera->new_frame();

		this->zero_frame_buffer_sse();

		error_check(cudaMemcpy(d_frame_buffer, this->frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x, cudaMemcpyHostToDevice));

		Object* obj = this->level->get_objects_ptr();
		std::cout << "Updating " << this->level->get_object_count() << " objects in world!" << std::endl;
		for (size_t i = 0; i < this->level->get_object_count(); i++) {
			obj[i].update(&this->level->get_model_instances()[obj[i].get_instance_index()], this->camera, glfwGetTime() - this->camera->get_last_time(), this->win);
		}

		if (frame_count != 0) {
			level->clean_d_objects();
		}

		this->level->upload_objects();
		this->level->upload_instances();
		cudaDeviceSynchronize();

		if (Runtime::KEY_USED) {
			this->input_handle();
			Runtime::control::reset_key();
		}
		if (Runtime::MOUSE_USED) {
			this->mouse_handle();
			Runtime::control::reset_mouse();
		}

			// Render functions
		this->camera->capture(this->level->get_d_model_instances(), this->level->get_d_model_instance_count(), this->level->get_d_device_models(), this->level->get_d_ambient_light(), this->level->get_d_point_lights(), this->level->get_d_point_lights_size(), d_frame_buffer);

		//this->camera->copy_to_frame_buffer(this->frame_buffer, 0);

		error_check(cudaMemcpy(this->frame_buffer, d_frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x, cudaMemcpyDeviceToHost));


		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawPixels(this->dims.x, this->dims.y, GL_BGRA_EXT, GL_FLOAT, this->frame_buffer);
		glfwSwapBuffers(this->win);

		this->camera->cleanup_frame();

		delete this->frame_buffer;
		cudaFree(d_frame_buffer);

		this->camera->debug_print();
		//this->cam->last_time = glfwGetTime();
		this->camera->set_last_time(glfwGetTime());
		frame_count++;
	}
	glfwDestroyWindow(this->win);
	glfwTerminate();
}