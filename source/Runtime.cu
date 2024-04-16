	// Runtime.cu
#include "../include/Runtime.h"

std::vector<HostModel> HOST_MODELS;
std::vector<d_Model> DEVICE_MODELS;

std::vector<Object> Runtime::OBJECTS;
std::vector<BulletWeapon> Runtime::WEAPONS;

int sql_callback(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		for (size_t i = 0; i < num_fields; i++) {
			HOST_MODELS.push_back(HostModel(p_fields[i]));
			DEVICE_MODELS.push_back(HOST_MODELS.back().to_gpu());
		}
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::runtime_load(sqlite3* sql) {
	std::string filepath = "resources/models.txt";

	const char* command = "SELECT path FROM assets;";
	char* err;

	HOST_MODELS = *new std::vector<HostModel>();
	DEVICE_MODELS = *new std::vector<d_Model>();

	int r = sqlite3_exec(sql, command, sql_callback, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL error: " << err << std::endl;
		return;
	}



	std::cout << "Loading complete!" << std::endl;
}

int sql_callback_object(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		for (size_t i = 0; i < num_fields; i++) {
			std::cout << p_fields[i] << std::endl;
		}
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::load_objects(sqlite3* sql) {
	const char* command = "SELECT * FROM objects;";
	char* err;

	Runtime::OBJECTS = *new std::vector<Object>();

	int r = sqlite3_exec(sql, command, sql_callback_object, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL error: " << err << std::endl;
		return;
	}
}

void Camera::add_to_euler_direction(glm::vec2 rot) {
	float x = rot.x, y = rot.y;
	//printf("X,Y input mouse coord = {%.2f, %.2f}\n", rot.x, rot.y);
	float normalized_coord_x = ((rot.x - (static_cast<float>(this->dims.x) * 0.5f)) / static_cast<float>(this->dims.x));
	float normalized_coord_y = ((rot.y - (static_cast<float>(this->dims.y) * 0.5f)) / static_cast<float>(this->dims.y));
	//printf("X,Y normalized input mouse coord = {%.2f, %.2f}\n", normalized_coord_x, normalized_coord_y);

	float aspect_ratio = static_cast<float>(this->dims.x) / static_cast<float>(this->dims.y);

	float fov_hori_rad = this->fov.x;
	float fov_vert_rad = this->fov.y;
	float half_fov_hori_rad = fov_hori_rad * 0.5f;
	float half_fov_vert_rad = fov_vert_rad * 0.5f;

	float view_x = normalized_coord_x * half_fov_hori_rad * aspect_ratio;
	float view_y = normalized_coord_y * half_fov_vert_rad;

	this->euler_direction.x += view_x * Runtime::X_SENSITIVITY * aspect_ratio; //* (static_cast<float>(this->dims.x) / this->dims.y);
	this->euler_direction.y -= view_y * Runtime::Y_SENSITIVITY;
	this->euler_direction.z = 0.0f;

	if (this->euler_direction.y > 90.0f) {
		this->euler_direction.y = 90.0f;
	}
	if (this->euler_direction.y < -90.0f) {
		this->euler_direction.y = -90.0f;
	}

	float yaw = this->euler_direction.x * (PI / 180.0f),
		pitch = this->euler_direction.y * (PI / 180.0f);

	this->direction.x = cosf(yaw) * cosf(pitch);
	this->direction.y = sinf(pitch);
	this->direction.z = sinf(yaw) * cosf(pitch);

	this->direction = glm::normalize(this->direction);
}

void Camera::forward(float t) {

	float mag = sqrtf(this->direction.x * this->direction.x + this->direction.y * this->direction.y + this->direction.z + this->direction.z);

	this->position += t * Runtime::BASE_SPEED * this->direction;
}

void Camera::backward(float t) {

	float mag = sqrtf(this->direction.x * this->direction.x + this->direction.y * this->direction.y + this->direction.z + this->direction.z);

	this->position -= t * Runtime::BASE_SPEED * this->direction;
}

void Camera::right(float t) {
	this->position += t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

void Camera::left(float t) {
	this->position -= t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

int sql_callback_weapon(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		for (size_t i = 0; i < num_fields; i++) {
			std::cout << p_fields[i] << std::endl;
		}
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::load_weapons(sqlite3* sql) {
	const char* command = "SELECT * FROM weapons;";
	char* err;

	Runtime::WEAPONS = *new std::vector<BulletWeapon>();

	int r = sqlite3_exec(sql, command, sql_callback_weapon, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL error: " << err << std::endl;
		return;
	}
}

bool Runtime::host_models_contains(std::string name) {
	for (HostModel mod : HOST_MODELS) {
		if (mod.get_name() == name) {
			return true;
		}
	}
	return false;
}

void Level::load_from(std::string path) {

	std::ifstream in;
	in.open(path, std::ios::in);
	if (!in) {
		std::cout << "Cannot find Level: " << path << std::endl;
		return;
	}

	std::cout << "Loading Level: " << path << std::endl;

	this->objects = *new thrust::host_vector<Object>();

	std::string line;
	std::getline(in, line);
	std::istringstream parse(line);
	size_t leng = 0;

	parse >> leng;
	std::cout << leng << " static models detected!" << std::endl;
	std::string model;

	thrust::host_vector<d_Model> d_models(DEVICE_MODELS.size());
	uint32_t model_count = 0;

	for (size_t i = 0; i < leng; i++) {
		std::getline(in, line);
		std::istringstream in0(line);

		float x, y, z, x_r, y_r, z_r;

		in0 >> x >> y >> z >> x_r >> y_r >> z_r >> model;
		//std::cout << model << std::endl;

		glm::vec3 position = glm::vec3(x, y, z);
		glm::vec3 rotation = glm::vec3(x_r, y_r, z_r);

		HostModel* h_model = Runtime::find_host_model(model);

		this->model_instances.push_back(create_instance(Runtime::find_host_model_index(model), position, rotation, Runtime::find_host_model(model)->get_triangle_count(), false));

		std::cout << "d_model = " << this->model_instances.back().model_index << std::endl;

		//d_Model d_model2 = Runtime::find_host_model(model)->to_gpu();

		//d_models.push_back(DEVICE_MODELS.at(Runtime::find_host_model_index(model)));	// -> Dis piece o shite

	}
	thrust::copy(DEVICE_MODELS.begin(), DEVICE_MODELS.end(), d_models.begin());
	size_t light_leng;
	std::string line2;
	std::getline(in, line2);
	std::istringstream parse2(line2);
	parse2 >> light_leng;
	std::cout << light_leng << " lights detected!" << std::endl;
	std::vector<d_PointLight> point_lights;
	for (size_t i = 0; i < light_leng; i++) {
		std::getline(in, line);
		std::istringstream in1(line);

		float x, y, z, r, g, b, a, s_r, s_g, s_b, s_a, intensity, falloff, range;

		in1 >> x >> y >> z >> r >> g >> b >> a >> s_r >> s_g >> s_b >> s_a >> intensity >> falloff >> range;
		//std::cout << model << std::endl;

		glm::vec3 position = glm::vec3(x, y, z);
		glm::vec4 color = glm::vec4(r, g, b, a);
		glm::vec4 s_color = glm::vec4(s_r, s_g, s_b, s_a);

		point_lights.push_back(d_PointLight{ position, color, s_color, intensity, falloff, range });
	}

	std::getline(in, line);
	std::istringstream in2(line);

	float r, g, b, a, s_r, s_g, s_b, s_a, intensity;

	in2 >> r >> g >> b >> a >> s_r >> s_g >> s_b >> s_a >> intensity;

	d_AmbientLight amb_light = { glm::vec4(r, g, b, a), glm::vec4(s_r, s_g, s_b, s_a), intensity };

	this->d_model_instance_count = static_cast<uint32_t>(this->model_instances.size());

	this->d_point_lights_count = static_cast<uint32_t>(point_lights.size());

	std::getline(in, line);
	std::istringstream in3(line);
	size_t object_count;
	in3 >> object_count;
	std::cout << object_count << " objects detected!" << std::endl;
	bool has_player = false;
	for (size_t i = 0; i < object_count; i++) {
		std::getline(in, line);
		std::istringstream in_obj(line);
		uint32_t type;
		float x, y, z, x_d, y_d, z_d;
		std::string visual_model, rigid_model;

		in_obj >> type >> x >> y >> z >> x_d >> y_d >> z_d >> visual_model;

		ObjectType obj_type = ObjectType::AI;
		std::cout << "Type " << type << std::endl;

		d_ModelInstance instance;
		uint32_t model_idx = 0, instance_idx = 0, hitbox_index = 0;

		if (type == 0) {

			obj_type = ObjectType::AI;

			instance = create_instance(Runtime::find_host_model_index(visual_model), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count(), false);

			this->model_instances.push_back(instance);

			model_idx = Runtime::find_host_model_index(visual_model);
			instance_idx = static_cast<uint32_t>(this->model_instances.size() - 1);

			this->model_instances.push_back(create_instance(Runtime::find_host_model_index("player_hitbox.txt"), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model("player_hitbox.txt")->get_triangle_count(), true));

			hitbox_index = static_cast<uint32_t>(this->model_instances.size() - 1);

			this->add_object(Object(obj_type, glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), model_idx, instance_idx, hitbox_index));
		}
		else if (type == 1) {
			obj_type = ObjectType::Physics;
			instance = create_instance(Runtime::find_host_model_index(visual_model), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count(), false);

			this->model_instances.push_back(instance);

			this->model_instances.push_back(create_instance(Runtime::find_host_model_index(visual_model), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count(), true));

			hitbox_index = static_cast<uint32_t>(this->model_instances.size() - 1);

			model_idx = Runtime::find_host_model_index(visual_model);
			instance_idx = static_cast<uint32_t>(this->model_instances.size() - 1);
			this->add_object(Object(obj_type, glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), model_idx, instance_idx, hitbox_index));
		}
		else if (type == 2 && !has_player) {
			has_player = true;
			obj_type = ObjectType::Player;

			instance = create_instance(Runtime::find_host_model_index(visual_model), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count(), false);

			this->model_instances.push_back(instance);

			model_idx = Runtime::find_host_model_index(visual_model);
			instance_idx = static_cast<uint32_t>(this->model_instances.size() - 1);

			this->model_instances.push_back(create_instance(Runtime::find_host_model_index("player_hitbox.txt"), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model("player_hitbox.txt")->get_triangle_count(), true));

			hitbox_index = static_cast<uint32_t>(this->model_instances.size() - 1);

			Object obj(obj_type, glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), 0, 0, hitbox_index);
			obj.attach_camera(this->camera_ptr);
			obj.set_health(100.0f);
			this->add_object(obj);

			this->PLAYER_OBJECT = new Object(ObjectType::Player, glm::vec3(0.0f, 0.0f, 0.0f));
			this->PLAYER_OBJECT = &this->objects[this->objects.size() - 1];
		}
		else if (type == 3) {
			obj_type = ObjectType::Weapon;


			instance = create_instance(Runtime::find_host_model_index(visual_model), this->PLAYER_OBJECT->get_position(), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count(), false);

			model_idx = Runtime::find_host_model_index(visual_model);

			this->model_instances.push_back(instance);

			instance_idx = static_cast<uint32_t>(this->model_instances.size() - 1);

			Object obj(obj_type, this->PLAYER_OBJECT->get_position(), this->PLAYER_OBJECT->get_direction(), model_idx, instance_idx, hitbox_index);

			obj.set_weapon_offset(glm::vec3(x, y, z));
			obj.set_in_inventory(false);
			obj.set_in_use(false);

			this->add_object(obj);
		}

		std::cout << "Object added: " << &this->objects.back() << " @ " << this->objects.back().get_model_index() << " index with model " << visual_model << " of type " << this->objects.back().get_object_type() << std::endl;
	}

	error_check(cudaMalloc((void**)&this->d_DEVICE_MODELS, sizeof(d_Model) * d_models.size()));
	error_check(cudaMemcpy(this->d_DEVICE_MODELS, thrust::raw_pointer_cast(d_models.data()), sizeof(d_Model) * d_models.size(), cudaMemcpyHostToDevice));
	this->d_DEVICE_MODEL_COUNT = static_cast<uint32_t>(d_models.size());

	error_check(cudaMalloc((void**)&this->d_ambient_light, sizeof(d_AmbientLight)));
	error_check(cudaMemcpy(this->d_ambient_light, &amb_light, sizeof(d_AmbientLight), cudaMemcpyHostToDevice));

	error_check(cudaMalloc((void**)&this->d_point_lights, sizeof(d_PointLight) * point_lights.size()));
	error_check(cudaMemcpy(this->d_point_lights, point_lights.data(), sizeof(d_PointLight) * point_lights.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	//this->upload_objects();

	//error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * this->model_instances.size()));
	//this->upload_instances();
}

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

Application::Application(int32_t dimx, int32_t dimy) {
	this->dims = glm::ivec2(dimx, dimy);

	int err = sqlite3_open("databases\\master.db", &this->database_connection);
	if (err != SQLITE_OK) {
		std::cout << "Cannot open asset database! : " << sqlite3_errmsg(this->database_connection) << std::endl;
		return;
	}

	Runtime::runtime_load(this->database_connection);
	Runtime::load_objects(this->database_connection);
	Runtime::load_weapons(this->database_connection);

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

			bool* intersection_tests = new bool[this->level->get_object_count()], * d_intersection_tests;
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
		//std::cout << "Updating " << this->level->get_object_count() << " objects in world!" << std::endl;
		for (size_t i = 0; i < this->level->get_object_count(); i++) {
			obj[i].update(&this->level->get_model_instances()[obj[i].get_instance_index()], &this->level->get_model_instances()[obj[i].get_hitbox_instance_index()], this->camera, glfwGetTime() - this->camera->get_last_time(), this->win, this->level->get_player_object());
			if (obj[i].get_object_type() == ObjectType::Player && obj[i].get_health() <= 0.0f) {
				std::cout << "Game Over" << std::endl;
				this->loop = false;
			}
		}

		if (frame_count != 0) {
			level->clean_d_objects();
		}

		this->level->upload_objects();
		this->level->upload_instances();

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



HostModel* Runtime::find_host_model(std::string name) {
	for (uint32_t i = 0; i < HOST_MODELS.size(); i++) {
		if (HOST_MODELS.at(i).get_name() == name) {
			return &HOST_MODELS.at(i);
		}
	}
	return &HOST_MODELS.at(0);	// Def fix this later, don't know what to do with it now
}

uint32_t Runtime::find_host_model_index(std::string name) {
	for (uint32_t i = 0; i < HOST_MODELS.size(); i++) {
		if (HOST_MODELS.at(i).get_name() == name) {
			return i;
		}
	}
	return 0;
}
