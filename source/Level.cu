
#include "../include/Level.cuh"

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

		this->model_instances.push_back(create_instance(Runtime::find_host_model_index(model), position, rotation, Runtime::find_host_model(model)->get_triangle_count()));

		std::cout << "d_model = " << this->model_instances.back().model_index << std::endl;

		//d_Model d_model2 = Runtime::find_host_model(model)->to_gpu();

		//d_models.push_back(DEVICE_MODELS.at(Runtime::find_host_model_index(model)));	// -> Dis piece o shite

	}
	thrust::copy(DEVICE_MODELS.begin(), DEVICE_MODELS.end(),d_models.begin());
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
	for (size_t i = 0; i < object_count; i++) {
		std::getline(in, line);
		std::istringstream in_obj(line);
		uint8_t type;
		float x, y, z, x_d, y_d, z_d;
		std::string visual_model, rigid_model;

		in_obj >> type >> x >> y >> z >> x_d >> y_d >> z_d >> visual_model;

		ObjectType obj_type;
		switch (type) {
		case 0:
			obj_type = AI;
			break;
		case 1:
			obj_type = Physics;
			break;
		case 2:
			obj_type = Player;
			break;
		}

		d_ModelInstance instance = create_instance(Runtime::find_host_model_index(visual_model), glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model(visual_model)->get_triangle_count());

		this->model_instances.push_back(instance);

		this->add_object(Object(obj_type, glm::vec3(x, y, z), glm::vec3(x_d, y_d, z_d), Runtime::find_host_model_index(visual_model), static_cast<uint32_t>(this->model_instances.size() - 1)));

		std::cout << "Object added: " << &this->objects.back() << " @ " << this->objects.back().get_model_index() << " index with model " << visual_model << std::endl;

	}

	error_check(cudaMalloc((void**)&this->d_DEVICE_MODELS, sizeof(d_Model) * d_models.size()));
	error_check(cudaMemcpy(this->d_DEVICE_MODELS, thrust::raw_pointer_cast(d_models.data()), sizeof(d_Model) * d_models.size(), cudaMemcpyHostToDevice));

	error_check(cudaMalloc((void**)&this->d_ambient_light, sizeof(d_AmbientLight)));
	error_check(cudaMemcpy(this->d_ambient_light, &amb_light, sizeof(d_AmbientLight), cudaMemcpyHostToDevice));

	error_check(cudaMalloc((void**)&this->d_point_lights, sizeof(d_PointLight) * point_lights.size()));
	error_check(cudaMemcpy(this->d_point_lights, point_lights.data(), sizeof(d_PointLight) * point_lights.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	//this->upload_objects();

	//error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * this->model_instances.size()));
	//this->upload_instances();
}

Level::Level() {
}


Level::Level(std::string path) {
	std::cout << "Initializing level from: " << path << std::endl;
	this->load_from(path);
}

void Level::add_model_instance(d_ModelInstance inst) {
	this->model_instances.push_back(inst);
}

void Level::update_instance(uint32_t index, d_ModelInstance model) {
	this->model_instances[index] =  model;
}

void Level::update_object(uint32_t index, Object object) {
	this->objects[index] = object;

	d_ModelInstance instance = this->model_instances[this->objects[index].get_instance_index()];
	instance.position = this->objects[index].get_position();
	instance.rotation = this->objects[index].get_direction();
	this->update_instance(this->objects[index].get_instance_index(), instance);
}

void Level::upload_instances() {
	if (this->d_model_instances != nullptr) {
		error_check(cudaFree(this->d_model_instances));
	}
	error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * this->model_instances.size()));
	error_check(cudaMemcpy(this->d_model_instances, this->model_instances.data(), sizeof(d_ModelInstance) * this->model_instances.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	this->d_model_instance_count = static_cast<uint32_t>(this->model_instances.size());
}

void Level::add_object(Object obj) {
	this->objects.push_back(obj);

	this->model_instances.push_back(create_instance(obj.get_model_index(), obj.get_position(), obj.get_direction(), HOST_MODELS[obj.get_model_index()].get_triangle_count()));
}

void Level::upload_objects() {
	if (this->d_model_instances != nullptr) {
		error_check(cudaFree(this->d_objects));
	}
	error_check(cudaMalloc((void**)&this->d_objects, sizeof(Object) * this->objects.size()));
	error_check(cudaMemcpy(this->d_objects, this->objects.data(), sizeof(Object) * this->objects.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	this->d_object_count = static_cast<uint32_t>(this->objects.size());
}

uint32_t get_instance_index(d_ModelInstance);