	// Level.cpp
#include "../include/Level.h"

void Level::load_from(std::string path) {

	std::ifstream in;
	in.open(path, std::ios::in);
	if (!in) {
		std::cout << "Cannot find Level: " << path << std::endl;
		return;
	}

	std::string line;
	std::getline(in, line);
	std::istringstream parse(line);
	size_t leng = 0;

	parse >> leng;
	//std::cout << leng << std::endl;
	std::string model;

	//Runtime::HOST_MODELS = *new std::vector<HostModel>();
	//Runtime::DEVICE_MODELS = *new std::vector<d_Model>();
	std::vector<d_Model> d_models;
	uint32_t model_count = 0;

	for (size_t i = 0; i < leng; i++) {
		std::getline(in, line);
		std::istringstream in(line);

		float x, y, z, x_r, y_r, z_r;

		in >> x >> y >> z >> x_r >> y_r >> z_r;
		in >> model;
		//std::cout << model << std::endl;

		glm::vec3 position = glm::vec3(x, y, z);
		glm::vec3 rotation = glm::vec3(x_r, y_r, z_r);

		HostModel* h_model;

		if (Runtime::host_models_contains(model)) {
			h_model = Runtime::find_host_model(model);
		}
		else {
			h_model = new HostModel("resources/models/" + model);

			Runtime::HOST_MODELS.push_back(*h_model);
			Runtime::DEVICE_MODELS.push_back(h_model->to_gpu());
			model_count++;

			d_models.push_back(Runtime::DEVICE_MODELS.back());
		}

		d_Model* d_model = &Runtime::DEVICE_MODELS.back();

		this->model_instances.push_back(create_instance(model_count, position, rotation, h_model->get_triangle_count()));
	}

	this->d_model_instance_count = static_cast<uint32_t>(this->model_instances.size());

	error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * this->model_instances.size()));
	error_check(cudaMemcpy(this->d_model_instances, this->model_instances.data(), sizeof(d_ModelInstance) * this->model_instances.size(), cudaMemcpyHostToDevice));

	error_check(cudaMalloc((void**)&this->d_DEVICE_MODELS, sizeof(d_Model) * d_models.size()));
	error_check(cudaMemcpy(this->d_DEVICE_MODELS, d_models.data(), sizeof(d_Model) * d_models.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
}

Level::Level() {

}

Level::Level(std::string path) {
	this->load_from(path);
}

