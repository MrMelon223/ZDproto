	// Runtime.cpp
#include "../include/Runtime.h"

std::vector<HostModel> HOST_MODELS;
std::vector<d_Model> DEVICE_MODELS;

void Runtime::runtime_load() {
	std::string filepath= "resources/models.txt";

	std::ifstream in;
	in.open(filepath, std::ios::in);
	if (!in) {
		std::cout << "Cannot find Runtime Model List: " << filepath << std::endl;
		return;
	}

	HOST_MODELS = *new std::vector<HostModel>();
	DEVICE_MODELS = *new std::vector<d_Model>();

	std::string path;
	while (std::getline(in, path)) {
		std::cout << "Loading model: " << path << std::endl;
		HostModel model = HostModel(path);
		HOST_MODELS.push_back(model);
		DEVICE_MODELS.push_back(model.to_gpu());
		std::cout << "Became -> " << HOST_MODELS.back().get_name() << std::endl;
		std::cout << "GPU ptr -> " << &DEVICE_MODELS.back() << std::endl;
	}
	std::cout << "Loading complete!" << std::endl;
}

bool Runtime::host_models_contains(std::string name) {
	for (HostModel mod : HOST_MODELS) {
		if (mod.get_name() == name) {
			return true;
		}
	}
	return false;
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
