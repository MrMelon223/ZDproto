#ifndef LEVEL_H
#define LEVEL_H

#include "Model.h"
#include "Runtime.h"

class Level {
protected:

	std::string name;

	std::vector<d_ModelInstance> model_instances;
	d_ModelInstance* d_model_instances;
	uint32_t d_model_instance_count;

	d_Model* d_DEVICE_MODELS;

	void load_from(std::string);
public:
	Level();
	Level(std::string);

	d_ModelInstance* get_d_model_instances() { return this->d_model_instances; }
	uint32_t get_d_model_instance_count() { return this->d_model_instance_count; }

	d_Model* get_d_device_models() { return this->d_DEVICE_MODELS; }

};

#endif
