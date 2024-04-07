#ifndef LEVEL_CUH
#define LEVEL_CUH


#include "Model.h"
#include "Runtime.h"
#include "Light.h"
#include "Object.h"

class Level {
protected:

	std::string name;

	thrust::host_vector<Object> objects;
	Object* d_objects;
	uint32_t d_object_count;

	thrust::host_vector<d_ModelInstance> model_instances;
	d_ModelInstance* d_model_instances;
	uint32_t d_model_instance_count;

	d_Model* d_DEVICE_MODELS;

	d_AmbientLight* d_ambient_light;
	d_PointLight* d_point_lights;
	uint32_t d_point_lights_count;

	void load_from(std::string);

public:
	Level();
	Level(std::string);

	void add_model_instance(d_ModelInstance);
	d_ModelInstance* get_d_model_instances() { return this->d_model_instances; }
	uint32_t get_d_model_instance_count() { return this->d_model_instance_count; }

	d_Model* get_d_device_models() { return this->d_DEVICE_MODELS; }

	d_AmbientLight* get_d_ambient_light() { return this->d_ambient_light; }
	d_PointLight* get_d_point_lights() { return this->d_point_lights; }
	uint32_t get_d_point_lights_size() { return this->d_point_lights_count; }

	void update_instance(uint32_t, d_ModelInstance);

	void upload_instances();

	void add_object(Object);
	void update_object(uint32_t, Object);
	Object* get_objects_ptr() { return thrust::raw_pointer_cast(this->objects.data()); }
	void upload_objects();

	//uint32_t get_instance_index(d_ModelInstance);

};

#endif