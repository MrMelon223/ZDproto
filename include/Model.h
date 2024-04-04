#ifndef MODEL_H
#define MODEL_H

#include "Primitives.h"
#include "Texture.h"

struct d_Model;

class HostModel {
protected:
	std::string filepath;
	std::string name;
	std::vector<Vertex> vertices;
	std::vector<Tri> triangles;

	Texture color_map;

	void load_from(std::string);

public:
	HostModel();
	HostModel(std::string);

	std::string get_filepath() { return this->filepath; }
	std::string get_name() { return this->name; }

	uint32_t get_triangle_count() { return static_cast<uint32_t>(this->triangles.size()); }

	d_Model to_gpu();
};

struct d_Model {
	uint32_t* vertex_count,* triangle_count;
	HostModel* host_ptr;
	Vertex* vertices;
	Tri* triangles;
	TextureInstance* color_map;
};

struct d_ModelInstance {
	uint32_t model_index;
	bool* visible_triangles;
	glm::vec3 position, rotation;
};

d_ModelInstance create_instance(uint32_t, glm::vec3, glm::vec3, uint32_t);

#endif
