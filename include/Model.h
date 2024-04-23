#ifndef MODEL_H
#define MODEL_H

#include "Primitives.h"
#include "Texture.h"

struct d_Model;
class HostModel;

__device__
const int BVH_TRIANGLE_COUNT = 16;

struct BoundingVolume {
	uint32_t triangles[BVH_TRIANGLE_COUNT];
	uint8_t triangle_count;
	glm::vec3 vertices[2];
	bool is_base, is_top;
};

struct BVHNode {
	BoundingVolume volume;
	int32_t left_child_index, right_child_index;
};

struct d_BVH {
	BVHNode* nodes;
	uint32_t initial;
	uint32_t layers;
	uint32_t node_size;
};

class BVH {
protected:
	std::vector<BVHNode> nodes;
	uint32_t layers;
public:
	BVH();
	BVH(HostModel*);

	d_BVH to_gpu();

	void debug_print();
};

class HostModel {
protected:
	std::string filepath;
	std::string name;
	std::vector<Vertex> vertices;
	std::vector<Tri> triangles;

	Texture color_map;

	BVH bvh;

	void load_from(std::string);

public:
	HostModel();
	HostModel(std::string);

	std::string get_filepath() { return this->filepath; }
	std::string get_name() { return this->name; }

	uint32_t get_triangle_count() { return static_cast<uint32_t>(this->triangles.size()); }

	std::vector<Vertex>* get_vertices() { return &this->vertices; }
	std::vector<Tri>* get_triangles() { return &this->triangles; }

	d_Model to_gpu();
};

struct d_Model {
	uint32_t* vertex_count,* triangle_count;
	HostModel* host_ptr;
	Vertex* vertices;
	Tri* triangles;
	TextureInstance* color_map;
	d_BVH bvh;
};

struct d_ModelInstance {
	uint32_t model_index;
	bool* visible_triangles;
	glm::vec3 position, rotation;
	bool is_hitbox;
	float scale;
};

d_ModelInstance create_instance(uint32_t, glm::vec3, glm::vec3, uint32_t, bool, float);

#endif
