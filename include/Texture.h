#ifndef TEXTURE_H
#define TEXTURE_H

#include "Helper.cuh"

static const std::string DEFAULT_TEXTURE = "resources/textures/default.png";

struct TextureInstance {
	glm::ivec2 dims;
	glm::vec4* data;
};

class Texture {
protected:
	std::string name;
	std::string path;

	glm::ivec2 dims;
	std::vector<glm::vec4> data;

	std::string beautify_name(std::string);
public:
	Texture();
	Texture(std::string);

	glm::ivec2 get_dims() { return this->dims; }

	TextureInstance instance_of();

	std::string get_name() { return this->name; }
	std::string get_path() { return this->path; }

	void debug_print();
};

struct d_Material {
	TextureInstance color_map,
		normal_map,
		gloss_map,
		specular_map,
		ambientocclusion_map,
		tesselation_map;
};

class Material {
protected:
	std::string name;
	std::string path;

	Texture color_map,
		normal_map,
		gloss_map,
		specular_map,
		ambientocclusion_map,
		tesselation_map;

	void load_from(std::string);
public:

	Material();
	Material(std::string, std::string);

	d_Material to_gpu();

	std::string get_name() { return this->name; }
};


#endif
