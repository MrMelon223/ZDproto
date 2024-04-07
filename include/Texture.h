#ifndef TEXTURE_H
#define TEXTURE_H

#include "Helper.cuh"

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

	void debug_print();
};

#endif
