	// Texture.cpp

#define STB_IMAGE_IMPLEMENTATION
#include "../include/Texture.h"

Texture::Texture() {

}

Texture::Texture(std::string path) {
	this->path = path;
	this->name = extract_name(path);

	this->data = *new std::vector<glm::vec4>();

	int dims = 0;
	unsigned char* ptr = stbi_load(path.c_str(), &this->dims.x, &this->dims.y, &dims, 4);
	if (ptr == nullptr) {
		std::cout << "Cannot find texture: " << path << std::endl;
		free(ptr);
		return;
	}

	for (int y = 0; y < this->dims.y; y++) {
		for (int x = 0; x < this->dims.x; x++) {
			glm::vec4 cols(0.0f);
			if (dims == 4) {
				cols.x = static_cast<float>(ptr[y * 4 + (x * dims)]) / 255;
				cols.y = static_cast<float>(ptr[y * 4 + (x * dims) + 1]) / 255;
				cols.z = static_cast<float>(ptr[y * 4 + (x * dims) + 2]) / 255;
				cols.w = static_cast<float>(ptr[y * 4 + (x * dims) + 3]) / 255;
			}
			else if (dims <= 3) {
				cols.x = static_cast<float>(ptr[y * 3 + (x * dims)]) / 255;
				cols.y = static_cast<float>(ptr[y * 3 + (x * dims) + 1]) / 255;
				cols.z = static_cast<float>(ptr[y * 3 + (x * dims) + 2]) / 255;
				cols.w = 1.0f;
			}
			this->data.push_back(cols);
		}
	}
	free(ptr);
}

TextureInstance Texture::instance_of() {
	TextureInstance ret{};

	this->debug_print();

	ret.dims = this->dims;

	error_check(cudaMalloc((void**)&ret.data, sizeof(glm::vec4) * this->dims.y * this->dims.x));
	error_check(cudaMemcpy(ret.data, this->data.data(), sizeof(glm::vec4) * this->dims.y * this->dims.x, cudaMemcpyHostToDevice));

	return ret;
}

void Texture::debug_print() {
	std::cout << "Texture: " << this->name << std::endl;
	std::cout << std::setw(10) << "Dims= {" << this->dims.x << ", " << this->dims.y << " }" << std::endl;
}