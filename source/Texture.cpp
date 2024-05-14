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

Material::Material() {

}

Material::Material(std::string path, std::string name) {
	this->path = path;
	this->name = name;

	this->load_from(this->path);
}

void Material::load_from(std::string m_path) {
	std::ifstream in;
	in.open(m_path.c_str(), std::ios::in);
	if (!in) {
		std::cout << "Cannot find Material file: " << m_path << std::endl;
		return;
	}

	std::string line;
	std::getline(in, line);
	std::istringstream color_tex(line);

	std::string color_path;
	color_tex >> color_path;
	if (color_path != "NULL") {
		this->color_map = Texture(color_path);
	}
	else {
		this->color_map = Texture(DEFAULT_TEXTURE);
	}

	std::getline(in, line);
	std::istringstream normal_tex(line);
	std::string normal_path;
	normal_tex >> normal_path;
	if (normal_path != "NULL") {
		this->normal_map = Texture(normal_path);
	}
	else {
		this->normal_map = Texture(DEFAULT_TEXTURE);
	}

	std::getline(in, line);
	std::istringstream gloss_tex(line);
	std::string gloss_path;
	gloss_tex >> gloss_path;
	if (gloss_path != "NULL") {
		this->gloss_map = Texture(gloss_path);
	}
	else {
		this->gloss_map = Texture(DEFAULT_TEXTURE);
	}

	std::getline(in, line);
	std::istringstream spec_tex(line);
	std::string spec_path;
	spec_tex >> spec_path;
	if (spec_path != "NULL") {
		this->specular_map = Texture(spec_path);
	}
	else {
		this->specular_map = Texture(DEFAULT_TEXTURE);
	}

	std::getline(in, line);
	std::istringstream ao_tex(line);
	std::string ao_path;
	ao_tex >> ao_path;
	if (ao_path != "NULL") {
		this->ambientocclusion_map = Texture(ao_path);
	}
	else {
		this->ambientocclusion_map = Texture(DEFAULT_TEXTURE);
	}

	std::getline(in, line);
	std::istringstream tess_tex(line);
	std::string tess_path;
	tess_tex >> tess_path;
	if (tess_path != "NULL") {
		this->tesselation_map = Texture(tess_path);
	}
	else {
		this->tesselation_map = Texture(DEFAULT_TEXTURE);
	}
}

d_Material Material::to_gpu() {
	d_Material m{};

	m.color_map = this->color_map.instance_of();
	m.normal_map = this->normal_map.instance_of();
	m.gloss_map = this->gloss_map.instance_of();
	m.specular_map = this->specular_map.instance_of();
	m.ambientocclusion_map = this->ambientocclusion_map.instance_of();
	m.tesselation_map = this->tesselation_map.instance_of();

	return m;
}