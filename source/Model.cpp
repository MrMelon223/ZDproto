	// Model.cpp
#include "../include/Model.h"

std::string extract_name(std::string path) {
	std::string bin_path = "";

	int last = path.length();
	for (int i = last - 1; i >= 0; i--) {
		if (path.c_str()[i] == '/' || path.c_str()[i] == '//' || path.c_str()[i] == '\\') {
			last = i;
			break;
		}
	}
	int last_dot = path.length();
	for (int i = last_dot - 1; i >= 0; i--) {
		if (path.c_str()[i] == '.') {
			last_dot = i;
			break;
		}
	}

	bin_path = path.substr(last + 1, last_dot - 1);

	return bin_path;
}

void HostModel::load_from(std::string path) {
	this->filepath = path;
	this->name = extract_name(this->filepath);

	std::ifstream in;
	in.open(path, std::ios_base::in);

	if (!in) {
		std::cout << "Cannot find model " << path << std::endl;
		return;
	}

	this->vertices = *new std::vector<Vertex>();
	this->triangles = *new std::vector<Tri>();

	size_t len;
	std::string line;
	std::getline(in, line);
	std::istringstream in_s(line);
	in_s >> len;
	for (int i = 0; i < len; i++) {
		std::getline(in, line);
		std::istringstream in(line);

		std::string type;

		float x, y, z;
		float r, g, b, a;
		float u, v;
		in >> x >> y >> z;
		in >> r >> g >> b >> a;
		in >> u >> v;

		this->vertices.push_back(Vertex{ glm::vec3(x, y, z), glm::cross(glm::vec3(x, y, z), glm::vec3(x, y, z)), glm::vec4(r, g, b, a), glm::vec2(u, v) });
	}

	std::getline(in, line);
	std::istringstream in_s2(line);
	in_s2 >> len;

	for (int i = 0; i < len; i++) {
		std::getline(in, line);
		std::istringstream in(line);

		std::string type;

		uint32_t x, y, z, w;
		in >> x >> y >> z;

		glm::vec3 t_norm = glm::cross(this->vertices.at(x).position - this->vertices.at(y).position, this->vertices.at(x).position - this->vertices.at(z).position),
			norm_w = (this->vertices.at(x).normal + this->vertices.at(y).normal + this->vertices.at(z).normal) / 3.0f;
		if (glm::dot(t_norm, norm_w) < 0.0f) {
			t_norm = -1.0f * t_norm;
		}

		this->triangles.push_back(Tri{ x, y, z, glm::cross(this->vertices.at(x).position - this->vertices.at(y).position, this->vertices.at(x).position - this->vertices.at(z).position) });
	}
	this->color_map = Texture("resources/textures/default.png");
}

HostModel::HostModel() {

}

HostModel::HostModel(std::string path) {
	this->load_from(path);
}

d_Model HostModel::to_gpu() {
	uint32_t v_count = static_cast<uint32_t>(this->vertices.size()), t_count = static_cast<uint32_t>(this->triangles.size());

	d_Model ret{ nullptr, nullptr, this, nullptr, nullptr, nullptr};

	error_check(cudaMalloc((void**)&ret.vertices, sizeof(Vertex) * v_count));
	error_check(cudaMalloc((void**)&ret.triangles, sizeof(Tri) * t_count));

	error_check(cudaMemcpy(ret.vertices, this->vertices.data(), sizeof(Vertex) * v_count, cudaMemcpyHostToDevice));
	error_check(cudaMemcpy(ret.triangles, this->triangles.data(), sizeof(Tri) * t_count, cudaMemcpyHostToDevice));

	uint32_t* vert_count, * tri_count;
	error_check(cudaMalloc((void**)&ret.vertex_count, sizeof(uint32_t)));
	error_check(cudaMalloc((void**)&ret.triangle_count, sizeof(uint32_t)));

	error_check(cudaMemcpy(ret.vertex_count, &v_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
	error_check(cudaMemcpy(ret.triangle_count, &t_count, sizeof(uint32_t), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();

	TextureInstance text = this->color_map.instance_of();

	error_check(cudaMalloc((void**)&ret.color_map, sizeof(TextureInstance)));
	error_check(cudaMemcpy(ret.color_map, &text, sizeof(TextureInstance), cudaMemcpyHostToDevice));

	return ret;
}

d_ModelInstance create_instance(uint32_t model_index, glm::vec3 position, glm::vec3 rotation, uint32_t triangle_count) {
	d_ModelInstance ret{};
	ret.model_index = model_index;
	ret.position = position;
	ret.rotation = rotation;

	error_check(cudaMalloc((void**)&ret.visible_triangles, triangle_count));

	return ret;
}