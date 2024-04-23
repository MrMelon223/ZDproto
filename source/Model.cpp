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
	in.open(this->filepath, std::ios_base::in);

	if (!in) {
		std::cout << "Cannot find model " << path << std::endl;
		return;
	}

	this->vertices = *new std::vector<Vertex>();
	this->triangles = *new std::vector<Tri>();

	size_t len;
	std::string line2;
	std::getline(in, line2);
	std::istringstream in_s(line2);
	in_s >> len;
	for (int i = 0; i < len; i++) {

		std::string type;
		std::getline(in, type);
		std::istringstream in2(type);

		float x, y, z;
		float r, g, b, a;
		float u, v;
		in2 >> x >> y >> z;
		in2 >> r >> g >> b >> a;
		in2 >> u >> v;

		this->vertices.push_back(Vertex{ glm::vec3(x, y, z), glm::cross(glm::vec3(x, y, z), glm::vec3(x, y, z)), glm::vec4(r, g, b, a), glm::vec2(u, v) });
	}
	std::string line;
	std::getline(in, line);
	std::istringstream in_s2(line);
	in_s2 >> len;

	for (int i = 0; i < len; i++) {
		std::string type;
		std::getline(in, type);
		std::istringstream in3(type);

		uint32_t x, y, z, w;
		in3 >> x >> y >> z;

		std::cout << "Triangle idxs = { " << x << ", " << y << ", " << z << " }" << std::endl;

		glm::vec3 t_norm = glm::cross(this->vertices.at(x).position - this->vertices.at(y).position, this->vertices.at(x).position - this->vertices.at(z).position),
			norm_w = (this->vertices.at(x).normal + this->vertices.at(y).normal + this->vertices.at(z).normal) / 3.0f;
		if (glm::dot(t_norm, norm_w) < 0.0f) {
			t_norm = -1.0f * t_norm;
		}

		this->triangles.push_back(Tri{ x, y, z, t_norm * norm_w});
	}
	this->color_map = Texture("resources/textures/default.png");

	in.close();
}

HostModel::HostModel() {

}

HostModel::HostModel(std::string path) {
	this->load_from(path);

	this->bvh = BVH(this);
	this->bvh.debug_print();
}

d_Model HostModel::to_gpu() {
	uint32_t v_count = static_cast<uint32_t>(this->vertices.size()), t_count = static_cast<uint32_t>(this->triangles.size());

	d_Model ret{ nullptr, nullptr, this, nullptr, nullptr, nullptr, NULL};

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

	ret.bvh = this->bvh.to_gpu();

	return ret;
}

d_ModelInstance create_instance(uint32_t model_index, glm::vec3 position, glm::vec3 rotation, uint32_t triangle_count, bool is_hitbox, float scale) {
	d_ModelInstance ret{};
	ret.model_index = model_index;
	ret.position = position;
	ret.rotation = rotation;

	ret.is_hitbox = is_hitbox;
	ret.scale = scale;

	error_check(cudaMalloc((void**)&ret.visible_triangles, triangle_count));

	return ret;
}

	// BVH

int32_t bvh_layer_count(size_t volume_count) {
	int32_t layer_count = 1;
	int32_t v_count = static_cast<int32_t>(volume_count);
	int32_t last = v_count;
	while (v_count > 1) {
		v_count -= last / 2;
		last = v_count;
		layer_count++;
	}
	return layer_count;
}

BVH::BVH() {

}

uint32_t find_closest(Tri* t, std::vector<Vertex> verts, std::vector<Tri> tris, bool* used, int cur_idx) {
	glm::vec3 center = (verts[t->a].position + verts[t->b].position + verts[t->c].position) * 0.333f;

	float closest = 100000.0f;
	int32_t tracker = 0, ret = -1;
	for (size_t i = 0; i < tris.size(); i++) {
		Tri tr = tris[i];
		if (!used[i] && i != cur_idx) {
			glm::vec3 center_test = (verts[tr.a].position + verts[tr.b].position + verts[tr.c].position) * 0.333f;
			if (ret == -1) {
				ret == i;
			}
			if ((center_test - center).length() < closest) {
				ret = i;
			}
		}
		tracker++;
	}
	return ret;
}

BVH::BVH(HostModel* model) {
	std::vector<Vertex> verts = *model->get_vertices();
	std::vector<Tri> tris = *model->get_triangles();
	if (tris.size() < BVH_TRIANGLE_COUNT) {
			BoundingVolume vol = {};
			//vol.triangles = (Tri**)malloc(BVH_TRIANGLE_COUNT * sizeof(uint32_t));
			for (uint32_t j = 0; j < tris.size(); j++) {
				vol.triangles[j] = j;
			}
			vol.triangle_count = static_cast<uint8_t>(tris.size());

			glm::vec3 min, max = min = verts.at(tris.at(vol.triangles[0]).a).position;
			for (uint8_t j = 1; j < vol.triangle_count; j++) {
				Tri* t = &tris.at(vol.triangles[j]);
				Vertex* va = &verts.at(t->a);
				Vertex* vb = &verts.at(t->b);
				Vertex* vc = &verts.at(t->c);
				min = glm::min(min, glm::min(glm::min(va->position, vb->position), vc->position));
				max = glm::max(min, glm::max(glm::max(va->position, vb->position), vc->position));
			}

			vol.vertices[0] = glm::vec3(min);
			vol.vertices[1] = glm::vec3(max);
			vol.is_base = true;

			BVHNode node = { vol, -1, -1 };

			this->nodes.push_back(node);
	}
	else {
		bool* used = new bool[tris.size()];


		for (size_t i = 0; i < tris.size(); i++) {
			used[i] = false;
		}

		uint32_t idx = 0;
		size_t count = 0;
		int32_t remaining = static_cast<uint32_t>(tris.size());

		//std::cout << "Model Vertex Count: " << verts.size() << std::endl;
		//std::cout << "Model Triangle Count: " << tris.size() << std::endl;
		while (remaining > 1) {
			//std::cout << "Remaining: " << remaining << std::endl;
			BoundingVolume vol = {};

			used[idx] = true;
			bool none_found = false;
			int cap = -1;
			for (uint32_t j = 0; j < BVH_TRIANGLE_COUNT; j++) {
				int found = find_closest(&tris[idx], verts, tris, used, count * BVH_TRIANGLE_COUNT + j);
				if (found >= 0) {
					vol.triangles[j] = static_cast<uint32_t>(found);
					used[vol.triangles[j]] = true;
					//printf("Triangle found = %i\n", vol.triangles[j]);
				}
				else {
					int32_t next = -1;
					for (int32_t k = 0; k < tris.size(); k++) {
						if (!used[k]) {
							next = k;
						}
					}
					if (next == -1) {
						none_found = true;
						cap = j;
						break;
					}
					vol.triangles[j] = next;
					used[vol.triangles[j]] = true;
				}
				//printf("Triangle found = %i\n", vol.triangles[j]);
				//used[vol.triangles[j]] = true;
			}
			if (none_found) {
				vol.triangle_count = cap;
			}
			else {
				vol.triangle_count = BVH_TRIANGLE_COUNT;
			}
			remaining -= vol.triangle_count;
			bool made_it = true;

			glm::vec3 min, max = min = verts.at(tris.at(vol.triangles[0]).a).position;
			for (uint8_t j = 1; j < vol.triangle_count; j++) {
				//std::cout << "Triangle " << vol.triangles[j] << std::endl;
				Tri* t = &tris.at(vol.triangles[j]);
				Vertex* va = &verts.at(t->a);
				Vertex* vb = &verts.at(t->b);
				Vertex* vc = &verts.at(t->c);

				min = glm::min(min, glm::min(glm::min(va->position, vb->position), vc->position));
				max = glm::max(min, glm::max(glm::max(va->position, vb->position), vc->position));
			}

			vol.vertices[0] = glm::vec3(min);
			vol.vertices[1] = glm::vec3(max);
			vol.is_base = true;

			BVHNode node = { vol, -1, -1 };

			this->nodes.push_back(node);

			count++;
		}
		delete[] used;
	}
	this->layers = bvh_layer_count(this->nodes.size());
	//std::cout << "Need Layers #" << this->layers << std::endl;
	int32_t last = static_cast<int32_t>(this->nodes.size());
	int32_t running_total = 0;
	if (this->nodes.size() > 1) {
		for (int32_t n = 1; n < this->layers; n++) {
			std::cout << "Layering tree for iteration " << n << std::endl;
			last = (last / 2);
			for (int32_t i = 0; i < last; i++) {
				BVHNode* n1 = &this->nodes[running_total],
					* n2 = &this->nodes[running_total + 1];

				glm::vec3 n1_min = n1->volume.vertices[0],
					n1_max = n1->volume.vertices[1];
				glm::vec3 n2_min = n2->volume.vertices[0],
					n2_max = n2->volume.vertices[1];

				glm::vec3 min = glm::min(n1_min, n2_min), max = glm::max(n1_max, n2_max);

				BoundingVolume vol{};
				vol.is_base = false;
				vol.vertices[0] = min;
				vol.vertices[1] = max;

				BVHNode n3 = {};
				n3.volume = vol;
				vol.is_top = false;
				if (last == 1) {
					n3.left_child_index = static_cast<uint32_t>(running_total);
					n3.right_child_index = static_cast<uint32_t>(-1);
				}
				else {
					n3.left_child_index = static_cast<uint32_t>(running_total);
					n3.right_child_index = static_cast<uint32_t>(running_total + 1);
				}

				printf("Children indexs = %i,%i\n", n3.left_child_index, n3.right_child_index);

				this->nodes.push_back(n3);
				running_total += 2;
			}
		}
		this->nodes.back().volume.is_top = true;
	}
}

d_BVH BVH::to_gpu() {
	d_BVH d_bvh{};
	error_check(cudaMalloc((void**)&d_bvh.nodes, sizeof(BVHNode) * this->nodes.size()));
	error_check(cudaMemcpy(d_bvh.nodes, this->nodes.data(), sizeof(BVHNode) * this->nodes.size(), cudaMemcpyHostToDevice));
	printf("BVH Node count = %i\n", this->nodes.size());
	d_bvh.initial = static_cast<uint32_t>(this->nodes.size() - 1);
	d_bvh.layers = this->layers;
	d_bvh.node_size = static_cast<uint32_t>(this->nodes.size());

	return d_bvh;
}

void BVH::debug_print() {
	for (BVHNode n : this->nodes) {
		printf("Node left = %i, Node right = %i\n", n.left_child_index, n.right_child_index);
		for (int32_t i = 0; i < n.volume.triangle_count; i++) {
			printf("     Tri %i = %i\n", i, n.volume.triangles[i]);
		}
	}
}