	// Runtime.cu
#include "../include/Runtime.h"

std::vector<HostModel> HOST_MODELS;
std::vector<d_Model> DEVICE_MODELS;

std::vector<Material> HOST_MATERIALS;
std::vector<d_Material> DEVICE_MATERIALS;

std::vector<Object> Runtime::OBJECTS;
std::vector<BulletWeapon> Runtime::WEAPONS;

Object* Runtime::PLAYER_OBJECT;
std::vector<d_ModelInstance> Runtime::model_instances;

int sql_callback_material(void* p_data, int num_fields, char** p_fields, char** p_col_names) {
	try {
		std::cout << "Fields: " << num_fields << std::endl;
		std::cout << "     Loading Material " << p_fields[0] << std::endl;
		HOST_MATERIALS.push_back(Material(std::string(p_fields[0]), std::string(p_fields[1])));
		DEVICE_MATERIALS.push_back(HOST_MATERIALS.back().to_gpu());
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::load_materials(sqlite3* sql) {
	const char* command = "SELECT * from material_assets";
	char* err;

	HOST_MATERIALS = *new std::vector<Material>();
	DEVICE_MATERIALS = *new std::vector<d_Material>();

	std::cout << std::setw(15) << "Loading Materials..." << std::endl;

	int r = sqlite3_exec(sql, command, sql_callback_material, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL Material error: " << err << std::endl;
		return;
	}
}

int sql_callback(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		std::cout << "Fields: " << num_fields << std::endl;
		std::cout << "     Loading Asset " << p_fields[0] << std::endl;
		HOST_MODELS.push_back(HostModel(std::string(p_fields[0]), std::string(p_fields[1])));
		DEVICE_MODELS.push_back(HOST_MODELS.back().to_gpu());
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::runtime_load(sqlite3* sql) {
	std::string filepath = "resources/models.txt";

	const char* command = "SELECT * FROM assets;";
	char* err;

	HOST_MODELS = *new std::vector<HostModel>();
	DEVICE_MODELS = *new std::vector<d_Model>();

	int r = sqlite3_exec(sql, command, sql_callback, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL Model error: " << err << std::endl;
		return;
	}

}

int sql_callback_object(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		std::cout << "	Objects:" << std::endl;
			Object o;

			std::string obj_name, vis_model_name, hitbox_model_name;
			int32_t obj_type;
			float mass_kg;
			int bullet_coll;

			for (size_t i = 0; i < num_fields; i++) {
				char* str = p_fields[i];
				std::istringstream in(str);

				switch (i) {
				case 0:
					in >> obj_name;

					break;
				case 1:
					in >> vis_model_name;
					break;
				case 2:
					in >> hitbox_model_name;
					break;
				case 3:
					in >> mass_kg;
					break;
				case 4:
					in >> obj_type;
					break;
				case 5:
					in >> bullet_coll;
					break;
				}

				std::cout << str << std::endl;
			}
			if (obj_type == 0) {
				ObjIndexs obj_idx;

				d_ModelInstance d_mi = create_instance(Runtime::find_host_model_index(vis_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(vis_model_name)->get_triangle_count(), false, 1.0f);
				Runtime::model_instances.push_back(d_mi);
				uint32_t d_mi_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);
				d_ModelInstance d_hitbox = create_instance(Runtime::find_host_model_index(hitbox_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(hitbox_model_name)->get_triangle_count(), true, 1.0f);
				Runtime::model_instances.push_back(d_hitbox);
				uint32_t d_hitbox_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

				obj_idx.physics_object_index = Runtime::OBJECTS.size() - 1;
				o = Object(ObjectType::AI, obj_idx, obj_name, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f),Runtime::find_host_model_index(vis_model_name),  d_mi_idx, d_hitbox_idx);
				o.set_mass(mass_kg);
				o.set_max_health(100.0f);
				o.set_health(100.0f);
			}
			else if (obj_type == 1) {
				ObjIndexs obj_idx;

				d_ModelInstance d_mi = create_instance(Runtime::find_host_model_index(vis_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(vis_model_name)->get_triangle_count(), false, 1.0f);
				Runtime::model_instances.push_back(d_mi);
				uint32_t d_mi_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);
				d_ModelInstance d_hitbox = create_instance(Runtime::find_host_model_index(hitbox_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(hitbox_model_name)->get_triangle_count(), true, 1.0f);
				Runtime::model_instances.push_back(d_hitbox);
				uint32_t d_hitbox_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

				obj_idx.physics_object_index = Runtime::OBJECTS.size() - 1;
				o = Object(ObjectType::Physics, obj_idx, obj_name, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model_index(vis_model_name), d_mi_idx, d_hitbox_idx);
				o.set_mass(mass_kg);
			}
			else if (obj_type == 2) {
				ObjIndexs obj_idx;

				d_ModelInstance d_mi = create_instance(Runtime::find_host_model_index(vis_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), Runtime::find_host_model(vis_model_name)->get_triangle_count(), false, 1.0f);
				Runtime::model_instances.push_back(d_mi);
				uint32_t d_mi_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);
				d_ModelInstance d_hitbox = create_instance(Runtime::find_host_model_index(hitbox_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), Runtime::find_host_model(hitbox_model_name)->get_triangle_count(), true, 1.0f);
				Runtime::model_instances.push_back(d_hitbox);
				uint32_t d_hitbox_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

				obj_idx.player_index = 0;
				o = Object(ObjectType::Player, obj_idx, obj_name, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(1.0f, 0.0f, 0.0f), Runtime::find_host_model_index(vis_model_name), d_mi_idx, d_hitbox_idx);
				o.set_mass(mass_kg);
			}
			else if (obj_type == 3) {
				ObjIndexs obj_idx;

				d_ModelInstance d_mi = create_instance(Runtime::find_host_model_index(vis_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(vis_model_name)->get_triangle_count(), false, 0.001f);
				Runtime::model_instances.push_back(d_mi);
				uint32_t d_mi_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);
				d_ModelInstance d_hitbox = create_instance(Runtime::find_host_model_index(hitbox_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(hitbox_model_name)->get_triangle_count(), true, 0.001f);
				Runtime::model_instances.push_back(d_hitbox);
				uint32_t d_hitbox_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

				obj_idx.weapon_index= Runtime::find_weapon_index("Default Weapon");
				o = Object(ObjectType::Weapon, obj_idx, obj_name, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model_index(vis_model_name), d_mi_idx, d_hitbox_idx);
				o.set_mass(mass_kg);
			}
			else if (obj_type == 4) {
				ObjIndexs obj_idx;

				d_ModelInstance d_mi = create_instance(Runtime::find_host_model_index(vis_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(vis_model_name)->get_triangle_count(), false, 1.0f);
				Runtime::model_instances.push_back(d_mi);
				uint32_t d_mi_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);
				d_ModelInstance d_hitbox = create_instance(Runtime::find_host_model_index(hitbox_model_name), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(hitbox_model_name)->get_triangle_count(), true, 1.0f);
				Runtime::model_instances.push_back(d_hitbox);
				uint32_t d_hitbox_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

				obj_idx.static_index = Runtime::OBJECTS.size();
				o = Object(ObjectType::Static, obj_idx, obj_name, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model_index(vis_model_name), d_mi_idx, d_hitbox_idx);
				o.set_mass(mass_kg);
			}
			Runtime::OBJECTS.push_back(o);

			if (o.get_object_type() == ObjectType::Player) {
				Runtime::PLAYER_OBJECT = &Runtime::OBJECTS[Runtime::OBJECTS.size() - 1];
			}
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::load_objects(sqlite3* sql) {
	const char* command = "SELECT * FROM object_assets;";
	char* err;

	std::cout << "Loading Models" << std::endl << std::endl;

	std::cout << "	Loading Objects" << std::endl << std::endl;

	Runtime::OBJECTS = *new std::vector<Object>();

	int r = sqlite3_exec(sql, command, sql_callback_object, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL Objects error: " << err << std::endl;
		return;
	}
}

Object* find_object(std::string n) {
	uint32_t c = 0;
	for (Object o : Runtime::OBJECTS) {
		if (o.get_name() == n) {
			return Runtime::OBJECTS.data() + (sizeof(Object) * c);
		}
		c++;
	}
}
uint32_t Runtime::find_object_index(std::string n) {
	uint32_t c = 0;
	for (Object o : Runtime::OBJECTS) {
		if (o.get_name() == n) {
			return c;
		}
		c++;
	}
}

uint32_t Runtime::find_weapon_index(std::string n) {
	uint32_t c = 0;
	for (BulletWeapon w : Runtime::WEAPONS) {
		if (w.get_name() == n) {
			return c;
		}
		c++;
	}
}

bool Runtime::objects_collide(Object* a, Object* b) {
	BVHNode* a_node = HOST_MODELS[Runtime::model_instances[a->get_hitbox_instance_index()].model_index].get_initial();
	BVHNode* b_node = HOST_MODELS[Runtime::model_instances[b->get_hitbox_instance_index()].model_index].get_initial();

	return ((a_node->volume.vertices[0].x <= b_node->volume.vertices[1].x && a_node->volume.vertices[1].x >= b_node->volume.vertices[0].x) || (a_node->volume.vertices[0].y <= b_node->volume.vertices[1].y && a_node->volume.vertices[1].y >= b_node->volume.vertices[0].y) ||
		(a_node->volume.vertices[0].z <= b_node->volume.vertices[1].z && a_node->volume.vertices[1].z >= b_node->volume.vertices[0].z));
}

void Camera::add_to_euler_direction(glm::vec2 rot) {
	float x = rot.x, y = rot.y;
	//printf("X,Y input mouse coord = {%.2f, %.2f}\n", rot.x, rot.y);
	float normalized_coord_x = ((rot.x - (static_cast<float>(this->dims.x) * 0.5f)) / static_cast<float>(this->dims.x));
	float normalized_coord_y = ((rot.y - (static_cast<float>(this->dims.y) * 0.5f)) / static_cast<float>(this->dims.y));
	//printf("X,Y normalized input mouse coord = {%.2f, %.2f}\n", normalized_coord_x, normalized_coord_y);

	float aspect_ratio = static_cast<float>(this->dims.x) / static_cast<float>(this->dims.y);

	float fov_hori_rad = this->fov.x;
	float fov_vert_rad = this->fov.y;
	float half_fov_hori_rad = fov_hori_rad * 0.5f;
	float half_fov_vert_rad = fov_vert_rad * 0.5f;

	float view_x = normalized_coord_x * half_fov_hori_rad * aspect_ratio;
	float view_y = normalized_coord_y * half_fov_vert_rad;

	this->euler_direction.x += view_x * Runtime::X_SENSITIVITY * aspect_ratio; //* (static_cast<float>(this->dims.x) / this->dims.y);
	this->euler_direction.y -= view_y * Runtime::Y_SENSITIVITY;
	this->euler_direction.z = 0.0f;

	if (this->euler_direction.y > 90.0f) {
		this->euler_direction.y = 90.0f;
	}
	if (this->euler_direction.y < -90.0f) {
		this->euler_direction.y = -90.0f;
	}

	float yaw = this->euler_direction.x * (PI / 180.0f),
		pitch = this->euler_direction.y * (PI / 180.0f);

	this->direction.x = cosf(yaw) * cosf(pitch);
	this->direction.y = sinf(pitch);
	this->direction.z = sinf(yaw) * cosf(pitch);

	this->direction = glm::normalize(this->direction);
}

void Camera::forward(float t) {

	float mag = sqrtf(this->direction.x * this->direction.x + this->direction.y * this->direction.y + this->direction.z + this->direction.z);

	this->position += t * Runtime::BASE_SPEED * this->direction;
}

void Camera::backward(float t) {

	float mag = sqrtf(this->direction.x * this->direction.x + this->direction.y * this->direction.y + this->direction.z + this->direction.z);

	this->position -= t * Runtime::BASE_SPEED * this->direction;
}

void Camera::right(float t) {
	this->position += t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

void Camera::left(float t) {
	this->position -= t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

int sql_callback_weapon(void* p_data, int num_fields, char** p_fields, char** p_col_names)
{
	try {
		std::cout << "	Weapons:" << std::endl;
		for (size_t h = 0; h < (num_fields / 12); h++) {
			BulletWeapon w = BulletWeapon();

				for (size_t i = 0; i < 12; i++) {
					char* str = p_fields[h * 12 + i];
					std::istringstream in(str);

					int wpn_type = -1;
					WeaponType type;

					float w_speed = 0.0f;
					float r_speed = 0.0f;
					int32_t b_dmg = 0;
					float d_range = 0.0f;
					float f_delay = 0.01f;
					float mass_kg;
					std::string model_name;

					uint32_t model_index = 0;
					d_ModelInstance d_m;
					uint32_t inst_idx = 0;

					switch (i) {
					case 0:
						w.set_name(str);
						break;
					case 1:
						in >> wpn_type;

						type = WeaponType::SemiAutomatic;
						if (wpn_type == 0) {
							type = WeaponType::SemiAutomatic;
						}
						else if (wpn_type == 1) {
							type = WeaponType::FullyAutomatic;
						}
						else if (wpn_type == 2) {
							type = WeaponType::Burst;
						}
						else if (wpn_type == 3) {
							type = WeaponType::BoltAction;
						}
						else if (wpn_type == 4) {
							type = WeaponType::SingleShot;
						}
						else if (wpn_type == 5) {
							type = WeaponType::Melee;
						}

						w.set_weapon_type(type);

						break;
					case 2:
						in >> model_name;

						model_index = Runtime::find_host_model_index(model_name);

						d_m = create_instance(model_index, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(model_name)->get_triangle_count(), false, 1.0f);
						Runtime::model_instances.push_back(d_m);

						inst_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

						w.set_instance_index(inst_idx);
						break;
					case 3:
						in >> model_name;

						model_index = Runtime::find_host_model_index(model_name);

						d_m = create_instance(model_index, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.0f), Runtime::find_host_model(model_name)->get_triangle_count(), false, 1.0f);
						Runtime::model_instances.push_back(d_m);

						inst_idx = static_cast<uint32_t>(Runtime::model_instances.size() - 1);

						//w.set_instance_index(inst_idx);
						break;
					case 4:
						in >> b_dmg;

						w.set_base_damage(static_cast<float>(b_dmg));

						break;
					case 5:
						in >> d_range;
						
						// Set weapon range

						break;
					case 6:

						break;
					case 7:
						
						break;
					case 8:

						break;
					case 9:

						break;
					case 10:

						break;
					case 11:
						in >> mass_kg;

						//w.set_mass(mass_kg);

						break;
					}
					std::cout << str << std::endl;
				}

			Runtime::WEAPONS.push_back(w);
		}
		
	}
	catch (...) {
		// abort select on failure, don't let exception propogate thru sqlite3 call-stack
		return 1;
	}
	return 0;
}

void Runtime::load_weapons(sqlite3* sql) {
	const char* command = "SELECT * FROM weapon_assets;";
	char* err;

	Runtime::WEAPONS = *new std::vector<BulletWeapon>();

	int r = sqlite3_exec(sql, command, sql_callback_weapon, NULL, &err);
	if (r != SQLITE_OK) {
		std::cerr << "SQL Weapons error: " << err << std::endl;
		return;
	}
}

bool Runtime::host_models_contains(std::string name) {
	for (HostModel mod : HOST_MODELS) {
		if (mod.get_name() == name) {
			return true;
		}
	}
	return false;
}

void Level::load_from(std::string path) {

	error_check(cudaMalloc((void**)&d_materials, sizeof(d_Material) * DEVICE_MATERIALS.size()));
	error_check(cudaMemcpy(d_materials, DEVICE_MATERIALS.data(), sizeof(d_Material) * DEVICE_MATERIALS.size(), cudaMemcpyHostToDevice));

	std::ifstream in;
	in.open(path, std::ios::in);
	if (!in) {
		std::cout << "Cannot find Level: " << path << std::endl;
		return;
	}

	std::cout << "Loading Level: " << path << std::endl;

	this->objects = *new std::vector<Object>();

	std::string line;
	std::getline(in, line);
	std::istringstream parse(line);
	size_t leng = 0;

	parse >> leng;
	std::cout << leng << " static models detected!" << std::endl;
	std::string model;

	thrust::host_vector<d_Model> d_models(DEVICE_MODELS.size());
	uint32_t model_count = 0;

	for (size_t i = 0; i < leng; i++) {
		std::getline(in, line);
		std::istringstream in0(line);

		float x, y, z, x_r, y_r, z_r, scale;

		in0 >> x >> y >> z >> x_r >> y_r >> z_r >> scale >> model;
		//std::cout << model << std::endl;

		glm::vec3 position = glm::vec3(x, y, z);
		glm::vec3 rotation = glm::vec3(x_r, y_r, z_r);

		HostModel* h_model = Runtime::find_host_model(model);

		Runtime::model_instances.push_back(create_instance(Runtime::find_host_model_index(model), position, rotation, Runtime::find_host_model(model)->get_triangle_count(), false, scale));

		std::cout << "d_model = " << Runtime::model_instances.back().model_index << std::endl;

		//d_Model d_model2 = Runtime::find_host_model(model)->to_gpu();

		//d_models.push_back(DEVICE_MODELS.at(Runtime::find_host_model_index(model)));	// -> Dis piece o shite

	}
	thrust::copy(DEVICE_MODELS.begin(), DEVICE_MODELS.end(), d_models.begin());
	size_t light_leng;
	std::string line2;
	std::getline(in, line2);
	std::istringstream parse2(line2);
	parse2 >> light_leng;
	std::cout << light_leng << " lights detected!" << std::endl;
	std::vector<d_PointLight> point_lights;
	for (size_t i = 0; i < light_leng; i++) {
		std::getline(in, line);
		std::istringstream in1(line);

		float x, y, z, r, g, b, a, s_r, s_g, s_b, s_a, intensity, falloff, range;

		in1 >> x >> y >> z >> r >> g >> b >> a >> s_r >> s_g >> s_b >> s_a >> intensity >> falloff >> range;
		//std::cout << model << std::endl;

		glm::vec3 position = glm::vec3(x, y, z);
		glm::vec4 color = glm::vec4(r, g, b, a);
		glm::vec4 s_color = glm::vec4(s_r, s_g, s_b, s_a);

		point_lights.push_back(d_PointLight{ position, color, s_color, intensity, falloff, range });
	}

	std::getline(in, line);
	std::istringstream in2(line);

	float r, g, b, a, s_r, s_g, s_b, s_a, intensity;

	in2 >> r >> g >> b >> a >> s_r >> s_g >> s_b >> s_a >> intensity;

	d_AmbientLight amb_light = { glm::vec4(r, g, b, a), glm::vec4(s_r, s_g, s_b, s_a), intensity };

	this->d_model_instance_count = static_cast<uint32_t>(Runtime::model_instances.size());

	this->d_point_lights_count = static_cast<uint32_t>(point_lights.size());

	std::getline(in, line);
	std::istringstream in3(line);
	size_t object_count;
	in3 >> object_count;
	std::cout << std::setw(10) << object_count << " objects detected!" << std::endl;
	bool has_player = false;
	this->objects = *new std::vector<Object>();
	for (size_t i = 0; i < object_count; i++) {
		std::getline(in, line);
		std::istringstream in_obj(line);
		uint32_t type = 0;
		float x, y, z, x_d, y_d, z_d, scale;
		std::string visual_model, rigid_model;

		in_obj >> x >> y >> z >> x_d >> y_d >> z_d >> scale >> visual_model;

		ObjectType obj_type = ObjectType::AI;
		//std::cout << "Type " << type << std::endl;

		d_ModelInstance instance;
		uint32_t model_idx = 0, instance_idx = 0, hitbox_index = 0;

		this->objects.push_back(Runtime::OBJECTS[Runtime::find_object_index(visual_model)]);
		this->objects.back().set_position(glm::vec3(x, y, z));
		this->objects.back().set_direction(glm::vec3(x_d, y_d, z_d));

		this->add_model_instance(create_instance(this->objects.back().get_model_index(), this->objects.back().get_position(), this->objects.back().get_direction(), HOST_MODELS[this->objects.back().get_model_index()].get_triangle_count(), false, scale));
		Runtime::model_instances[this->objects.back().get_hitbox_instance_index()].scale = scale;
		this->objects.back().set_instance_index(static_cast<uint32_t>(Runtime::model_instances.size() - 1));
		this->objects.back().set_spawn_point(glm::vec3(x, y, z));

		if (this->objects.back().get_object_type() == ObjectType::Player) {
			this->objects.back().attach_camera(this->camera_ptr);
		}
		std::cout << "Object added: " << &this->objects.back() << " @ " << this->objects.back().get_model_index() << " index with model " << visual_model << " of type " << this->objects.back().get_object_type() << std::endl;
	}

	error_check(cudaMalloc((void**)&this->d_DEVICE_MODELS, sizeof(d_Model) * d_models.size()));
	error_check(cudaMemcpy(this->d_DEVICE_MODELS, thrust::raw_pointer_cast(d_models.data()), sizeof(d_Model) * d_models.size(), cudaMemcpyHostToDevice));
	this->d_DEVICE_MODEL_COUNT = static_cast<uint32_t>(d_models.size());

	error_check(cudaMalloc((void**)&this->d_ambient_light, sizeof(d_AmbientLight)));
	error_check(cudaMemcpy(this->d_ambient_light, &amb_light, sizeof(d_AmbientLight), cudaMemcpyHostToDevice));

	error_check(cudaMalloc((void**)&this->d_point_lights, sizeof(d_PointLight) * point_lights.size()));
	error_check(cudaMemcpy(this->d_point_lights, point_lights.data(), sizeof(d_PointLight) * point_lights.size(), cudaMemcpyHostToDevice));
	cudaDeviceSynchronize();
	//this->upload_objects();

	//error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * this->model_instances.size()));
	//this->upload_instances();
}

static void keyboard_callback(GLFWwindow* win, int key, int scancode, int action, int mods) {

	KeyboardButtonUse k{};

	k.key = key;
	k.scancode = scancode;
	k.action = action;
	k.mods = mods;

	keyboard_button_uses.push(k);
}

static void mouse_callback(GLFWwindow* window, int button, int action, int mods) {

	MouseButtonUse k{};

	k.button = button;
	k.action = action;
	k.mods = mods;

	mouse_button_uses.push(k);
}

Application::Application(int32_t dimx, int32_t dimy) {
	this->dims = glm::ivec2(dimx, dimy);

	int err = sqlite3_open("databases\\master.db", &this->database_connection);
	if (err != SQLITE_OK) {
		std::cout << "Cannot open asset database! : " << sqlite3_errmsg(this->database_connection) << std::endl;
		return;
	}
	Runtime::model_instances = *new std::vector<d_ModelInstance>();
	Runtime::load_materials(this->database_connection);
	Runtime::runtime_load(this->database_connection);
	Runtime::load_objects(this->database_connection);
	Runtime::load_weapons(this->database_connection);

	glfwInit();
	glfwSwapInterval(0);
	glfwWindowHint(GLFW_CENTER_CURSOR, GLFW_TRUE);
	glfwWindowHint(GLFW_VISIBLE, GLFW_TRUE);
	glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

	this->win = glfwCreateWindow(this->dims.x, this->dims.y, "ZDproto-demo v0.00", NULL, NULL);
	this->loop = true;

	glm::vec3 init_position = glm::vec3(0.0f, 1.0f, 0.0f);
	glm::vec3 init_direction = glm::vec3(45.0f, 45.0f, 0.0f);

	this->camera = new Camera(this->dims, 100.0f, init_position, init_direction);

	this->level = new Level("resources/levels/test_level.txt", this->camera);

	//this->ai.push_back(FiniteStateMachine(glm::vec3(10.0f, 0.0f, 0.0f), this->level, ));


	glfwMakeContextCurrent(this->win);
}

void Application::input_handle(KeyboardButtonUse& k) {

	int CURRENT_KEY = k.key;
	int CURRENT_ACTION = k.action;

	std::cout << CURRENT_KEY << std::endl;

	this->is_walking = false, this->is_sprinting = false, this->trying_sprint = false;;
	switch (CURRENT_KEY) {
	case GLFW_KEY_W:
		if (CURRENT_ACTION == GLFW_RELEASE) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::Idle);
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::WalkF);
		}
		break;
	case GLFW_KEY_S:
		if (CURRENT_ACTION == GLFW_RELEASE) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::Idle);
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::WalkB);
		}
		break;
	case GLFW_KEY_A:
		if (CURRENT_ACTION == GLFW_RELEASE) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::Idle);
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::WalkL);
		}
		break;
	case GLFW_KEY_D:
		if (CURRENT_ACTION == GLFW_RELEASE) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::Idle);
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::WalkR);
		}
		break;
	case GLFW_KEY_R:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			//this->cam->set_capture_mode(RT);
		}
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		break;
	case GLFW_KEY_F:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			//this->camera->set_capture_mode(FULLBRIGHT);
		}
		break;
	case GLFW_KEY_ESCAPE:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			this->loop = false;
		}
		break;
	case GLFW_KEY_P:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			//this->cam->get_rays(0)->debug_stats();
		}
		break;
	case GLFW_KEY_SPACE:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			Runtime::PLAYER_OBJECT->set_velocity(Runtime::PLAYER_OBJECT->get_velocity() - glm::vec3(0.0f, 2.0f * 9.81f, 0.0f));
		}
		break;
	case GLFW_MOUSE_BUTTON_LEFT:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_REPEAT) {

		}
		break;
	}

	int CURRENT_MOD = k.mods;
	switch (CURRENT_MOD) {
	case GLFW_MOD_SHIFT:
		if (Runtime::PLAYER_OBJECT->get_player_state() == PlayerState::WalkF) {
			Runtime::PLAYER_OBJECT->set_player_state(PlayerState::RunF);
		}
		break;
	}


}

void Application::mouse_handle(MouseButtonUse& k) {

	int CURRENT_MOUSE = k.button;
	int CURRENT_ACTION = k.action;

	switch (CURRENT_MOUSE) {
	case GLFW_MOUSE_BUTTON_RIGHT:
		if (CURRENT_ACTION == GLFW_RELEASE) {
			this->camera->set_current_fov(this->camera->get_fov());
		}
		if (CURRENT_ACTION == GLFW_PRESS) {
			BulletWeapon* current_weapon = Runtime::PLAYER_OBJECT->get_current_weapon();

			this->camera->set_current_fov(glm::vec2(65.0f, 95.0f));
		}
		break;
	case GLFW_MOUSE_BUTTON_LEFT:
		if (CURRENT_ACTION == GLFW_RELEASE) {
		}
		if (CURRENT_ACTION == GLFW_PRESS || CURRENT_ACTION == GLFW_REPEAT) {
			glm::vec3 fire_direction = this->camera->get_direction();

			BulletWeapon* current_weapon = Runtime::PLAYER_OBJECT->get_current_weapon();
			bool fire = false;
			switch (current_weapon->get_weapon_type()) {
				case WeaponType::SemiAutomatic:
					if (glfwGetTime() - current_weapon->get_last_fire() >= current_weapon->get_fire_delay()) {
						fire = true;
					}
					break;
				case WeaponType::FullyAutomatic:
					if (glfwGetTime() - current_weapon->get_last_fire() >= current_weapon->get_fire_delay()) {
						fire = true;
					}
					break;
			}

			if (fire) {

				std::cout << "Shooting!" << std::endl;

				bool* intersection_tests = new bool[this->level->get_object_count()], * d_intersection_tests;
				float* distances = new float[this->level->get_object_count()], * d_distances;
				error_check(cudaMalloc((void**)&d_intersection_tests, this->level->get_object_count() * sizeof(bool)), "Application::mouse_handle cudaMalloc1");
				error_check(cudaMalloc((void**)&d_distances, this->level->get_object_count() * sizeof(float)), "Application::mouse_handle cudaMalloc2");
				//std::cout << this->level->get_d_object_count() << " Objects going in @ " << this->level->get_d_objects() << std::endl;
				test_intersection << <(this->level->get_d_object_count() / 128) + 1, 128 >> > (this->camera->get_position(), fire_direction, this->level->get_d_objects(), this->level->get_d_object_count(), this->level->get_d_model_instances(), this->level->get_d_model_instance_count(), this->level->get_d_device_models(), d_intersection_tests, d_distances);
				cudaDeviceSynchronize();
				error_check(cudaGetLastError(), "Application::mouse_handle kernel call");

				error_check(cudaMemcpy(intersection_tests, d_intersection_tests, this->level->get_object_count() * sizeof(bool), cudaMemcpyDeviceToHost), "Application::mouse_handle cudaMemcpy1");
				error_check(cudaMemcpy(distances, d_distances, this->level->get_object_count() * sizeof(float), cudaMemcpyDeviceToHost), "Application::mouse_handle cudaMemcpy2");
				float closest = 1000.0f;
				int32_t index = -1;
				for (size_t i = 0; i < this->level->get_object_count(); i++) {
					if (intersection_tests[i]) {
						if (distances[i] < closest) {
							index = static_cast<int32_t>(i);
						}
					}
				}
				if (index != -1) {
					std::cout << "Damaging object " << index << std::endl;
					Object objs = this->level->get_objects_ptr()[index];
					objs.set_health(objs.get_health() - current_weapon->get_base_damage());
					this->level->update_object(static_cast<uint32_t>(index), objs);
				}

				error_check(cudaFree(d_intersection_tests), "free d_intersection_tests");
				error_check(cudaFree(d_distances), "free d_distances");

				current_weapon->set_last_fire(glfwGetTime());
			}
		}
		break;

	}
}

void Application::main_loop() {
	this->camera->set_last_time(glfwGetTime());
	glfwSetKeyCallback(this->win, keyboard_callback);
	glfwSetMouseButtonCallback(this->win, mouse_callback);
	glfwMakeContextCurrent(this->win);

	//this->camera->set_position(Runtime::PLAYER_OBJECT->get_position());

	Runtime::PLAYER_OBJECT->set_primary_weapon(&Runtime::WEAPONS[Runtime::find_weapon_index("Default Weapon")]);
	Runtime::PLAYER_OBJECT->set_player_state(PlayerState::Idle);

	glfwSetInputMode(this->win, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);

	int frame_count = 0;
	while (this->loop && !glfwWindowShouldClose(this->win)) {

		glfwPollEvents();
		this->frame_buffer = new glm::vec4[this->dims.x * this->dims.y];
		glm::vec4* d_frame_buffer,* d_frame_buffer_post;

		error_check(cudaMalloc((void**)&d_frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x), "malloc frame buffer on device");
		error_check(cudaMalloc((void**)&d_frame_buffer_post, sizeof(glm::vec4) * this->dims.y * this->dims.x), "malloc post frame buffer on device");

		this->camera->new_frame();

		this->zero_frame_buffer_sse();

		error_check(cudaMemcpy(d_frame_buffer, this->frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x, cudaMemcpyHostToDevice), "copy frame buffer to device");

		Object* obj = this->level->get_objects_ptr();
		//std::cout << "Updating " << this->level->get_object_count() << " objects in world!" << std::endl;
		
		this->empty_input_queues();
		camera->update_to(Runtime::PLAYER_OBJECT->get_player_state());

		for (size_t i = 0; i < this->level->get_object_count(); i++) {
			obj[i].update(&Runtime::model_instances[obj[i].get_instance_index()], &Runtime::model_instances[obj[i].get_hitbox_instance_index()], this->camera, glfwGetTime() - this->camera->get_last_time(), this->win, Runtime::PLAYER_OBJECT);
			/*if (obj[i].get_object_type() == ObjectType::Player && obj[i].get_health() <= 0.0f) {
				std::cout << "Game Over" << std::endl;
				this->loop = false;
			}*/
		}

		if (frame_count != 0) {
			level->clean_d_objects();
		}

		this->level->upload_objects();
		this->level->upload_instances();

		// Render functions
		this->camera->capture(this->level->get_d_model_instances(), this->level->get_d_model_instance_count(), this->level->get_d_device_models(), this->level->get_d_materials(), this->level->get_d_ambient_light(), this->level->get_d_point_lights(), this->level->get_d_point_lights_size(), d_frame_buffer, d_frame_buffer_post);

		//this->camera->copy_to_frame_buffer(this->frame_buffer, 0);

		error_check(cudaMemcpy(this->frame_buffer, d_frame_buffer, sizeof(glm::vec4) * this->dims.y * this->dims.x, cudaMemcpyDeviceToHost), "copy frame buffer from device");

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glDrawPixels(this->dims.x, this->dims.y, GL_BGRA_EXT, GL_FLOAT, this->frame_buffer);
		glfwSwapBuffers(this->win);

		this->camera->cleanup_frame();

		delete this->frame_buffer;
		cudaFree(d_frame_buffer);
		cudaFree(d_frame_buffer_post);

		this->camera->debug_print();
		//this->cam->last_time = glfwGetTime();
		this->camera->set_last_time(glfwGetTime());
		frame_count++;
	}
	glfwDestroyWindow(this->win);
	glfwTerminate();
}

void Level::add_model_instance(d_ModelInstance inst) {
	Runtime::model_instances.push_back(inst);
}

void Level::update_instance(uint32_t index, d_ModelInstance model) {
	Runtime::model_instances[index] = model;
}

void Level::update_object(uint32_t index, Object object) {
	this->objects[index] = object;

	d_ModelInstance instance = Runtime::model_instances[this->objects[index].get_instance_index()];
	instance.position = this->objects[index].get_position();
	instance.rotation = this->objects[index].get_direction();
	this->update_instance(this->objects[index].get_instance_index(), instance);
}

void Level::upload_instances() {
	error_check(cudaMalloc((void**)&this->d_model_instances, sizeof(d_ModelInstance) * Runtime::model_instances.size()), "cudaMalloc Level instances");
	error_check(cudaMemcpy(this->d_model_instances, Runtime::model_instances.data(), sizeof(d_ModelInstance) * Runtime::model_instances.size(), cudaMemcpyHostToDevice), "cudaMemcpy Level instances to device");
	cudaDeviceSynchronize();
	this->d_model_instance_count = static_cast<uint32_t>(Runtime::model_instances.size());
}

Object::Object(ObjectType type, ObjIndexs idxs, std::string name, glm::vec3 position, glm::vec3 direction, uint32_t model, uint32_t instance, uint32_t hitbox_instance) {
	this->name = name;
	this->position = position;
	this->spawn_point = position;
	this->direction = direction;
	this->object_type = type;
	this->rotation = direction;

	this->model_index = model;
	this->instance_index = instance;
	this->hitbox_instance_index = hitbox_instance;

	this->obj_indices = idxs;

	this->creation_time = glfwGetTime();

	this->primary = &Runtime::WEAPONS[Runtime::find_weapon_index("Default Weapon")];
	this->velocity = glm::vec3(0.0f);

	this->last_time = 0.0f;
}

void Object::update(d_ModelInstance* instances, d_ModelInstance* hitbox_instance, Camera* cam, float t, GLFWwindow* win, Object* player) {
	//std::cout << "updating object" << std::endl;
	float to_player, time;
	if ((this->object_type == ObjectType::Physics || this->object_type == ObjectType::AI) && (this->object_type != ObjectType::Static && this->object_type != ObjectType::Weapon)) {
		float delta_time = this->last_time - time;

		//this->velocity = glm::vec3(0.0f);

		bool* collides = new bool[Runtime::OBJECTS.size()];
		glm::vec3* collide_to = new glm::vec3[Runtime::OBJECTS.size()];
		for (size_t i = 0; i < Runtime::OBJECTS.size(); i++) {
			if (&Runtime::OBJECTS[i] != this) {
				if (Runtime::objects_collide(this, &Runtime::OBJECTS[i])) {
					collides[i] = true;
					collide_to[i] = glm::normalize(Runtime::OBJECTS[i].get_position() - this->position);
				}
				else {
					collides[i] = false;
				}
			}
			else {
				collides[i] = false;
			}
		}

		//this->velocity += glm::vec3(0.0f, 9.81f * delta_time * delta_time, 0.0f);

		bool collide = false;

		this->position += this->velocity * delta_time;	// Commented out
		instances[this->instance_index].position = this->position;
		//instances[this->instance_index].rotation = this->direction;
		instances[this->hitbox_instance_index].position = this->position;
	}

	if (this->object_type == ObjectType::AI) {
		this->velocity = glm::vec3(0.0f);
		//std::cout << "	updating as AI" << std::endl;
		this->target_position = Runtime::PLAYER_OBJECT->get_position();
		this->direction = glm::normalize(this->target_position - this->position);
		this->position += this->direction * t * 5.0f;
		instances->position = this->position;
		instances->rotation = this->direction;

		glm::vec3 dist = this->target_position - this->position;
		to_player = glm::length(dist);
		printf("Distance to player = %.2f, %.2f, %.2f\n", dist.x, dist.y, dist.z);

		hitbox_instance->position = this->position;
		hitbox_instance->rotation = this->direction;

		time = glfwGetTime();
		if (to_player <= this->attack_range) {
			std::cout << std::setw(10) << "AI Within Attack Range" << std::endl;
			if (time - this->last_attack >= this->attack_cooldown) {
				player->set_health(player->get_health() - this->current_damage);
				this->last_attack = time;
				std::cout << "AI Attacking Player" << std::endl;
			}
		}
		if (this->current_health <= 0.0f) {
			this->position = this->spawn_point;
			this->current_health = max_health;
		}
	}

	else if (this->object_type == ObjectType::Player) {
		//std::cout << "	updating as Player" << std::endl;
		double x, y;
		glfwGetCursorPos(win, &x, &y);
		cam->add_to_euler_direction(glm::vec2(static_cast<float>(x), static_cast<float>(y)));
		glfwSetCursorPos(win, cam->get_dims().x * 0.5f, cam->get_dims().y * 0.5f);

		glm::vec3 p = cam->get_position();

		this->direction = cam->get_direction();
		this->position = p; ////* - glm::vec3(0.0f, (9.81f * (this->last_time - time)), 0.0f)*/ +this->velocity;
		cam->set_position(this->position);
		Runtime::PLAYER_OBJECT = this;

		hitbox_instance->position = this->position;
		hitbox_instance->rotation = this->direction;

		Runtime::WEAPONS[Runtime::find_weapon_index("Default Weapon")].set_offset(glm::vec3(0.025f, 0.0f, 0.025f));
		Runtime::PLAYER_OBJECT->set_primary_weapon(&Runtime::WEAPONS[Runtime::find_weapon_index("Default Weapon")]);

		d_ModelInstance* d_wpn = &Runtime::model_instances[Runtime::PLAYER_OBJECT->get_current_weapon()->get_instance_index()];
		d_wpn->scale = 0.01f;
		d_wpn->position = this->position + this->primary->get_offset();
		d_wpn->rotation = cam->get_direction();
		instances[this->primary->get_instance_index()].position = this->position + this->primary->get_offset();
		Runtime::PLAYER_OBJECT = this;
	}

	else if (this->object_type == ObjectType::Weapon) {
		//std::cout << "	updating as Weapon" << std::endl;
		uint32_t inst_idx = Runtime::WEAPONS[this->obj_indices.weapon_index].get_instance_index();
		BulletWeapon* wpn = &Runtime::WEAPONS[this->obj_indices.weapon_index];
		d_ModelInstance* d_mi = &Runtime::model_instances[inst_idx];

		//d_mi->position = Runtime::PLAYER_OBJECT->get_position() + Runtime::WEAPONS[this->obj_indices.weapon_index].get_offset();
		//d_mi->rotation = Runtime::PLAYER_OBJECT->get_direction();
	}

	else if (this->object_type == ObjectType::Physics) {
		float delta_time = this->creation_time - time;
	}

	else if (this->object_type == ObjectType::Static) {
		this->velocity = glm::vec3(0.0f);
		instances[this->instance_index].position = this->position;
		//instances[this->instance_index].rotation = this->direction;
		instances[this->hitbox_instance_index].position = this->position;
	}
	this->last_time = glfwGetTime();
}

HostModel* Runtime::find_host_model(std::string name) {
	for (uint32_t i = 0; i < HOST_MODELS.size(); i++) {
		if (HOST_MODELS.at(i).get_name() == name) {
			return &HOST_MODELS.at(i);
		}
	}
	return &HOST_MODELS.at(0);	// Def fix this later, don't know what to do with it now
}

uint32_t Runtime::find_host_model_index(std::string name) {
	for (uint32_t i = 0; i < HOST_MODELS.size(); i++) {
		if (HOST_MODELS.at(i).get_name() == name) {
			return i;
		}
	}
	return 0;
}

uint32_t Runtime::find_material_index(std::string mat) {
	uint32_t count = 0;
	for (Material m : HOST_MATERIALS) {
		if (mat == m.get_name()) {
			return count;
		}
		count++;
	}
	return 0;	// FIX THIS EVENTUALLY!
}
