#ifndef CAMERA_H
#define CAMERA_H

#include "Rendering.cuh"
#include "Runtime.h"
#include "Ray.h"

static const uint32_t MAX_BOUNCE_COUNT = 3;

class Camera {
protected:
	glm::ivec2 dims;
	glm::vec2 fov;

	glm::vec3 position;
	glm::vec3 euler_direction, direction;

	Ray** d_ray_matrix;

	float last_time;
public:
	Camera(glm::ivec2, float);
	Camera(glm::ivec2, float, glm::vec3, glm::vec3);

	void debug_show_render_data();

	Ray* get_ray_matrix(uint32_t i) { return this->d_ray_matrix[i]; }

	glm::vec3 get_position() { return this->position; }
	glm::vec3 get_direction() { return this->direction; }

	void set_position(glm::vec3 p) { this->position = p; }

	void set_euler_direction(glm::vec3 d) { this->euler_direction = d; }
	glm::vec3 get_euler_direction() { return this->euler_direction; }
	void add_to_euler_direction(glm::vec2);

	glm::ivec2 get_dims() { return this->dims; }

	void capture(d_ModelInstance*, uint32_t, d_Model*, d_AmbientLight*, d_PointLight*, uint32_t, glm::vec4*);

	void new_frame();
	void cleanup_frame();
	void copy_to_frame_buffer(glm::vec4*, uint32_t);

	void forward(float);
	void backward(float);
	void right(float);
	void left(float);

	float get_last_time() { return this->last_time; }
	void set_last_time(float t) { this->last_time = t; }

	void debug_print();
};

#endif