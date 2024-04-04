	// Camera.cpp
#include "../include/Camera.h"

Camera::Camera(glm::ivec2 dims, float fov) {
	this->dims = dims;

	this->fov = glm::vec2(fov, static_cast<float>(this->dims.x) / this->dims.y * fov);
}

Camera::Camera(glm::ivec2 dims, float fov, glm::vec3 position, glm::vec3 euler_direction) {
	this->dims = dims;

	this->fov = glm::vec2(fov, static_cast<float>(this->dims.x) / this->dims.y * fov);

	this->euler_direction = euler_direction;

	float yaw = this->euler_direction.x * (PI / 180.0f),
		pitch = this->euler_direction.y * (PI / 180.0f);

	this->direction.x = cosf(yaw) * cosf(pitch);
	this->direction.y = sinf(pitch);
	this->direction.z = sinf(yaw) * cosf(pitch);

	this->direction = glm::normalize(this->direction);
}

void Camera::debug_show_render_data() {

}

void Camera::add_to_euler_direction(glm::vec2 rot) {
	float x = rot.x, y = rot.y;
	printf("X,Y input mouse coord = {%.2f, %.2f}\n", rot.x, rot.y);
	float normalized_coord_x = ((rot.x - (static_cast<float>(this->dims.x* 0.5f))) / static_cast<float>(this->dims.x));
	float normalized_coord_y = ((rot.y - (static_cast<float>(this->dims.y *0.5f))) / static_cast<float>(this->dims.y));
	printf("X,Y normalized input mouse coord = {%.2f, %.2f}\n", normalized_coord_x, normalized_coord_y);

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

void Camera::new_frame() {

	this->d_ray_matrix = (Ray**)malloc(sizeof(Ray*) * MAX_BOUNCE_COUNT);

	for (uint32_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		error_check(cudaMalloc((void**)&this->d_ray_matrix[i], sizeof(Ray) * this->dims.y * this->dims.x));
	}
	cudaDeviceSynchronize();
}

void Camera::cleanup_frame() {

	for (uint32_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		error_check(cudaFree(this->d_ray_matrix[i]));
	}
	free(this->d_ray_matrix);
}

void Camera::copy_to_frame_buffer(glm::vec4* frame_buffer, uint32_t bounce) {
	Ray* h_ray_matrix = (Ray*)malloc(sizeof(Ray) * this->dims.y * this->dims.x);

	error_check(cudaMemcpy(h_ray_matrix, this->d_ray_matrix[bounce], sizeof(Ray) * this->dims.y * this->dims.x, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	for (int y = 0; y < this->dims.y; y++) {
		for (int x = 0; x < this->dims.x; x++) {
			int idx = y * this->dims.x + x;
			frame_buffer[idx] = h_ray_matrix[idx].payload.color;
		}
	}

	free(h_ray_matrix);
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
	this->position -= t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

void Camera::left(float t) {
	this->position += t * Runtime::BASE_SPEED * glm::normalize(glm::cross(this->direction, glm::vec3(0, 1, 0)));
}

void Camera::debug_print() {
	std::cout << "Camera:" << std::endl;

	std::cout << std::setw(8) << this->dims.x << "x" << this->dims.y << "p" << std::endl;
	std::cout << std::setw(15) << "Position: {" << this->position.x << ", " << this->position.y << ", " << this->position.z << " }" << std::endl;
	std::cout << std::setw(15) << "Direction: {" << this->direction.x << ", " << this->direction.y << ", " << this->direction.z << " }" << std::endl;
}