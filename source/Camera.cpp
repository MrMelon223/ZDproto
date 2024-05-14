	// Camera.cpp
#include "../include/Camera.h"

Camera::Camera(glm::ivec2 dims, float fov) {
	this->dims = dims;

	this->fov = glm::vec2(fov, static_cast<float>(this->dims.x) / this->dims.y * fov);
}

Camera::Camera(glm::ivec2 dims, float fov, glm::vec3 position, glm::vec3 euler_direction) {
	this->dims = dims;

	this->fov = this->current_fov = glm::vec2(fov, static_cast<float>(this->dims.x) / this->dims.y * fov);

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

void Camera::new_frame() {

	this->d_ray_matrix = (Ray**)malloc(sizeof(Ray*) * MAX_BOUNCE_COUNT);

	for (uint32_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		error_check(cudaMalloc((void**)&this->d_ray_matrix[i], sizeof(Ray) * this->dims.y * this->dims.x), "new frame");
	}
	cudaDeviceSynchronize();
}

void Camera::cleanup_frame() {

	for (uint32_t i = 0; i < MAX_BOUNCE_COUNT; i++) {
		error_check(cudaFree(this->d_ray_matrix[i]), "cleanup frame");
	}
	free(this->d_ray_matrix);
}

void Camera::copy_to_frame_buffer(glm::vec4* frame_buffer, uint32_t bounce) {
	Ray* h_ray_matrix = (Ray*)malloc(sizeof(Ray) * this->dims.y * this->dims.x);

	error_check(cudaMemcpy(h_ray_matrix, this->d_ray_matrix[bounce], sizeof(Ray) * this->dims.y * this->dims.x, cudaMemcpyDeviceToHost), "copy Ray payloads to frame buffer");
	cudaDeviceSynchronize();

	for (int y = 0; y < this->dims.y; y++) {
		for (int x = 0; x < this->dims.x; x++) {
			int idx = y * this->dims.x + x;
			frame_buffer[idx] = h_ray_matrix[idx].payload.color;
		}
	}

	free(h_ray_matrix);
}

void Camera::debug_print() {
	std::cout << "Camera:" << std::endl;

	std::cout << std::setw(8) << this->dims.x << "x" << this->dims.y << "p" << std::endl;
	std::cout << std::setw(15) << "Position: {" << this->position.x << ", " << this->position.y << ", " << this->position.z << " }" << std::endl;
	std::cout << std::setw(15) << "Direction: {" << this->direction.x << ", " << this->direction.y << ", " << this->direction.z << " }" << std::endl;
}

void Camera::update_to(PlayerState p) {
	float delta_t = glfwGetTime() - this->last_time;
	switch (p) {
	case PlayerState::WalkF:
		this->forward(delta_t);
		break;
	case PlayerState::WalkR:
		this->right(delta_t);
		break;
	case PlayerState::WalkL:
		this->left(delta_t);
		break;
	case PlayerState::WalkB:
		this->backward(delta_t);
		break;
	case PlayerState::RunF:
		this->forward(2 * delta_t);
		break;
	}
}