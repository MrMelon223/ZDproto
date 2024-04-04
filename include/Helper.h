#ifndef HELPER_H
#define HELPER_H

	// Standard Libraries
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
	// Vectors/Matrices
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>	// Secret Sauce
	// Window Library
#include <GLFW/glfw3.h>
	// Textures
#include <stb_image.h>
	// SSE/AVX Load/Stores >:)
#include <immintrin.h>
	// CUDA "Schtuff"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <vector_types.h>
#include <vector_functions.h>

static inline void error_check(cudaError_t e) {
	if (e != cudaSuccess) {
		std::cout << std::setw(10) << "Cuda Error: " << cudaGetErrorString(e) << std::endl;
	}
}

const __device__ float PI = 3.14159265;

std::string extract_name(std::string);

#endif
