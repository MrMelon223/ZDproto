	// Runtime.cpp
#include "../include/Runtime.h"

bool Runtime::host_models_contains(std::string name) {
	for (HostModel mod : Runtime::HOST_MODELS) {
		if (mod.get_name() == name) {
			return true;
		}
	}
	return false;
}

HostModel* Runtime::find_host_model(std::string name) {
	for (uint32_t i = 0; i < Runtime::HOST_MODELS.size(); i++) {
		if (Runtime::HOST_MODELS.at(i).get_name() == name) {
			return &Runtime::HOST_MODELS.at(i);
		}
	}
	return &Runtime::HOST_MODELS.at(0);	// Def fix this later, don't know what to do with it now
}