#include "parameter.hpp"

void ParameterConfig::parameters(int argc, char* argv[]) {
	for (int i = 1; i < argc; ++i) {
		std::string arg = argv[i];
		if (arg == "--model_path" && i + 1 < argc) {
			model_path = argv[++i];
		}
		else if (arg == "--device" && i + 1 < argc) {
			device = argv[++i];
		}
		else if (arg == "--infer_step" && i + 1 < argc) {
			infer_steps = std::atoi(argv[++i]);
		} 
		else if (arg == "--seed" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--waiting_time" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--num_images" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--loop_num" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--encrypt_tag" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--pos_prompts" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
		else if (arg == "--neg_prompts" && i + 1 < argc) {
			seed = std::atoi(argv[++i]);
		}
	}
	return;
}
