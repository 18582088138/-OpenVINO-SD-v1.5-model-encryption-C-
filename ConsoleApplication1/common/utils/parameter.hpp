#ifndef PARAMETER_H
#define PARAMETER_H

#include <string>
#include <vector>

struct ParameterConfig {
	std::string model_path = "C:/Users/kk/Downloads/models/xiaomi_sd1.5_lcm_ov4.0";
	std::string device = "GPU";
	int infer_steps = 4;
	int seed = 42;
	int waiting_time = 1;
	int num_images = 1;
	int loop_num = 2;
	bool encrypt_tag = true;
	std::vector<std::string> pos_prompts = { "lion", "tiger", "grapes" ,"watermelon", "apple"};
	std::vector<std::string> neg_prompts = {};
	void parameters(int argc, char* argv[]);
};

#endif //PARAMETER_H
