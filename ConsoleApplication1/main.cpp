#include <string>
#include <iostream>
#include <fstream>
#include <thread>
#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include "common/model.hpp"
#include "common/pipeline.hpp"
#include "common/utils/parameter.hpp"

#if defined(_WIN32) && defined(DEBUG_MEMORY)
// To ensure correct resolution of symbols, add Psapi.lib to TARGETLIBS
// and compile with -DPSAPI_VERSION=1
static void DebugMemoryInfo(const char* header)
{
	std::this_thread::sleep_for(std::chrono::seconds(5));
	PROCESS_MEMORY_COUNTERS_EX2 pmc;
	if (GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))
	{
		//The following printout corresponds to the value of Resource Memory, respectively
		printf("%s:\tCommit \t\t\t=  0x%08X- %u (KB)\n", header, pmc.PrivateUsage, pmc.PrivateUsage / 1024);
		printf("%s:\tWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n", header, pmc.WorkingSetSize, pmc.WorkingSetSize / 1024);
		printf("%s:\tPrivateWorkingSetSize\t\t\t=  0x%08X- %u (KB)\n", header, pmc.PrivateWorkingSetSize, pmc.PrivateWorkingSetSize / 1024);
	}
}
#endif

void run_test(ParameterConfig parameter_config) {
	std::string model_path = parameter_config.model_path;
	std::string device = parameter_config.device;
	int num_inference_steps = parameter_config.infer_steps;
	int user_seed = parameter_config.seed;              //
	int waiting_time = parameter_config.waiting_time;   //
	int num_images = parameter_config.num_images;       //
	int loop_num = parameter_config.loop_num;           //
	bool encrypt_tag = parameter_config.encrypt_tag;    //
	std::vector<std::string> pos_prompts_list = parameter_config.pos_prompts;
	std::vector<std::string> neg_prompts_list = parameter_config.neg_prompts;

	std::cout << "Model Path: " << parameter_config.model_path << std::endl;
	std::cout << "Target Device: " << parameter_config.device << std::endl;

	std::shared_ptr<ov::Core> core_ptr = std::make_shared<ov::Core>();
	//core_ptr->add_extension("openvino_tokenizersd.dll");
	core_ptr->add_extension("openvino_tokenizers.dll");
	if (device.find("CPU") != std::string::npos) {
		core_ptr->set_property("CPU", { {"CPU_RUNTIME_CACHE_CAPACITY","0"} });
		std::cout << "Paraformer:: Set CPU_RUNTIME_CACHE_CAPACITY = 0 \n" << std::endl;
	}
	core_ptr->set_property(ov::cache_dir("./ov_cache_dir"));
	ov::AnyMap ov_config = {};
	// This two property can not use on openvino tokenizer model ???
	if (device.find("GPU") != std::string::npos) {
		ov_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
		ov_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
	}

	OVStableDiffusionModels ov_sd_models;
	ov_sd_models.load_model_cache(core_ptr, model_path, device, ov_config, encrypt_tag);
	ov_sd_models.create_infer_req();
	std::string positive_prompt = "tiger";
	std::string negative_prompt = "";
	sd_generation(ov_sd_models, positive_prompt, negative_prompt, user_seed, num_images, num_inference_steps);
	ov_sd_models.unload(device);
	core_ptr->unload_plugin(device);
	core_ptr.reset();

	return;
}

void run_test_multi(ParameterConfig parameter_config) {
	std::string model_path = parameter_config.model_path;
	std::string device = parameter_config.device;
	int num_inference_steps = parameter_config.infer_steps;
	int user_seed = parameter_config.seed;              //
	int waiting_time = parameter_config.waiting_time;   //
	int num_images = parameter_config.num_images;       //
	int loop_num = parameter_config.loop_num;           //
	bool encrypt_tag = parameter_config.encrypt_tag;    //
	std::vector<std::string> pos_prompts_list = parameter_config.pos_prompts;
	std::vector<std::string> neg_prompts_list = parameter_config.neg_prompts;

	std::cout << "Model Path: " << parameter_config.model_path << std::endl;
	std::cout << "Target Device: " << parameter_config.device << std::endl;

	std::shared_ptr<ov::Core> core_ptr = std::make_shared<ov::Core>();
	//core_ptr->add_extension("openvino_tokenizersd.dll");
	core_ptr->add_extension("openvino_tokenizers.dll");
	if (device.find("CPU") != std::string::npos) {
		core_ptr->set_property("CPU", { {"CPU_RUNTIME_CACHE_CAPACITY","0"} });
		std::cout << "Paraformer:: Set CPU_RUNTIME_CACHE_CAPACITY = 0 \n" << std::endl;
	}
	core_ptr->set_property(ov::cache_dir("./ov_cache_dir"));
	ov::AnyMap ov_config = {};
	// This two property can not use on openvino tokenizer model ???
	if (device.find("GPU") != std::string::npos) {
		ov_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
		ov_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;
	}

	OVStableDiffusionModels ov_sd_models;
	//ov_sd_models.load_model_cache(core_ptr, model_path, device, ov_config, encrypt_tag);
	ov_sd_models.load_model_cache(core_ptr, model_path, device, ov_config, encrypt_tag);
	ov_sd_models.create_infer_req();
	uint32_t x = 0;
	std::string negative_prompt = "";

	while (x < loop_num) {
		try {
			std::cout << "Starting iteration " << x + 1 << " of 2..." << std::endl;
			//sd_generation(ov_sd_models, positive_prompt, negative_prompt);
			for (const std::string& positive_prompt : pos_prompts_list) {
				sd_generation(ov_sd_models, positive_prompt, negative_prompt, user_seed, num_images, num_inference_steps);
			}
			std::cout << "Successfully completed iteration " << x + 1 << std::endl;
			std::cout << "Waiting 5 seconds before next iteration..." << std::endl;
			std::this_thread::sleep_for(std::chrono::seconds(waiting_time));
#ifdef DEBUG_MEMORY
			DebugMemoryInfo("finally");
#endif
			x += 1;
		}
		catch (const std::exception& e) {
			std::cerr << "Error during iteration " << x + 1 << ": " << e.what() << std::endl;
			x += 1;
		}
		catch (...) {
			std::cerr << "Unknown error during iteration " << x + 1 << std::endl;
			x += 1;
		}
	}
	ov_sd_models.unload(device);
	core_ptr->unload_plugin(device);
	core_ptr.reset();

	return;
}


int main(int argc, char* argv[]) {
		ParameterConfig parameter_config;
		parameter_config.parameters(argc, argv);

		{
			std::thread gen_thread(run_test_multi, parameter_config);
			if (gen_thread.joinable()) {
				gen_thread.join();
			}

		}
#ifdef DEBUG_MEMORY
		DebugMemoryInfo("finally");
#endif

	std::cout << "Completed all iterations, entering monitoring loop..." << std::endl;
	std::cout << "sleep now!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	while (true)  ////////////////tempory test, need to remove soon
	{
		std::cout << "test idle cpu cl issue ..." << std::endl;
		std::this_thread::sleep_for(std::chrono::seconds(100000));
	}
	
	return 0;



	
	
	


}