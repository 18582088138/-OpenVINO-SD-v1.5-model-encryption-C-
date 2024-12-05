#ifndef MODEL_UTILS
#define MODEL_UTILS

#include <iostream>

#include "openvino/runtime/core.hpp"

std::shared_ptr<ov::Model> ov_unet_ppp(std::shared_ptr<ov::Model> unet_model);

std::shared_ptr<ov::Model> ov_vae_decoder_ppp(std::shared_ptr<ov::Model> unet_model);

std::shared_ptr<ov::CompiledModel> readOVModel(
	std::shared_ptr<ov::Core>& core_ptr,
	std::string model_path,
	std::string device = "GPU",
	ov::AnyMap ov_config = {},
	std::vector<uint32_t> gen_shape = { 512,512 });  // {width, height}

std::shared_ptr<ov::CompiledModel> readOVModelCache(
	std::shared_ptr<ov::Core> core_ptr,
	std::string model_cache_path,
	std::string device = "GPU",
	ov::AnyMap ov_config = {},
	bool encrypt_tag = true);

void saveOVModelCache(
	std::shared_ptr<ov::CompiledModel>& ov_compiled_ptr,
	std::string model_cache_path,
	bool encrypt_tag = true);


#endif // !MODEL_UTILS

