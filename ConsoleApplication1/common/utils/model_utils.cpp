#define OPENSSL_SUPPRESS_DEPRECATED  //workaround
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cctype>
#include <random>


#include "model_utils.hpp"
#include "encrypt_utils.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

std::shared_ptr<ov::Model> ov_unet_ppp(std::shared_ptr<ov::Model> unet_model,
									   std::vector<uint32_t> gen_shape = { 512,512 }) {
	ov::preprocess::PrePostProcessor unet_ppp(unet_model);

	uint32_t unet_w = gen_shape[0] / 8;
	uint32_t unet_h = gen_shape[1] / 8;

	unet_ppp.input("sample").tensor()  // Replace with the actual input name
		.set_element_type(ov::element::f32) // Set input tensor element type, NOT ov::element::i8
		.set_shape({ 2, 4, unet_h, unet_w });       // Set input tensor shape, not 1, 4, 64, 64 ;  not 2, 4, 512, 512  (512 is  cmd ConsoleApplication1.exe  -d GPU  -p "A boy is swimming"    -m  "C:\\sd_ov_int8"  --width 512 --height 512 )
	// Configure output tensor preprocessing
	unet_ppp.output(0).tensor()  // Assuming you have one output or want to configure the first one
		.set_element_type(ov::element::f32);  // Set output tensor element type, NOT ov::element::i8
	unet_ppp.output().tensor().set_layout("NCHW");
	std::shared_ptr<ov::Model> ppp_unet_model = unet_ppp.build();
	return ppp_unet_model;
}

std::shared_ptr<ov::Model> ov_vae_decoder_ppp(std::shared_ptr<ov::Model> vae_decoder_model) {
	ov::preprocess::PrePostProcessor vae_decoder_ppp(vae_decoder_model);
	vae_decoder_ppp.output().model().set_layout("NCHW");
	vae_decoder_ppp.output().tensor().set_layout("NHWC");
	std::shared_ptr<ov::Model> ppp_vae_decoder_model = vae_decoder_ppp.build();

	return ppp_vae_decoder_model;
}

std::shared_ptr<ov::CompiledModel> readOVModel(
		std::shared_ptr<ov::Core>& core_ptr,
		std::string model_path, 
		std::string device,
		ov::AnyMap ov_config,
		std::vector<uint32_t> gen_shape) {  // {width, height}
	std::string model_name;
	std::shared_ptr<ov::Model> ov_model;
	std::shared_ptr<ov::Model> ppp_ov_model;
	std::shared_ptr<ov::CompiledModel> ov_compile_model_ptr;

	if (model_path.find("text_encoder") != std::string::npos) {
		model_name = "text_encoder";
		ov_model = core_ptr->read_model(model_path + "/openvino_model.xml");
		ov_compile_model_ptr = std::make_shared<ov::CompiledModel>(core_ptr->compile_model(ov_model, device, ov_config));
	}
	else if (model_path.find("unet") != std::string::npos) {
		model_name = "unet";
		ov_model = core_ptr->read_model(model_path + "/openvino_model.xml");
		ppp_ov_model = ov_unet_ppp(ov_model, gen_shape);
		ov_compile_model_ptr =
			std::make_shared<ov::CompiledModel>(core_ptr->compile_model(ppp_ov_model, device, ov_config));
	}
	else if (model_path.find("vae_decoder") != std::string::npos) {
		model_name = "vae_decoder";
		ov_model = core_ptr->read_model(model_path + "/openvino_model.xml");
		ppp_ov_model = ov_vae_decoder_ppp(ov_model);
		ov_compile_model_ptr =
			std::make_shared<ov::CompiledModel>(core_ptr->compile_model(ppp_ov_model, device, ov_config));
	}
	else if (model_path.find("tokenizer") != std::string::npos) {
		model_name = "tokenizer";
		ov_model = core_ptr->read_model(model_path + "/openvino_tokenizer.xml");
		ov_compile_model_ptr = std::make_shared<ov::CompiledModel>(core_ptr->compile_model(ov_model, "CPU", ov_config));
	}
	else {
		std::cerr << "Error: Model path error, pls check the model exist. \n"<< model_path <<"\n" << std::endl;
		exit(EXIT_FAILURE);
	}
	std::cout <<"====== read_ov_model " << model_name << " success ===== " << std::endl;
	return ov_compile_model_ptr;
}

std::shared_ptr<ov::CompiledModel> readOVModelCache(
	std::shared_ptr<ov::Core> core_ptr,
	std::string model_cache_path,
	std::string device,
	ov::AnyMap ov_config,
	bool encrypt_tag) {
	std::shared_ptr<ov::CompiledModel> ov_compiled_cache_ptr;
	if (encrypt_tag) {
		std::cout << "== " << model_cache_path << " model decryption ==" << std::endl;
		std::istringstream ov_cache_str = decryptCompiledModel(model_cache_path);
		ov_compiled_cache_ptr = std::make_shared<ov::CompiledModel>(core_ptr->import_model(ov_cache_str, device, ov_config));
		auto new_req = ov_compiled_cache_ptr->create_infer_request();
	}
	else {
		std::cout << "==" << model_cache_path << " do not need decryption ==" << std::endl;
		auto ifstr = std::ifstream(model_cache_path, std::ifstream::binary);
		ov_compiled_cache_ptr = std::make_shared<ov::CompiledModel>(core_ptr->import_model(ifstr, device, ov_config));
		auto new_req = ov_compiled_cache_ptr->create_infer_request();
	}
	return ov_compiled_cache_ptr;
}

void saveOVModelCache(
		std::shared_ptr<ov::CompiledModel>& ov_compiled_ptr, 
		std::string model_cache_path,
		bool encrypt_tag) {
	if (encrypt_tag) {
		std::cout << "==== Model Cache Encryption, Saving" << model_cache_path << " ====" << std::endl;
		encryptCompiledModel(ov_compiled_ptr, model_cache_path);
	}
	else {
		std::cout << "==== Model Cache Saving" << model_cache_path << " ====" << std::endl;
		auto ofstr = std::ofstream(model_cache_path, std::ofstream::binary);
		ov_compiled_ptr->export_model(ofstr);
		ofstr.close();
	}
	return;
}

