#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <iostream>
#include <fstream>

#include <openvino/runtime/core.hpp>

#include "model_utils.hpp"
#include "encrypt_utils.hpp"

struct OVStableDiffusionModels {
	std::shared_ptr<ov::CompiledModel> tokenizer_model;
	std::shared_ptr<ov::CompiledModel> text_encoder_model;
	std::shared_ptr<ov::CompiledModel> unet_model;
	std::shared_ptr<ov::CompiledModel> vae_decoder_model;

	std::shared_ptr<ov::InferRequest> tokenizer_inf_req;
	std::shared_ptr<ov::InferRequest> text_encoder_inf_req;
	std::shared_ptr<ov::InferRequest> unet_inf_req;
	std::shared_ptr<ov::InferRequest> vae_decoder_inf_req;

	std::string tokenizer_cache_path;
	std::string text_encoder_cache_path;
	std::string unet_cache_path;
	std::string vae_decoder_cache_path;

	void load_model(std::shared_ptr<ov::Core>& core_ptr,
			std::string model_path,
			std::string device = "GPU",
			ov::AnyMap ov_config = {}) {
		std::string tokenizer_path = model_path + "/tokenizer/";
		std::string text_encoder_path = model_path + "/text_encoder/";
		std::string unet_path = model_path + "/unet/";
		std::string vae_decoder_path = model_path + "/vae_decoder/";

		tokenizer_model = readOVModel(core_ptr, tokenizer_path, "CPU");
		text_encoder_model = readOVModel(core_ptr, text_encoder_path, device, ov_config);
		unet_model = readOVModel(core_ptr, unet_path, device, ov_config);
		vae_decoder_model = readOVModel(core_ptr, vae_decoder_path, device, ov_config);
		std::cout << "==== OpenVINO StableDiffusion Models Loading Success ====" << std::endl;
	}

	void save_model_cache(std::string model_path, bool encrypt_tag = true) {
		if (encrypt_tag) {
			//tokenizer_cache_path = model_path + "/tokenizer_enc.blob";
			text_encoder_cache_path = model_path + "/text_encoder_enc.blob";
			unet_cache_path = model_path + "/unet_enc.blob";
			vae_decoder_cache_path = model_path + "/vae_decoder_enc.blob";
		}
		else {
			//tokenizer_cache_path = model_path + "/tokenizer.blob";
			text_encoder_cache_path = model_path + "/text_encoder.blob";
			unet_cache_path = model_path + "/unet.blob";
			vae_decoder_cache_path = model_path + "/vae_decoder.blob";
		}
		//openvino_tokenizer model can't use weightless cache feature!!!
		//saveOVModelCache(tokenizer_model, tokenizer_cache_path, encrypt_tag);
		saveOVModelCache(text_encoder_model, text_encoder_cache_path, encrypt_tag);
		saveOVModelCache(unet_model, unet_cache_path, encrypt_tag);
		saveOVModelCache(vae_decoder_model, vae_decoder_cache_path, encrypt_tag);
		std::cout << "==== OV SD Models Encryption Success, Saving ====" << std::endl;
		return;
	}

	void load_model_cache(std::shared_ptr<ov::Core>& core_ptr,
			std::string model_path,
			std::string device = "GPU",
			ov::AnyMap ov_config = {},
			bool encrypt_tag = true) {
		std::string tokenizer_path = model_path + "/tokenizer/";
		tokenizer_model = readOVModel(core_ptr, tokenizer_path, "CPU");

		if (encrypt_tag) {
			//tokenizer_cache_path = model_path + "/tokenizer_enc.blob";
			text_encoder_cache_path = model_path + "/text_encoder_enc.blob";
			unet_cache_path = model_path + "/unet_enc.blob";
			vae_decoder_cache_path = model_path + "/vae_decoder_enc.blob";
		}
		else {
			//tokenizer_cache_path = model_path + "/tokenizer.blob";
			text_encoder_cache_path = model_path + "/text_encoder.blob";
			unet_cache_path = model_path + "/unet.blob";
			vae_decoder_cache_path = model_path + "/vae_decoder.blob";
		}
		//tokenizer_model = readOVModelCache(core_ptr, tokenizer_cache_path, "CPU", {}, encrypt_tag);
		text_encoder_model = readOVModelCache(core_ptr, text_encoder_cache_path, device, ov_config, encrypt_tag);
		unet_model = readOVModelCache(core_ptr, unet_cache_path, device, ov_config, encrypt_tag);
		vae_decoder_model = readOVModelCache(core_ptr, vae_decoder_cache_path, device, ov_config, encrypt_tag);
		std::cout << "==== OV SD Models Cache Loading Success ====" << std::endl;
		return;
	}

	void create_infer_req() {
		if (!tokenizer_model) {
			std::cerr << "Error: The ptr is nullptr, pls check the sd model has been loaded" << std::endl;
			std::exit(EXIT_FAILURE);
		}
		tokenizer_inf_req = std::make_shared<ov::InferRequest>(tokenizer_model->create_infer_request());
		text_encoder_inf_req = std::make_shared<ov::InferRequest>(text_encoder_model->create_infer_request());
		unet_inf_req = std::make_shared<ov::InferRequest>(unet_model->create_infer_request());
		vae_decoder_inf_req = std::make_shared<ov::InferRequest>(vae_decoder_model->create_infer_request());
		std::cout << "==== OV SD Create InferRequest Success ====" << std::endl;
		return;
	}

	void unload_infer_req() {
		// reset inference request
		tokenizer_inf_req.reset();
		text_encoder_inf_req.reset();
		unet_inf_req.reset();
		vae_decoder_inf_req.reset();
		std::cout << "==== OV SD InferRequest UnLoading Success ====" << std::endl;
	}

	void unload_model() {
		// reset intermediate variable(CPU only)
		tokenizer_model->release_memory();
		text_encoder_model->release_memory();
		unet_model->release_memory();
		vae_decoder_model->release_memory();

		// reset compiled model
		tokenizer_model.reset();
		text_encoder_model.reset();
		unet_model.reset();
		vae_decoder_model.reset();
		std::cout << "==== OV SD Models UnLoading Success ====" << std::endl;
	}

	void unload() {
		unload_infer_req();
		unload_model();
	}
};
#endif
