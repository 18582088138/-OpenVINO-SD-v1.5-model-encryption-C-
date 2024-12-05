#ifndef PIPELINE_H
#define PIPELINE_H

#include <string>
#include <iostream>
#include <fstream>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>


std::string shapeToString(const ov::Shape& shape);

ov::Tensor randn_tensor(uint32_t height, uint32_t width, bool use_np_latents, uint32_t seed = 42);

ov::Tensor postprocess_image(ov::Tensor decoded_image);

ov::Tensor text_encoder_infer(std::shared_ptr<ov::InferRequest> tokenizer_inf_ptr,
	std::shared_ptr<ov::InferRequest> text_encoder_inf_ptr,
	const size_t hd_size,
	std::string& pos_prompt, std::string& neg_prompt);

ov::Tensor unet_infer(std::shared_ptr<ov::InferRequest> unet_inf_req,
	ov::Tensor sample, ov::Tensor timestep,
	ov::Tensor text_embedding_1d);

ov::Tensor vae_decoder_infer(std::shared_ptr<ov::InferRequest> vae_decoder_inf_req, ov::Tensor sample);

void sd_generation(OVStableDiffusionModels ov_sd_models,
	std::string positive_prompt, std::string negative_prompt,
	uint32_t user_seed = 42,
	uint32_t num_images = 1,
	uint32_t num_inference_steps = 4,
	std::vector<uint32_t> gen_shape = { 512,512 });

#endif // PIPELINE_H
