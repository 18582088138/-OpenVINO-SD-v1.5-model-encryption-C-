#include <string>
#include <iostream>

#include <fstream>

#include <openvino/runtime/core.hpp>
#include <openvino/runtime/intel_gpu/properties.hpp>

#include "common/utils/model.cpp"
#include "common/utils/parameter.hpp"
#include "common/diffusers/include/scheduler_lcm.hpp"
#include "common/imwrite/include/imwrite.hpp"

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

std::string shapeToString(const ov::Shape& shape) {
	std::string result = "[";
	for (size_t i = 0; i < shape.size(); ++i) {
		result += std::to_string(shape[i]);
		if (i < shape.size() - 1) {
			result += ", ";
		}
	}
	result += "]";
	return result;
}

class Timer {
	const decltype(std::chrono::steady_clock::now()) m_start;
public:
	Timer(const std::string& scope) :
		m_start(std::chrono::steady_clock::now()) {
		(std::cout << scope << ": ").flush();
	}
	~Timer() {
		auto m_end = std::chrono::steady_clock::now();
		std::cout << std::chrono::duration<double, std::milli>(m_end - m_start).count() << " ms" << std::endl;
	}
};


ov::Tensor randn_tensor(uint32_t height, uint32_t width, bool use_np_latents, uint32_t seed = 42) {
	ov::Tensor noise(ov::element::f32, { 1, 4, height / 8, width / 8 });
	if (use_np_latents) {
		// read np generated latents with defaut seed 42
		const char* latent_file_name = "../scripts/np_latents_512x512.txt";
		std::ifstream latent_copy_file(latent_file_name, std::ios::ate);
		OPENVINO_ASSERT(latent_copy_file.is_open(), "Cannot open ", latent_file_name);

		size_t file_size = latent_copy_file.tellg() / sizeof(float);
		OPENVINO_ASSERT(file_size >= noise.get_size(), "Cannot generate ", noise.get_shape(), " with ", latent_file_name, ". File size is small");

		latent_copy_file.seekg(0, std::ios::beg);
		for (size_t i = 0; i < noise.get_size(); ++i)
			latent_copy_file >> noise.data<float>()[i];
	}
	else {
		std::mt19937 gen{ seed };
		std::normal_distribution<float> normal{ 0.0f, 1.0f };
		std::generate_n(noise.data<float>(), noise.get_size(), [&]() {
			return normal(gen);
			});
	}
	return noise;
}

ov::Tensor postprocess_image(ov::Tensor decoded_image) {
	ov::Tensor generated_image(ov::element::u8, decoded_image.get_shape());

	// convert to u8 image
	const float* decoded_data = decoded_image.data<const float>();
	std::uint8_t* generated_data = generated_image.data<std::uint8_t>();
	for (size_t i = 0; i < decoded_image.get_size(); ++i) {
		generated_data[i] = static_cast<std::uint8_t>(std::clamp(decoded_data[i] * 0.5f + 0.5f, 0.0f, 1.0f) * 255);
	}
	return generated_image;
}

ov::Tensor text_encoder_infer(std::shared_ptr<ov::InferRequest> tokenizer_inf_ptr, 
			std::shared_ptr<ov::InferRequest> text_encoder_inf_ptr, 
			const size_t hd_size,
			std::string& pos_prompt, std::string& neg_prompt) {
	if (!tokenizer_inf_ptr || !text_encoder_inf_ptr) {
		throw std::runtime_error("Invalid model or inf_req ptr is null, pls verify the model load");
	}
	const size_t MAX_LENGTH = 77;
	const size_t HIDDEN_SIZE = hd_size;
	const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
	const ov::Shape input_ids_shape({ 1, MAX_LENGTH });
	ov::Tensor input_ids(ov::element::i32, input_ids_shape);
	ov::Tensor text_embeddings(ov::element::f32, { 2, MAX_LENGTH, HIDDEN_SIZE });

	auto compute_text_embedding = [&](std::string& prompt, ov::Tensor encoder_output_tensor) {
		std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), PAD_TOKEN_ID);
		tokenizer_inf_ptr->set_input_tensor(ov::Tensor{ ov::element::string,{1},&prompt });
		tokenizer_inf_ptr->start_async();
		tokenizer_inf_ptr->wait();
		ov::Tensor input_ids_token = tokenizer_inf_ptr->get_tensor("input_ids");
		std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());
		
		text_encoder_inf_ptr->set_tensor("input_ids",input_ids);
		text_encoder_inf_ptr->set_output_tensor(0, encoder_output_tensor);
		text_encoder_inf_ptr->start_async();
		text_encoder_inf_ptr->wait();
		};

	compute_text_embedding(neg_prompt, ov::Tensor(text_embeddings, { 0,0,0 }, { 1,MAX_LENGTH,HIDDEN_SIZE }));
	compute_text_embedding(pos_prompt, ov::Tensor(text_embeddings, { 1,0,0 }, { 2,MAX_LENGTH,HIDDEN_SIZE }));
	return text_embeddings;
}

ov::Tensor unet_infer(std::shared_ptr<ov::InferRequest> unet_inf_req, 
			ov::Tensor sample, ov::Tensor timestep, 
			ov::Tensor text_embedding_1d) {
	auto sample_shape = sample.get_shape();
	unet_inf_req->set_tensor("sample", sample);
	unet_inf_req->set_tensor("timestep", timestep);
	unet_inf_req->set_tensor("encoder_hidden_states", text_embedding_1d);
	unet_inf_req->start_async();
	unet_inf_req->wait();

	ov::Tensor noise_pred_tensor = unet_inf_req->get_output_tensor();
	ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
	noise_pred_shape[0] = 1;

	const float guidance_scale = 7.5f;
	const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
	const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

	ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
	for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
		noisy_residual.data<float>()[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);
	return noisy_residual;
}

ov::Tensor vae_decoder_infer(std::shared_ptr<ov::InferRequest> vae_decoder_inf_req, ov::Tensor sample) {
	const float coeffs_const{ 1 / 0.18215 };
	for (size_t i = 0; i < sample.get_size(); ++i)
		sample.data<float>()[i] *= coeffs_const;
	vae_decoder_inf_req->set_input_tensor(sample);
	vae_decoder_inf_req->start_async();
	vae_decoder_inf_req->wait();
	ov::Tensor vae_result = vae_decoder_inf_req->get_output_tensor();
	return vae_result;
}

void sd_generation(OVStableDiffusionModels ov_sd_models, 
			std::string positive_prompt, std::string negative_prompt,
			uint32_t user_seed = 42,
			uint32_t num_images = 1,
			uint32_t num_inference_steps = 4,
			uint32_t img_height = 512,
			uint32_t img_width = 512) {
	Timer t("Running OV Stable Diffusion pipeline");
	const size_t hd_size = static_cast<size_t>(ov_sd_models.text_encoder_model->output(0).get_partial_shape()[2].get_length());
	ov::Tensor text_embeddings = text_encoder_infer(
				ov_sd_models.tokenizer_inf_req, 
				ov_sd_models.text_encoder_inf_req,
				hd_size, positive_prompt, negative_prompt);
	std::shared_ptr<Scheduler> scheduler = std::make_shared<LCMScheduler>();
	scheduler->set_timesteps(num_inference_steps);
	std::vector<std::int64_t> timesteps = scheduler->get_timesteps();

	for (uint32_t n = 0; n < num_images; n++) {
		std::uint32_t seed = num_images == 1 ? user_seed : user_seed + n;
		const bool read_np_latent = false;
		ov::Tensor noise = randn_tensor(img_height, img_width, read_np_latent, seed);

		ov::Shape latent_shape = noise.get_shape();
		ov::Shape latent_model_input_shape = latent_shape;
		latent_model_input_shape[0] = 2; // Unet accepts batch 2

		ov::Tensor latent(ov::element::f32, latent_shape), latent_model_input(ov::element::f32, latent_model_input_shape);
		for (size_t i = 0; i < noise.get_size(); ++i) {
			latent.data<float>()[i] = noise.data<float>()[i] * scheduler->get_init_noise_sigma();
		}
		for (size_t inference_step = 0; inference_step < num_inference_steps; inference_step++) {
			// concat the same latent twice along a batch dimension
			latent.copy_to(ov::Tensor(latent_model_input, { 0, 0, 0, 0 }, { 1, latent_shape[1], latent_shape[2], latent_shape[3] }));
			latent.copy_to(ov::Tensor(latent_model_input, { 1, 0, 0, 0 }, { 2, latent_shape[1], latent_shape[2], latent_shape[3] }));
			scheduler->scale_model_input(latent_model_input, inference_step);
			ov::Tensor timestep(ov::element::i64, { 1 }, &timesteps[inference_step]);   //this is original

			ov::Tensor noisy_residual = unet_infer(ov_sd_models.unet_inf_req, latent_model_input, timestep, text_embeddings);
			latent = scheduler->step(noisy_residual, latent, inference_step)["latent"];
		}
		ov::Tensor decoded_image = vae_decoder_infer(ov_sd_models.vae_decoder_inf_req, latent);
		std::string img_save_path = std::string("./images/seed_") + std::to_string(seed) + "_steps" + std::to_string(num_inference_steps)+"_" + positive_prompt + ".bmp";
		imwrite(img_save_path, postprocess_image(decoded_image), true);
		std::cout << "OV SD Inference Successfully" << std::endl;
	}
	return;
}

int main(int argc, char* argv[]) {
	std::shared_ptr<ov::Core> core_ptr = std::make_shared<ov::Core>();
	//core_ptr->add_extension("openvino_tokenizersd.dll");
	core_ptr->add_extension("openvino_tokenizers.dll");
	ParameterConfig parameter_config;
	parameter_config.parameters(argc, argv);

	std::cout << "Model Path: " << parameter_config.model_path << std::endl;
	std::cout << "Target Device: " << parameter_config.device << std::endl;

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

	if (device.find("CPU") != std::string::npos) {
		core_ptr->set_property("CPU", { {"CPU_RUNTIME_CACHE_CAPACITY","0"}});
		std::cout << "Paraformer:: Set CPU_RUNTIME_CACHE_CAPACITY = 0 \n" << std::endl;
	}
	//core_ptr->set_property(ov::cache_dir("./ov_cache_dir"));
	ov::AnyMap ov_config = {};
	// This two property can not use on openvino tokenizer model ???
	ov_config[ov::cache_mode.name()] = ov::CacheMode::OPTIMIZE_SPEED;
	ov_config[ov::intel_gpu::hint::enable_kernels_reuse.name()] = true;

	OVStableDiffusionModels ov_sd_models;
	//ov_sd_models.load_model(core_ptr, model_path, device, ov_config);
	//ov_sd_models.save_model_cache(model_path, encrypt_tag);
	//ov_sd_models.load_model_cache(core_ptr, model_path, device, ov_config, encrypt_tag);
	//ov_sd_models.create_infer_req();
	//ov_sd_models.unload();

	//============================Testing=============================================
	ov_sd_models.load_model_cache(core_ptr, model_path, device, ov_config, encrypt_tag);
	//ov_sd_models.load_model(core_ptr, model_path, device, ov_config);
	ov_sd_models.create_infer_req();
	std::string positive_prompt = "tiger";
	std::string negative_prompt = "";
	sd_generation(ov_sd_models, positive_prompt, negative_prompt, user_seed, num_images, num_inference_steps);
	//ov_sd_models.unload();
	//============================Testing=============================================

	uint32_t x = 0;
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
	ov_sd_models.unload();
	std::cout << "Completed all iterations, entering monitoring loop..." << std::endl;

	
	std::cout << "exit this app now!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
	
	return 0;
}