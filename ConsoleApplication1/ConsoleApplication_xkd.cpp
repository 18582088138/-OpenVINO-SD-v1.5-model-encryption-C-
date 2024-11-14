// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iostream>
#include <string>
#include <cstring>
#include <random>
#include <fstream>
#include <filesystem>
#include <utility>
#include <format>

#include "openvino/runtime/core.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

#include "cxxopts.hpp"
//#include "scheduler_lms_discrete.hpp"
#include "scheduler_lcm.hpp"
//#include "lora.hpp"
#include "imwrite.hpp"
#include <thread>
#define DEBUG_MEMORY
#if defined(_WIN32) && defined(DEBUG_MEMORY)
#include <windows.h>
#define PSAPI_VERSION 1 // PrintMemoryInfo 
#include <psapi.h>
#pragma comment(lib,"psapi.lib") //PrintMemoryInfod
#include "processthreadsapi.h"
#endif

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
// 打印函数
std::string shapeToString(const ov::Shape& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]); // 假设shape支持下标访问
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

struct StableDiffusionCompiledModels {
    std::unique_ptr<ov::CompiledModel> text_encoder_ptr;
    std::unique_ptr<ov::CompiledModel> unet_ptr;
    std::unique_ptr<ov::CompiledModel> vae_decoder_ptr;
    std::unique_ptr<ov::CompiledModel> tokenizer_ptr;

    void unload_compile() {
        std::cout << "==== Releasing StableDiffusionCompiledModels resources ====\n";
        text_encoder_ptr->release_memory();
        unet_ptr->release_memory();
        vae_decoder_ptr->release_memory();
        tokenizer_ptr->release_memory();

        text_encoder_ptr.reset();
        unet_ptr.reset();
        vae_decoder_ptr.reset();
        tokenizer_ptr.reset();
        std::cout << "==== Released StableDiffusionCompiledModels resources successfully ====\n";
    }
};

struct StableDiffusionInferReqs {
    std::unique_ptr<ov::InferRequest> text_encoder_req_ptr;
    std::unique_ptr<ov::InferRequest> unet_req_ptr;
    std::unique_ptr<ov::InferRequest> vae_decoder_req_ptr;
    std::unique_ptr<ov::InferRequest> tokenizer_req_ptr;

    void unload_compile() {
        std::cout << "==== Releasing StableDiffusionInferReqs resources ====\n";
        text_encoder_req_ptr.reset();
        unet_req_ptr.reset();
        vae_decoder_req_ptr.reset();
        tokenizer_req_ptr.reset();
        std::cout << "==== Released StableDiffusionInferReqs resources successfully ====\n";
    }
};

std::pair<StableDiffusionCompiledModels, StableDiffusionInferReqs> 
                    sd_model_prepare(
                        std::unique_ptr<ov::Core>& ov_core_ptr,
                                     const std::string& model_path,
                                     const std::string& device,
                                     const std::string& lora_path, 
                                     const float alpha, 
                                     const bool use_cache) {
    StableDiffusionCompiledModels sd_models;
    StableDiffusionInferReqs sd_reqs;

    //std::unique_ptr<ov::Core> ov_core_ptr = std::make_unique<ov::Core>();
    if (device.find("CPU") != std::string::npos) {
        ov_core_ptr->set_property("CPU", { {"CPU_RUNTIME_CACHE_CAPACITY", "0"} });
        std::cout << "Paraformer:: Set CPU_RUNTIME_CACHE_CAPACITY 0\n";
    }
    //if (device.find("GPU") != std::string::npos) {
    //    ov_core_ptr->set_property("GPU", { {"GPU_RUNTIME_CACHE_CAPACITY", "1"} });
    //    std::cout << "Paraformer:: Set GPU_RUNTIME_CACHE_CAPACITY 1\n";
    //}
    if (use_cache)
        ov_core_ptr->set_property(ov::cache_dir("./cache_dir"));
    //core.add_extension(TOKENIZERS_LIBRARY_PATH);
    ov_core_ptr->add_extension("openvino_tokenizers.dll");

    // Text encoder
    {
        Timer t("==== Loading and compiling text encoder ====");
        auto text_encoder_model = ov_core_ptr->read_model(model_path + "/text_encoder/openvino_model.xml");
        sd_models.text_encoder_ptr = std::make_unique<ov::CompiledModel>(ov_core_ptr->compile_model(text_encoder_model, device));
        sd_reqs.text_encoder_req_ptr = std::make_unique<ov::InferRequest>(sd_models.text_encoder_ptr->create_infer_request());
        text_encoder_model.reset();
    }

    // Unet
    {
        Timer t("==== Loading and compiling UNet ====");
        auto unet_model = ov_core_ptr->read_model(model_path + "/unet/openvino_model.xml");
        ov::preprocess::PrePostProcessor unet_ppp(unet_model);

        // Specifically targeting the input by name
        unet_ppp.input("sample").tensor()  // Replace with the actual input name
            .set_element_type(ov::element::f32) // Set input tensor element type, NOT ov::element::i8
            .set_shape({ -1, 4, 64, 64 });       // Set input tensor shape, not 1, 4, 64, 64 ;  not 2, 4, 512, 512  (512 is  cmd ConsoleApplication1.exe  -d GPU  -p "A boy is swimming"    -m  "C:\\sd_ov_int8"  --width 512 --height 512 )
        // Configure output tensor preprocessing
        unet_ppp.output(0).tensor()  // Assuming you have one output or want to configure the first one
            .set_element_type(ov::element::f32);  // Set output tensor element type, NOT ov::element::i8
        unet_ppp.output().tensor().set_layout("NCHW");

        sd_models.unet_ptr = std::make_unique<ov::CompiledModel>(ov_core_ptr->compile_model(unet_ppp.build(), device));
        sd_reqs.unet_req_ptr = std::make_unique<ov::InferRequest>(sd_models.unet_ptr->create_infer_request());
        unet_model.reset();
    }

    // VAE decoder
    {
        Timer t("==== Loading and compiling VAE decoder ====");
        auto vae_decoder_model = ov_core_ptr->read_model(model_path + "/vae_decoder/openvino_model.xml");

        ov::preprocess::PrePostProcessor vae_decoder_ppp(vae_decoder_model);
        vae_decoder_ppp.output().model().set_layout("NCHW");
        vae_decoder_ppp.output().tensor().set_layout("NHWC");

        sd_models.vae_decoder_ptr = std::make_unique<ov::CompiledModel>(ov_core_ptr->compile_model(vae_decoder_ppp.build(), device));
        sd_reqs.vae_decoder_req_ptr = std::make_unique<ov::InferRequest>(sd_models.vae_decoder_ptr->create_infer_request());
        vae_decoder_model.reset();
    }

    // Tokenizer
    {
        Timer t("Loading and compiling tokenizer");
        // Tokenizer model wil be loaded to CPU: OpenVINO Tokenizers can be inferred on a CPU device only.
        auto tokenizer_model = ov_core_ptr->read_model(model_path + "/tokenizer/openvino_tokenizer.xml");
        sd_models.tokenizer_ptr = std::make_unique<ov::CompiledModel>(ov_core_ptr->compile_model(tokenizer_model,"CPU"));
        sd_reqs.tokenizer_req_ptr = std::make_unique<ov::InferRequest>(sd_models.tokenizer_ptr->create_infer_request());
        tokenizer_model.reset();
    }
    //retturn 
    return std::make_pair(std::move(sd_models), std::move(sd_reqs));
}

ov::Tensor text_encoder(StableDiffusionCompiledModels* sd_models, StableDiffusionInferReqs* sd_reqs, std::string& pos_prompt, std::string& neg_prompt) {
    if (!sd_models || !sd_models->text_encoder_ptr) {
        throw std::runtime_error("Invalid model or text_encoder_ptr is null");
    }
    const size_t MAX_LENGTH = 77; // 'model_max_length' from 'tokenizer_config.json'
    const size_t HIDDEN_SIZE = static_cast<size_t>(sd_models->text_encoder_ptr->output(0).get_partial_shape()[2].get_length());
    const int32_t EOS_TOKEN_ID = 49407, PAD_TOKEN_ID = EOS_TOKEN_ID;
    const ov::Shape input_ids_shape({ 1, MAX_LENGTH });
    ov::Tensor input_ids(ov::element::i32, input_ids_shape);
    ov::Tensor text_embeddings(ov::element::f32, { 2, MAX_LENGTH, HIDDEN_SIZE });

    auto compute_text_embeddings = [&](std::string& prompt, ov::Tensor encoder_output_tensor) {
        std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), PAD_TOKEN_ID);
        sd_reqs->tokenizer_req_ptr->set_input_tensor(ov::Tensor{ ov::element::string, {1}, &prompt });
        sd_reqs->tokenizer_req_ptr->start_async();
        sd_reqs->tokenizer_req_ptr->wait();
        ov::Tensor input_ids_token = sd_reqs->tokenizer_req_ptr->get_tensor("input_ids");
        std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());
        std::cout << "== tokenizer infer done ==" << std::endl;

        sd_reqs->text_encoder_req_ptr->set_tensor("input_ids", input_ids);
        sd_reqs->text_encoder_req_ptr->set_output_tensor(0, encoder_output_tensor);
        sd_reqs->text_encoder_req_ptr->start_async();
        sd_reqs->text_encoder_req_ptr->wait();
        std::cout << "== text_encoder infer done ==" << std::endl;
        };
    
    compute_text_embeddings(neg_prompt, ov::Tensor(text_embeddings, { 0, 0, 0 }, { 1, MAX_LENGTH, HIDDEN_SIZE }));
    compute_text_embeddings(pos_prompt, ov::Tensor(text_embeddings, { 1, 0, 0 }, { 2, MAX_LENGTH, HIDDEN_SIZE }));
    std::cout << "==== compute_text_embeddings done ====" << std::endl;

    return text_embeddings;
}


ov::Tensor unet(StableDiffusionInferReqs* sd_reqs, ov::Tensor sample, ov::Tensor timestep, ov::Tensor text_embedding_1d) {
    sd_reqs->unet_req_ptr->set_tensor("sample", sample);
    sd_reqs->unet_req_ptr->set_tensor("timestep", timestep);
    sd_reqs->unet_req_ptr->set_tensor("encoder_hidden_states", text_embedding_1d);
    sd_reqs->unet_req_ptr->start_async();
    sd_reqs->unet_req_ptr->wait();

    ov::Tensor noise_pred_tensor = sd_reqs->unet_req_ptr->get_output_tensor();
    ov::Shape noise_pred_shape = noise_pred_tensor.get_shape();
    noise_pred_shape[0] = 1;

    // perform guidance
    const float guidance_scale = 7.5f;
    const float* noise_pred_uncond = noise_pred_tensor.data<const float>();
    const float* noise_pred_text = noise_pred_uncond + ov::shape_size(noise_pred_shape);

    ov::Tensor noisy_residual(noise_pred_tensor.get_element_type(), noise_pred_shape);
    for (size_t i = 0; i < ov::shape_size(noise_pred_shape); ++i)
        noisy_residual.data<float>()[i] = noise_pred_uncond[i] + guidance_scale * (noise_pred_text[i] - noise_pred_uncond[i]);

    return noisy_residual;
}

ov::Tensor vae_decoder(StableDiffusionInferReqs* sd_reqs, ov::Tensor sample) {
    const float coeffs_const{ 1 / 0.18215 };
    for (size_t i = 0; i < sample.get_size(); ++i)
        sample.data<float>()[i] *= coeffs_const;
    sd_reqs->vae_decoder_req_ptr->set_input_tensor(sample);
    sd_reqs->vae_decoder_req_ptr->start_async();
    sd_reqs->vae_decoder_req_ptr->wait();
    return sd_reqs->vae_decoder_req_ptr->get_output_tensor();
}

void picture() {
    const std::string folder_name = "images";
    try {
        std::filesystem::create_directory(folder_name);
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to create dir" << e.what() << std::endl;
    }
    std::cout << "OpenVINO version: " << ov::get_openvino_version() << std::endl;
    const std::string& model_base_path = "C:/Users/kk/Downloads/models/xiaomi_sd1.5_lcm_ov4.0";
    //const std::string& model_base_path = "D:/SD/sd1.5_v2/sd1.5_en_ftv2_ov_int8_fix_4_0";
    //const std::string& model_base_path = "C:/Users/kk/Downloads/xkd_1022/models/stable-diffusion-v1-5-int8-quant-ov";
    const std::string& device = "GPU";
    const std::string& lora_path = "";
    const float alpha = 0.75;
    const bool use_cache = true;
    const uint32_t user_seed = 42;
    const uint32_t height = 512;
    const uint32_t width = 512;
    const uint32_t num_images = 1;
    const uint32_t num_inference_steps = 4;

    std::unique_ptr<ov::Core> ov_core_ptr = std::make_unique<ov::Core>();
    auto [sd_models, sd_reqs] = sd_model_prepare(ov_core_ptr, model_base_path, device, lora_path, alpha, use_cache);
    //auto [sd_models, sd_reqs] = sd_model_prepare(model_base_path, device, lora_path, alpha, use_cache);
    Timer t("Running Stable Diffusion pipeline");
    //std::string positive_prompt = "lion";
    //std::string positive_prompt = "tiger";
    std::string positive_prompt = "grapes";
    //std::string positive_prompt = "watermelon";
    //std::string positive_prompt = "apple";
    //std::string positive_prompt = "A boy is swimming, and there are a lot of red flowers around the pool.";
    std::string negative_prompt = "";

    ov::Tensor text_embeddings = text_encoder(&sd_models, &sd_reqs, positive_prompt, negative_prompt);
    std::shared_ptr<Scheduler> scheduler = std::make_shared<LCMScheduler>();
    scheduler->set_timesteps(num_inference_steps);
    std::vector<std::int64_t> timesteps = scheduler->get_timesteps();

    for (uint32_t n = 0; n < num_images; n++) {
        std::uint32_t seed = num_images == 1 ? user_seed : user_seed + n;
        const bool read_np_latent = false;
        ov::Tensor noise = randn_tensor(height, width, read_np_latent, seed);

        // latents are multiplied by 'init_noise_sigma'
        ov::Shape latent_shape = noise.get_shape(), latent_model_input_shape = latent_shape;
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

            ov::Tensor noisy_residual = unet(&sd_reqs, latent_model_input, timestep, text_embeddings);
            latent = scheduler->step(noisy_residual, latent, inference_step)["latent"];
        }
        ov::Tensor decoded_image = vae_decoder(&sd_reqs, latent);
        imwrite(std::string("./images/seed_") + std::to_string(seed) + positive_prompt + ".bmp", postprocess_image(decoded_image), true);
        std::cout << "Successfully inference " << std::endl;
    }

    sd_reqs.unload_compile();
    sd_models.unload_compile();
    ov_core_ptr->unload_plugin(device);
    ov_core_ptr.reset();
    return;
}


int32_t main_bak(int32_t argc, char* argv[]) {
    uint32_t x = 0;
    while (x < 1) {
        try {
            std::cout << "Starting iteration " << x + 1 << " of 2..." << std::endl;
            picture();
            std::cout << "Successfully completed iteration " << x + 1 << std::endl;
            std::cout << "Waiting 20 seconds before next iteration..." << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(5));
            x += 1;
#ifdef DEBUG_MEMORY
            DebugMemoryInfo(std::format("round {}",x).c_str());
#endif
        }
        catch (const std::exception& e) {
            std::cerr << "Error during iteration " << x + 1 << ": " << e.what() << std::endl;
            // 可以选择是否继续循环
            x += 1;  // 即使发生错误也继续下一次迭代
            // 或者在这里添加一些恢复逻辑
        }
        catch (...) {
            std::cerr << "Unknown error during iteration " << x + 1 << std::endl;
            x += 1;  // 即使发生错误也继续下一次迭代
        }
    }
    std::cout << "Completed all iterations, entering monitoring loop..." << std::endl;
  
    /*while (true) {
        std::cout << "Now I am printing, pls check this process memory size ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }*/
    #ifdef DEBUG_MEMORY
    for (int i = 0; i < 2; ++i) {
      //  std::this_thread::sleep_for(std::chrono::seconds(5));
        DebugMemoryInfo("finally");
    }
    #endif
    std::cout << "exit this app now!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    return 1;
}