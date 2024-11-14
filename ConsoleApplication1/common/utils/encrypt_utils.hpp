#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cctype>
#include <random>

#include <regex>
#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/err.h>

#include <openvino/runtime/core.hpp>

std::vector<unsigned char> hexStringToBytes(const std::string& hex);

std::vector<unsigned char> serializeCompiledModel(ov::CompiledModel& compiled_model);

std::vector<unsigned char> readFile(const std::string& filePath);

void writeFile(std::string& filePath, std::vector<unsigned char>& data);

std::vector<unsigned char> aes_128_cbc_encrypt(
	const std::vector<unsigned char>& plaintext,
	const std::vector<unsigned char>& key,
	std::vector<unsigned char> iv);

std::vector<unsigned char> aes_128_cbc_decrypt(
	const std::vector<unsigned char>& ciphertext,
	const std::vector<unsigned char>& key,
	std::vector<unsigned char> iv);

void encryptCompiledModel(
	std::shared_ptr<ov::CompiledModel>& ov_compiled_ptr,
	std::string encrypt_model_path,
	std::string hexKey = "6f70656e76696e6f20656e6372797074",
	std::string hexIV = "6f70656e76696e6f20656e6372797074");

std::istringstream decryptCompiledModel(
	std::string decrypt_model_path,
	std::string hexKey = "6f70656e76696e6f20656e6372797074",
	std::string hexIV = "6f70656e76696e6f20656e6372797074");
