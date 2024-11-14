#define OPENSSL_SUPPRESS_DEPRECATED  //workaround
#include "encrypt_utils.hpp"

#include <openssl/aes.h>
#include <openssl/evp.h>
#include <openssl/err.h>

std::vector<unsigned char> hexStringToBytes(const std::string& hex) {
	std::vector<unsigned char> bytes;
	for (size_t i = 0; i < hex.length(); i += 2) {
		std::string byteString = hex.substr(i, 2);
		unsigned char byte = static_cast<unsigned char>(strtol(byteString.c_str(), nullptr, 16));
		bytes.push_back(byte);
	}
	return bytes;
}


std::vector<unsigned char> serializeCompiledModel(ov::CompiledModel& compiled_model) {
	std::stringstream ss;
	compiled_model.export_model(ss);
	const std::string& str = ss.str();
	std::vector<unsigned char> serializedata(str.begin(), str.end());
	return serializedata;
}

std::vector<unsigned char> readFile(const std::string& filePath) {
	std::ifstream file(filePath, std::ios::binary);
	return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)),
		std::istreambuf_iterator<char>());
}

void writeFile(std::string& filePath, std::vector<unsigned char>& data) {
	std::ofstream file(filePath, std::ios::binary);
	file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

std::vector<unsigned char> aes_128_cbc_encrypt(
		const std::vector<unsigned char>& plaintext,
		const std::vector<unsigned char>& key,
		std::vector<unsigned char> iv) {
	AES_KEY encryptKey;
	AES_set_encrypt_key(key.data(), 128, &encryptKey);
	size_t padding = 16 - (plaintext.size() % 16);
	std::vector<unsigned char> padded_plaintext = plaintext;
	padded_plaintext.insert(padded_plaintext.end(), padding, static_cast<unsigned char>(padding));
	std::vector<unsigned char> ciphertext(padded_plaintext.size());
	AES_cbc_encrypt(padded_plaintext.data(),ciphertext.data(),
		padded_plaintext.size(),&encryptKey,iv.data(),AES_ENCRYPT);
	return ciphertext;
}

std::vector<unsigned char> aes_128_cbc_decrypt(
	const std::vector<unsigned char>& ciphertext,
	const std::vector<unsigned char>& key,
	std::vector<unsigned char> iv) {
	AES_KEY decrptyKey;
	AES_set_decrypt_key(key.data(), 128, &decrptyKey);
	std::vector<unsigned char> plaintext(ciphertext.size());
	AES_cbc_encrypt(ciphertext.data(), plaintext.data(), ciphertext.size(), &decrptyKey, iv.data(), AES_DECRYPT);
	// remove PKCS#7 padding
	size_t padding = plaintext.back();
	if (padding > 16 || padding == 0) {
		throw std::runtime_error("Invalid padding during decryption");
	}
	plaintext.resize(plaintext.size() - padding);
	return plaintext;
}

void encryptCompiledModel(
		std::shared_ptr<ov::CompiledModel>& ov_compiled_ptr,
		std::string encrypt_model_path,
		std::string hexKey,
		std::string hexIV) {
	std::vector<unsigned char> key = hexStringToBytes(hexKey);
	std::vector<unsigned char> iv = hexStringToBytes(hexIV);
	std::vector<unsigned char> plaintext = serializeCompiledModel(*ov_compiled_ptr);
	std::vector<unsigned char> ciphertext = aes_128_cbc_encrypt(plaintext, key, iv);
	writeFile(encrypt_model_path, ciphertext);
	return;
}

std::istringstream decryptCompiledModel(
		std::string decrypt_model_path,
		std::string hexKey,
		std::string hexIV) {
	std::shared_ptr<ov::CompiledModel> ov_compiled_model;
	std::vector<unsigned char> key = hexStringToBytes(hexKey);
	std::vector<unsigned char> iv = hexStringToBytes(hexIV);
	std::vector<unsigned char> ciphertext = readFile(decrypt_model_path);
	std::vector<unsigned char> plaintext = aes_128_cbc_decrypt(ciphertext, key, iv);
	std::string str2(plaintext.begin(), plaintext.end());
	std::istringstream plaintextstr(str2);
	return plaintextstr;
}