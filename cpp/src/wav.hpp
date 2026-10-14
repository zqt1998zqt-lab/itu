#pragma once
#include <string>
#include <vector>
namespace itu {
struct Wav { unsigned sample_rate=0, channels=0; std::vector<float> samples; };
Wav read_wav(const std::string& path);
void write_wav(const std::string& path, unsigned rate, unsigned channels, const std::vector<float>& samples);
}
