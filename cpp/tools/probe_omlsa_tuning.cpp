#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef ITU_OMLSA_ORIGINAL_BUILD
#include "itu/omlsa_config.hpp"
#endif
#include "omlsa_imcra.hpp"

namespace {

struct Options {
  std::string input;
  std::string output;
  unsigned rate = 0;
  float floor_db = -9.f;
  float noise_update_scale = 1.f;
  float speech_recovery = 0.f;
  float release_ms = 60.f;
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument(message +
      "\nusage: probe_omlsa_tuning --input-f32 PATH --output-f32 PATH "
      "--rate RATE --noise-update-scale VALUE --speech-recovery VALUE "
      "--release-ms VALUE [--floor-db VALUE]");
}

std::string take_value(int& index, int argc, char** argv) {
  if (++index >= argc) usage_error(std::string("missing value for ") + argv[index - 1]);
  return argv[index];
}

float parse_float(const std::string& text, const char* name) {
  if (text.empty()) usage_error(std::string("empty value for ") + name);
  char* end = nullptr;
  errno = 0;
  const float value = std::strtof(text.c_str(), &end);
  if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value))
    usage_error(std::string("invalid finite float for ") + name + ": " + text);
  return value;
}

unsigned parse_unsigned(const std::string& text, const char* name) {
  if (text.empty() || text.front() == '-') usage_error(std::string("invalid unsigned value for ") + name);
  char* end = nullptr;
  errno = 0;
  const unsigned long long value = std::strtoull(text.c_str(), &end, 10);
  if (errno == ERANGE || end != text.c_str() + text.size() || value > std::numeric_limits<unsigned>::max())
    usage_error(std::string("invalid unsigned value for ") + name + ": " + text);
  return static_cast<unsigned>(value);
}

Options parse_options(int argc, char** argv) {
  Options options;
  bool have_input = false, have_output = false, have_rate = false;
  bool have_noise = false, have_speech = false, have_release = false, have_floor = false;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      std::cout << "usage: probe_omlsa_tuning --input-f32 PATH --output-f32 PATH "
                   "--rate RATE --noise-update-scale VALUE --speech-recovery VALUE "
                   "--release-ms VALUE [--floor-db VALUE]\n";
      std::exit(0);
    } else if (arg == "--input-f32") {
      if (have_input) usage_error("duplicate --input-f32");
      options.input = take_value(i, argc, argv); have_input = true;
    } else if (arg == "--output-f32") {
      if (have_output) usage_error("duplicate --output-f32");
      options.output = take_value(i, argc, argv); have_output = true;
    } else if (arg == "--rate") {
      if (have_rate) usage_error("duplicate --rate");
      options.rate = parse_unsigned(take_value(i, argc, argv), "--rate"); have_rate = true;
    } else if (arg == "--noise-update-scale") {
      if (have_noise) usage_error("duplicate --noise-update-scale");
      options.noise_update_scale = parse_float(take_value(i, argc, argv), "--noise-update-scale");
      have_noise = true;
    } else if (arg == "--speech-recovery") {
      if (have_speech) usage_error("duplicate --speech-recovery");
      options.speech_recovery = parse_float(take_value(i, argc, argv), "--speech-recovery");
      have_speech = true;
    } else if (arg == "--release-ms") {
      if (have_release) usage_error("duplicate --release-ms");
      options.release_ms = parse_float(take_value(i, argc, argv), "--release-ms");
      have_release = true;
    } else if (arg == "--floor-db") {
      if (have_floor) usage_error("duplicate --floor-db");
      options.floor_db = parse_float(take_value(i, argc, argv), "--floor-db");
      have_floor = true;
    } else {
      usage_error("unknown argument: " + arg);
    }
  }
  if (!have_input || !have_output || !have_rate || !have_noise || !have_speech || !have_release)
    usage_error("all required arguments must be supplied exactly once");
  if (options.rate != 16000 && options.rate != 24000 && options.rate != 48000)
    usage_error("--rate must be 16000, 24000, or 48000");
  if (options.noise_update_scale < .1f || options.noise_update_scale > 1.f)
    usage_error("--noise-update-scale must be in [0.1, 1]");
  if (options.speech_recovery < 0.f || options.speech_recovery > 1.f)
    usage_error("--speech-recovery must be in [0, 1]");
  if (options.release_ms < 0.f || options.release_ms > 200.f)
    usage_error("--release-ms must be in [0, 200]");
  if (options.floor_db < -40.f || options.floor_db > 0.f)
    usage_error("--floor-db must be in [-40, 0]");
  return options;
}

float read_float32_le(const unsigned char* bytes) {
  const std::uint32_t bits = std::uint32_t(bytes[0]) |
      (std::uint32_t(bytes[1]) << 8) |
      (std::uint32_t(bytes[2]) << 16) |
      (std::uint32_t(bytes[3]) << 24);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

void append_float32_le(std::vector<unsigned char>& output, float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  output.push_back(static_cast<unsigned char>(bits));
  output.push_back(static_cast<unsigned char>(bits >> 8));
  output.push_back(static_cast<unsigned char>(bits >> 16));
  output.push_back(static_cast<unsigned char>(bits >> 24));
}

std::vector<unsigned char> read_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("cannot open input: " + path.string());
  const std::streamoff end = stream.tellg();
  if (end < 0) throw std::runtime_error("cannot determine input size: " + path.string());
  if (static_cast<std::uintmax_t>(end) > std::numeric_limits<std::size_t>::max())
    throw std::runtime_error("input is too large");
  std::vector<unsigned char> bytes(static_cast<std::size_t>(end));
  stream.seekg(0);
  if (!bytes.empty() && !stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size()))
    throw std::runtime_error("cannot read input: " + path.string());
  return bytes;
}

void write_file(const std::filesystem::path& path, const std::vector<unsigned char>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("cannot open output: " + path.string());
  if (!bytes.empty()) stream.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  stream.close();
  if (!stream) throw std::runtime_error("cannot write output: " + path.string());
}

int run(const Options& options) {
  const std::vector<unsigned char> input_bytes = read_file(options.input);
  if (input_bytes.size() % sizeof(float) != 0)
    usage_error("input byte length must be a multiple of four for float32LE");
  const std::size_t input_samples = input_bytes.size() / sizeof(float);
  const unsigned hop = options.rate / 100;
  if (input_samples % hop != 0)
    usage_error("input sample length must be an integer multiple of one hop");

  std::vector<float> input(input_samples);
  for (std::size_t i = 0; i < input_samples; ++i)
    input[i] = read_float32_le(input_bytes.data() + i * sizeof(float));
  if (!std::all_of(input.begin(), input.end(), [](float x) { return std::isfinite(x); }))
    usage_error("input contains a non-finite float");

#ifdef ITU_OMLSA_ORIGINAL_BUILD
  itu::detail::OmlsaImcra engine(options.rate, options.floor_db, true,
                                 itu::detail::SpectralGainRule::omlsa);
#else
  itu::OmlsaTuningConfig tuning;
  tuning.noise_update_scale = options.noise_update_scale;
  tuning.speech_recovery = options.speech_recovery;
  tuning.release_ms = options.release_ms;
  itu::detail::OmlsaImcra engine(options.rate, options.floor_db, true,
                                 itu::detail::SpectralGainRule::omlsa, tuning);
#endif

  std::vector<float> output;
  output.reserve(input_samples + hop);
  std::vector<float> frame(hop);
  double probability_sum = 0;
  std::size_t probability_frames = 0;
  auto process_frame = [&](const float* samples) {
    std::copy(samples, samples + hop, frame.begin());
    engine.process(frame.data());
    if (!std::all_of(frame.begin(), frame.end(), [](float x) { return std::isfinite(x); }))
      throw std::runtime_error("OM-LSA produced a non-finite output sample");
    output.insert(output.end(), frame.begin(), frame.end());
    probability_sum += engine.mean_probability();
    ++probability_frames;
  };

  for (std::size_t offset = 0; offset < input_samples; offset += hop)
    process_frame(input.data() + offset);
  std::fill(frame.begin(), frame.end(), 0.f);
  process_frame(frame.data());
  if (output.size() != input_samples + hop)
    throw std::runtime_error("internal output-length invariant failed");

  std::vector<unsigned char> output_bytes;
  output_bytes.reserve(output.size() * sizeof(float));
  float peak = 0;
  for (float sample : output) {
    peak = std::max(peak, std::abs(sample));
    append_float32_le(output_bytes, sample);
  }
  write_file(options.output, output_bytes);

  const double mean_probability = probability_frames ?
      probability_sum / static_cast<double>(probability_frames) : 0.;
  std::cout.setf(std::ios::fixed);
  std::cout.precision(9);
  std::cout << "{\"schema\":\"itu_omlsa_probe.v1\","
            << "\"delay\":" << engine.delay() << ','
            << "\"rate\":" << options.rate << ','
            << "\"input_samples\":" << input_samples << ','
            << "\"output_samples\":" << output.size() << ','
            << "\"meanprob\":" << mean_probability << ','
            << "\"peak\":" << peak << ','
            << "\"floor_db\":" << options.floor_db << ','
            << "\"noise_update_scale\":" << options.noise_update_scale << ','
            << "\"speech_recovery\":" << options.speech_recovery << ','
            << "\"release_ms\":" << options.release_ms << ','
            << "\"drain_samples\":" << hop << "}\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_options(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "probe_omlsa_tuning: " << error.what() << '\n';
    return 2;
  }
}
