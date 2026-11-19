#include "omlsa_imcra.hpp"
#include "residual_feature_bands.hpp"
#include "residual_ns_control.hpp"
#include "tiny_ns_residual.hpp"
#include "wav.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using itu::detail::OmlsaImcra;
using itu::detail::ResidualFeatureBands;
using itu::detail::ResidualNsControl;
using itu::detail::TinyNsResidual;

constexpr unsigned kRate = 16000;
constexpr unsigned kHop = 160;
constexpr unsigned kFft = 512;
constexpr unsigned kBins = 257;
constexpr unsigned kDrainHops = 12;

struct Options {
  std::map<std::string, std::string> values;

  bool has(const std::string& key) const { return values.contains(key); }
  const std::string& get(const std::string& key) const {
    return values.at(key);
  }
};

bool known_option(const std::string& key) {
  static const std::array<const char*, 9> keys = {
      "--mic", "--out", "--model", "--floor-db", "--mix",
      "--absolute-floor-db", "--features", "--gru-input", "--band-input"};
  return std::find_if(keys.begin(), keys.end(), [&](const char* candidate) {
           return key == candidate;
         }) != keys.end();
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; i += 2) {
    const std::string key = argv[i];
    if (!known_option(key)) throw std::runtime_error("unknown option: " + key);
    if (i + 1 >= argc) throw std::runtime_error("missing value for " + key);
    const std::string value = argv[i + 1];
    if (value.empty()) throw std::runtime_error("empty value for " + key);
    if (!options.values.emplace(key, value).second)
      throw std::runtime_error("duplicate option: " + key);
  }
  if (options.values.empty()) throw std::runtime_error("missing arguments");
  return options;
}

float parse_float(const Options& options, const std::string& key,
                  float fallback) {
  if (!options.has(key)) return fallback;
  const std::string& text = options.get(key);
  errno = 0;
  char* end = nullptr;
  const float value = std::strtof(text.c_str(), &end);
  if (errno == ERANGE || end != text.c_str() + text.size() || !std::isfinite(value))
    throw std::runtime_error("invalid finite number for " + key);
  return value;
}

void reject_mode_option(const Options& options, const std::string& key,
                        const std::string& mode) {
  if (options.has(key))
    throw std::runtime_error(key + " is not valid with " + mode);
}

std::vector<float> read_float_file(const std::string& path,
                                   std::size_t columns) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open float input: " + path);
  const auto bytes = file.tellg();
  const std::size_t row_bytes = columns * sizeof(float);
  if (bytes <= 0 || static_cast<std::uint64_t>(bytes) % row_bytes != 0)
    throw std::runtime_error("float input has wrong row width");
  std::vector<float> values(static_cast<std::size_t>(bytes) / sizeof(float));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), bytes);
  if (!file) throw std::runtime_error("float input read failed");
  for (float value : values)
    if (!std::isfinite(value))
      throw std::runtime_error("float input contains a non-finite value");
  return values;
}

void put_u16(std::ofstream& file, std::uint16_t value) {
  for (unsigned i = 0; i < 2; ++i) file.put(static_cast<char>(value >> (8 * i)));
}

void put_u32(std::ofstream& file, std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) file.put(static_cast<char>(value >> (8 * i)));
}

// Probe output deliberately stays float32 so it does not add a second PCM16
// quantization after the frozen teacher.  This is local to the probe; the
// shared WAV utility is unchanged.
void write_float_wav(const std::string& path, unsigned rate, unsigned channels,
                     const std::vector<float>& samples) {
  if (!channels || channels > 8 || samples.size() % channels != 0 ||
      samples.size() > (UINT32_MAX - 36) / sizeof(float))
    throw std::runtime_error("invalid FLOAT WAV output shape");
  const std::uint32_t data_bytes =
      static_cast<std::uint32_t>(samples.size() * sizeof(float));
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write FLOAT WAV: " + path);
  file.write("RIFF", 4);
  put_u32(file, 36 + data_bytes);
  file.write("WAVEfmt ", 8);
  put_u32(file, 16);
  put_u16(file, 3);  // WAVE_FORMAT_IEEE_FLOAT
  put_u16(file, static_cast<std::uint16_t>(channels));
  put_u32(file, rate);
  put_u32(file, rate * channels * sizeof(float));
  put_u16(file, static_cast<std::uint16_t>(channels * sizeof(float)));
  put_u16(file, 32);
  file.write("data", 4);
  put_u32(file, data_bytes);
  for (float sample : samples) {
    if (!std::isfinite(sample))
      throw std::runtime_error("non-finite FLOAT WAV output");
    put_u32(file, std::bit_cast<std::uint32_t>(sample));
  }
  if (!file) throw std::runtime_error("FLOAT WAV output write failed");
}

int run_gru(const Options& options) {
  if (!options.has("--gru-input") || !options.has("--model") ||
      !options.has("--out")) {
    throw std::runtime_error(
        "--gru-input mode requires --gru-input --model --out");
  }
  for (const char* key : {"--mic", "--floor-db", "--mix",
                          "--absolute-floor-db", "--features", "--band-input"})
    reject_mode_option(options, key, "--gru-input");

  TinyNsResidual model;
  model.load(options.get("--model"));
  const auto input = read_float_file(options.get("--gru-input"),
                                     TinyNsResidual::input_size);
  std::ofstream output(options.get("--out"), std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot open GRU parity output");
  std::array<float, TinyNsResidual::output_size + TinyNsResidual::hidden_size>
      row{};
  for (std::size_t offset = 0; offset < input.size();
       offset += TinyNsResidual::input_size) {
    if (!model.process(input.data() + offset, row.data(),
                       row.data() + TinyNsResidual::output_size))
      throw std::runtime_error("C64 GRU inference failed");
    output.write(reinterpret_cast<const char*>(row.data()),
                 row.size() * sizeof(float));
    if (!output) throw std::runtime_error("GRU parity output write failed");
  }
  return 0;
}

int run_band(const Options& options) {
  if (!options.has("--band-input") || !options.has("--out"))
    throw std::runtime_error("--band-input mode requires --band-input --out");
  for (const char* key : {"--mic", "--model", "--floor-db", "--mix",
                          "--absolute-floor-db", "--features", "--gru-input"})
    reject_mode_option(options, key, "--band-input");

  constexpr std::size_t input_columns = 2 * kBins;
  constexpr std::size_t output_columns =
      ResidualFeatureBands::input_size + 2 * ResidualFeatureBands::bands;
  const auto input = read_float_file(options.get("--band-input"), input_columns);
  std::ofstream output(options.get("--out"), std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("cannot open band parity output");
  ResidualFeatureBands bands;
  std::array<float, output_columns> row{};
  for (std::size_t offset = 0; offset < input.size(); offset += input_columns) {
    if (!bands.process(input.data() + offset, input.data() + offset + kBins))
      throw std::runtime_error("band feature extraction failed");
    std::copy(bands.features().begin(), bands.features().end(), row.begin());
    std::copy(bands.projected_power().begin(), bands.projected_power().end(),
              row.begin() + ResidualFeatureBands::input_size);
    std::copy(bands.projected_noise().begin(), bands.projected_noise().end(),
              row.begin() + ResidualFeatureBands::input_size +
                  ResidualFeatureBands::bands);
    output.write(reinterpret_cast<const char*>(row.data()),
                 row.size() * sizeof(float));
    if (!output) throw std::runtime_error("band parity output write failed");
  }
  return 0;
}

double percentile(std::vector<double> values, double fraction) {
  std::sort(values.begin(), values.end());
  const std::size_t index = std::min(
      values.size() - 1,
      static_cast<std::size_t>(fraction * static_cast<double>(values.size() - 1)));
  return values[index];
}

int run_mic(const Options& options) {
  if (!options.has("--mic") || !options.has("--out"))
    throw std::runtime_error("--mic mode requires --mic --out");
  reject_mode_option(options, "--gru-input", "--mic");
  reject_mode_option(options, "--band-input", "--mic");

  const float base_floor_db = parse_float(options, "--floor-db", -9.f);
  const float mix = parse_float(options, "--mix", 1.f);
  const float final_floor_db =
      parse_float(options, "--absolute-floor-db", -9.f);
  const std::string model_path = options.has("--model")
      ? options.get("--model")
      : std::string{};
  const auto input = itu::read_wav(options.get("--mic"));
  if (input.sample_rate != kRate || input.channels != 1 || input.samples.empty())
    throw std::runtime_error("probe requires nonempty mono 16 kHz WAV");

  ResidualNsControl control(kRate, kFft, model_path, final_floor_db, mix);
  OmlsaImcra engine(kRate, base_floor_db);
  if (engine.delay() != kHop || engine.fft_size() != kFft)
    throw std::runtime_error("OM-LSA frontend contract changed");

  const std::size_t sample_count = input.samples.size();
  const std::size_t input_frames = (sample_count + kHop - 1) / kHop;
  std::vector<float> rendered((input_frames + kDrainHops) * kHop);
  std::vector<double> process_ms;
  process_ms.reserve(input_frames);
  std::array<float, kHop> frame{};
  std::ofstream features;
  if (options.has("--features")) {
    features.open(options.get("--features"),
                  std::ios::binary | std::ios::trunc);
    if (!features) throw std::runtime_error("cannot open feature output");
  }
  std::ofstream csv(options.get("--out") + ".csv", std::ios::trunc);
  if (!csv) throw std::runtime_error("cannot open probe CSV");
  csv << "frame,time_s,process_ms,inferences,applied_frames,fallback_frames\n";

  constexpr std::size_t feature_columns =
      ResidualFeatureBands::input_size + 2 * ResidualFeatureBands::bands + kBins;
  std::array<float, feature_columns> feature_row{};
  const auto wall_start = std::chrono::steady_clock::now();
  for (std::size_t frame_index = 0; frame_index < input_frames; ++frame_index) {
    for (unsigned i = 0; i < kHop; ++i) {
      const std::size_t index = frame_index * kHop + i;
      frame[i] = index < sample_count
          ? static_cast<float>(std::lround(std::clamp(
                input.samples[index] * 32768.f, -32768.f, 32767.f))) /
                32768.f
          : 0.f;
    }
    const auto start = std::chrono::steady_clock::now();
    engine.process(frame.data(), &control);
    const double milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    process_ms.push_back(milliseconds);
    std::copy(frame.begin(), frame.end(),
              rendered.begin() + frame_index * kHop);
    if (features.is_open()) {
      std::copy(control.features().begin(), control.features().end(),
                feature_row.begin());
      std::copy(control.projected_power().begin(), control.projected_power().end(),
                feature_row.begin() + ResidualFeatureBands::input_size);
      std::copy(control.projected_noise().begin(), control.projected_noise().end(),
                feature_row.begin() + ResidualFeatureBands::input_size +
                    ResidualFeatureBands::bands);
      std::copy(control.base_gain().begin(), control.base_gain().end(),
                feature_row.begin() + ResidualFeatureBands::input_size +
                    2 * ResidualFeatureBands::bands);
      features.write(reinterpret_cast<const char*>(feature_row.data()),
                     feature_row.size() * sizeof(float));
      if (!features) throw std::runtime_error("feature output write failed");
    }
    csv << frame_index << ',' << std::fixed << std::setprecision(6)
        << static_cast<double>(frame_index) * kHop / kRate << ','
        << milliseconds << ',' << control.inferences() << ','
        << control.applied_frames() << ',' << control.fallback_frames() << '\n';
    if (!csv) throw std::runtime_error("probe CSV write failed");
  }
  const std::size_t reported_inferences = control.inferences();
  const std::size_t reported_applied = control.applied_frames();
  const std::size_t reported_fallback = control.fallback_frames();

  frame.fill(0.f);
  for (unsigned hop = 0; hop < kDrainHops; ++hop) {
    engine.process(frame.data(), &control);
    std::copy(frame.begin(), frame.end(),
              rendered.begin() + (input_frames + hop) * kHop);
  }
  const double runtime_seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wall_start).count();
  rendered.resize(sample_count + kHop);
  write_float_wav(options.get("--out") + ".after.wav", kRate, 1, rendered);

  const double mean_ms =
      std::accumulate(process_ms.begin(), process_ms.end(), 0.) / process_ms.size();
  const double p50_ms = percentile(process_ms, .50);
  const double p95_ms = percentile(process_ms, .95);
  const double p99_ms = percentile(process_ms, .99);
  const double max_ms = *std::max_element(process_ms.begin(), process_ms.end());
  std::ofstream json(options.get("--out") + ".json", std::ios::trunc);
  if (!json) throw std::runtime_error("cannot open probe JSON");
  json << std::fixed << std::setprecision(6)
       << "{\"modelstate\":\"" << (control.model_ready() ? "ready" : "absent")
       << "\",\"inferences\":" << reported_inferences
       << ",\"applied\":" << reported_applied
       << ",\"fallback\":" << reported_fallback
       << ",\"rate\":" << kRate
       << ",\"delay\":" << engine.delay()
       << ",\"base_floor_db\":" << base_floor_db
       << ",\"final_floor_db\":" << final_floor_db
       << ",\"mix\":" << mix
       << ",\"output_subtype\":\"FLOAT\""
       << ",\"runtime\":" << runtime_seconds
       << ",\"frames\":" << input_frames
       << ",\"drain_hops\":" << kDrainHops
       << ",\"mean_process_ms\":" << mean_ms
       << ",\"p50_process_ms\":" << p50_ms
       << ",\"p95_process_ms\":" << p95_ms
       << ",\"p99_process_ms\":" << p99_ms
       << ",\"max_process_ms\":" << max_ms << "}\n";
  if (!json) throw std::runtime_error("probe JSON write failed");
  std::cout << "probe modelstate=" << (control.model_ready() ? "ready" : "absent")
            << " frames=" << input_frames << " inferences=" << reported_inferences
            << " applied=" << reported_applied
            << " fallback=" << reported_fallback
            << " delay=" << engine.delay()
            << " base_floor_db=" << base_floor_db
            << " final_floor_db=" << final_floor_db
            << " output_subtype=FLOAT output_samples=" << rendered.size() << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    const unsigned modes = options.has("--mic") + options.has("--gru-input") +
                           options.has("--band-input");
    if (modes != 1)
      throw std::runtime_error(
          "select exactly one of --mic, --gru-input, or --band-input");
    if (options.has("--gru-input")) return run_gru(options);
    if (options.has("--band-input")) return run_band(options);
    return run_mic(options);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
