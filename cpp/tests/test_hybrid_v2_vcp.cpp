#include "itu/audio_engine.hpp"
#include "itu/vehicle_ns.hpp"
#include "residual_ns_control_v2.hpp"
#include "tiny_ns_residual_v2.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {
using itu::detail::NsSpectralFrame;
using itu::detail::ResidualNsControlV2;
using itu::detail::TinyNsResidualV2;

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

template <typename Callback>
void require_throw(Callback&& callback, const char* label) {
  bool rejected = false;
  try {
    callback();
  } catch (const std::exception&) {
    rejected = true;
  }
  require(rejected, label);
}

void put_u32(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint32_t value) {
  for (unsigned i = 0; i < 4; ++i) bytes[offset + i] = value >> (8 * i);
}

void put_u64(std::vector<unsigned char>& bytes, std::size_t offset,
             std::uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) bytes[offset + i] = value >> (8 * i);
}

void put_u16(std::ofstream& file, std::uint16_t value) {
  const char bytes[2] = {char(value), char(value >> 8)};
  file.write(bytes, 2);
}

void put_u32(std::ofstream& file, std::uint32_t value) {
  const char bytes[4] = {char(value), char(value >> 8), char(value >> 16),
                         char(value >> 24)};
  file.write(bytes, 4);
}

void write_fixture_wav(const std::filesystem::path& path, unsigned rate = 16000) {
  const unsigned samples = rate;
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write CLI fixture WAV");
  file.write("RIFF", 4);
  put_u32(file, 36 + samples * 2);
  file.write("WAVEfmt ", 8);
  put_u32(file, 16);
  put_u16(file, 1);
  put_u16(file, 1);
  put_u32(file, rate);
  put_u32(file, rate * 2);
  put_u16(file, 2);
  put_u16(file, 16);
  file.write("data", 4);
  put_u32(file, samples * 2);
  for (unsigned i = 0; i < samples; ++i) {
    const float value = .03f * std::sin(.07f * float(i)) +
                        .006f * std::sin(.31f * float(i));
    const std::int16_t pcm = static_cast<std::int16_t>(std::lround(value * 32767.f));
    put_u16(file, static_cast<std::uint16_t>(pcm));
  }
  if (!file) throw std::runtime_error("cannot finish CLI fixture WAV");
}

std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read file: " + path.string());
  return std::vector<char>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

std::string read_text(const std::filesystem::path& path) {
  const auto bytes = read_bytes(path);
  return std::string(bytes.begin(), bytes.end());
}

std::uint64_t file_hash(const std::filesystem::path& path) {
  const auto bytes = read_bytes(path);
  return TinyNsResidualV2::fnv1a64(
      reinterpret_cast<const unsigned char*>(bytes.data()), bytes.size());
}

bool contains(const std::string& text, const std::string& needle) {
  return text.find(needle) != std::string::npos;
}

std::string line_with(const std::string& text, const std::string& needle) {
  const auto position = text.find(needle);
  if (position == std::string::npos) return {};
  const auto begin = text.rfind('\n', position);
  const auto end = text.find('\n', position);
  return text.substr(begin == std::string::npos ? 0 : begin + 1,
                     end == std::string::npos ? std::string::npos : end - begin - 1);
}

int run_cli(const std::filesystem::path& cli, std::vector<std::string> args) {
  std::vector<std::string> owned;
  owned.push_back(cli.string());
  owned.insert(owned.end(), std::move(args.begin()), std::move(args.end()));
  std::vector<char*> argv;
  for (auto& value : owned) argv.push_back(value.data());
  argv.push_back(nullptr);
  const pid_t child = ::fork();
  if (child < 0) throw std::runtime_error("cannot fork itu_audio_cli");
  if (child == 0) {
    ::execv(cli.c_str(), argv.data());
    ::_exit(127);
  }
  int status = 0;
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status)) return 255;
  return WEXITSTATUS(status);
}

void run_cli_ok(const std::filesystem::path& cli, std::vector<std::string> args) {
  require(run_cli(cli, std::move(args)) == 0, "itu_audio_cli failed");
}

void run_cli_rejected(const std::filesystem::path& cli,
                      std::vector<std::string> args) {
  require(run_cli(cli, std::move(args)) != 0,
          "itu_audio_cli accepted a forbidden hybrid V2 option");
}

std::filesystem::path write_zero_model(const std::filesystem::path& directory,
                                       unsigned input_dim) {
  const std::size_t count = TinyNsResidualV2::parameter_count_for(input_dim);
  const std::size_t size = TinyNsResidualV2::file_size_for(input_dim);
  require(count != 0 && size != 0, "invalid V2 fixture input dimension");
  const auto path = directory / ("zero-" + std::to_string(input_dim) + ".bin");
  std::vector<unsigned char> bytes(size, 0);
  std::memcpy(bytes.data(), "ITUNSR02", 8);
  const std::array<std::uint32_t, 14> metadata = {
      2, 16000, 64, input_dim, 64, 64, 64, static_cast<std::uint32_t>(count),
      1, input_dim == 256 ? 1u : 2u, 512, 160, 320, 0};
  for (std::size_t i = 0; i < metadata.size(); ++i)
    put_u32(bytes, 8 + i * 4, metadata[i]);
  const std::uint64_t hash = TinyNsResidualV2::fnv1a64(
      bytes.data(), bytes.size() - TinyNsResidualV2::checksum_size);
  put_u64(bytes, bytes.size() - TinyNsResidualV2::checksum_size, hash);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write V2 fixture");
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) throw std::runtime_error("cannot finish V2 fixture");
  return path;
}

void profile_contract() {
  const auto c64 = itu::barrot_vcp_c64_config();
  const auto hybrid = itu::barrot_vcp_hybrid_v2_config();
  require(hybrid.sample_rate == 16000 && hybrid.frame_size == 160 &&
              hybrid.microphones == 1 && !hybrid.aec && !hybrid.beamforming,
          "vcp_hybrid_v2 is not native 16 kHz mono with AEC off");
  require(hybrid.noise_suppression &&
              hybrid.noise_backend == itu::NoiseBackend::omlsa_imcra &&
              hybrid.noise_level == 0 && hybrid.vehicle_ns_control &&
              hybrid.vehicle_ns.backend == itu::VehicleNsBackend::omlsa_imcra &&
              hybrid.vehicle_ns.strength == 0,
          "vcp_hybrid_v2 did not derive strength-0 native OM-LSA");
  require(hybrid.vehicle_ns.hybrid_v2_model_path.empty() &&
              hybrid.vehicle_ns.hybrid_v2_mix == 1.f &&
              hybrid.vehicle_ns.hybrid_v2_floor_db == -21.f &&
              hybrid.vehicle_ns.hybrid_v2_pitch_strength == 0.f,
          "vcp_hybrid_v2 V2 defaults/path are wrong");
  require(hybrid.vehicle_ns.dry_max == 0.f &&
              !hybrid.vehicle_ns.unified_raw_budget &&
              hybrid.vehicle_ns.repair_mode == 0 &&
              hybrid.vehicle_ns.spectral_dry_guard == 0.f &&
              hybrid.vehicle_ns.weak_speech_recovery == 0.f &&
              hybrid.vehicle_ns.speech_band_recovery == 0.f &&
              hybrid.vehicle_ns.noise_step_recovery == 0.f &&
              hybrid.vehicle_ns.transient_dry_guard == 0.f &&
              hybrid.vehicle_ns.transient_spectral_guard_db == 0.f &&
              hybrid.vehicle_ns.residual_max_db == 0.f,
          "vcp_hybrid_v2 retained a raw-mix or legacy guard");
  require(hybrid.vehicle_ns.student_model_path.empty() &&
              hybrid.vehicle_ns.residual_model_path.empty() &&
              !hybrid.mild_near_only_ns,
          "vcp_hybrid_v2 retained a legacy model or routing override");
  require(hybrid.vehicle_ns.omlsa_tuning.noise_update_scale == 1.f &&
              hybrid.vehicle_ns.omlsa_tuning.speech_recovery == 0.f &&
              hybrid.vehicle_ns.omlsa_tuning.release_ms == 60.f &&
              hybrid.vehicle_ns.omlsa_tuning.speech_floor_db == -40.f &&
              !hybrid.vehicle_ns.omlsa_tuning.adaptive_noise_tracking &&
              !hybrid.vehicle_ns.omlsa_tuning.high_snr_transparency &&
              !hybrid.vehicle_ns.omlsa_tuning.low_band_protection,
          "vcp_hybrid_v2 did not keep default OM-LSA tuning");
  require(c64.vehicle_ns.hybrid_v2_model_path.empty() &&
              c64.vehicle_ns.hybrid_v2_pitch_strength == 0.f,
          "vcp_c64 was changed by the V2 profile defaults");
  itu::VehicleNsProcessor processor(16000, hybrid.vehicle_ns);
  require(processor.delay_samples() == 160,
          "vcp_hybrid_v2 disabled V2 path changed the 160-sample delay");
}

void control_contract(const std::filesystem::path& model256,
                      const std::filesystem::path& model320) {
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
  power.fill(.02f);
  noise.fill(.01f);
  xi.fill(1.f);
  gamma.fill(2.f);
  probability.fill(.5f);
  gain.fill(.5f);
  const auto original_gain = gain;
  const NsSpectralFrame frame{16000, 512, 257, power.data(), noise.data(),
                              xi.data(), gamma.data(), probability.data(),
                              gain.data()};

  ResidualNsControlV2 disabled(16000, 512, "", -21.f, 1.f);
  disabled.observe(frame);
  require(!disabled.model_ready() && disabled.fallback_frames() == 1 &&
              disabled.inferences() == 0 && gain == original_gain,
          "disabled V2 controller did not preserve base gain");

  ResidualNsControlV2 mix0(16000, 512, model256.string(), -21.f, 0.f);
  mix0.observe(frame);
  require(mix0.model_ready() && mix0.inferences() == 1 &&
              mix0.fallback_frames() == 0 && mix0.applied_frames() == 0 &&
              !mix0.applied() && mix0.input_dim() == 256 &&
              mix0.parameter_count() == 45568 && gain == original_gain,
          "mix=0 V2 controller changed the original control gain");

  ResidualNsControlV2 model320_control(16000, 512, model320.string(), -21.f, 1.f);
  model320_control.observe(frame);
  require(model320_control.model_ready() && model320_control.inferences() == 1 &&
              model320_control.fallback_frames() == 0 &&
              model320_control.applied_frames() == 1 &&
              model320_control.applied() && model320_control.input_dim() == 320 &&
              model320_control.parameter_count() == 49664,
          "320-dimensional V2 model counts are wrong");

  gain = original_gain;
  power[17] = std::numeric_limits<float>::quiet_NaN();
  ResidualNsControlV2 finite_fallback(16000, 512, model256.string(), -21.f, 1.f);
  finite_fallback.observe(frame);
  require(finite_fallback.fallback_frames() == 1 &&
              finite_fallback.inferences() == 0 && gain == original_gain,
          "runtime non-finite V2 input did not keep the base gain");
}

void processor_contract(const std::filesystem::path& model256,
                        const std::filesystem::path& model320) {
  std::array<float, 160> input{};
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = .02f * std::sin(.09f * float(i)) +
               .005f * std::sin(.37f * float(i));

  auto config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = model256.string();
  config.hybrid_v2_mix = 0.f;
  itu::VehicleNsProcessor mix0(16000, config);
  require(mix0.process(input.data(), input.size()),
          "mix=0 V2 VehicleNsProcessor frame failed");
  auto stats = mix0.statistics();
  require(stats.hybrid_v2_state == 1 && stats.hybrid_v2_inferences == 1 &&
              stats.hybrid_v2_fallback_frames == 0 &&
              stats.hybrid_v2_applied_frames == 0 && !stats.hybrid_v2_applied &&
              stats.hybrid_v2_parameters == 45568 &&
              stats.hybrid_v2_input_dim == 256 && stats.hybrid_v2_raw_mix == 0.f,
          "mix=0 V2 processor telemetry is wrong");

  config.hybrid_v2_model_path = model320.string();
  config.hybrid_v2_mix = 1.f;
  itu::VehicleNsProcessor applied(16000, config);
  require(applied.process(input.data(), input.size()),
          "320-dimensional V2 VehicleNsProcessor frame failed");
  stats = applied.statistics();
  require(stats.hybrid_v2_state == 1 && stats.hybrid_v2_inferences == 1 &&
              stats.hybrid_v2_fallback_frames == 0 &&
              stats.hybrid_v2_applied_frames == 1 && stats.hybrid_v2_applied &&
              stats.hybrid_v2_parameters == 49664 &&
              stats.hybrid_v2_input_dim == 320 && stats.hybrid_v2_raw_mix == 0.f,
          "320-dimensional V2 processor telemetry is wrong");
}

void validation_contract(const std::filesystem::path& model,
                         const std::filesystem::path& missing) {
  auto config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(24000, config); },
                "hybrid V2 accepted non-16-kHz processing");

  config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = model.string();
  config.backend = itu::VehicleNsBackend::webrtc;
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "hybrid V2 accepted a non-OM-LSA backend");

  config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = model.string();
  config.student_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "hybrid V2 and legacy student were both accepted");

  config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = model.string();
  config.residual_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "hybrid V2 and legacy residual model were both accepted");

  for (float mix : {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN()}) {
    config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
    config.hybrid_v2_mix = mix;
    require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                  "invalid hybrid V2 mix was accepted");
  }
  for (float floor : {-40.01f, 0.01f, std::numeric_limits<float>::quiet_NaN()}) {
    config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
    config.hybrid_v2_floor_db = floor;
    require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                  "invalid hybrid V2 floor was accepted");
  }
  for (float strength : {-0.01f, 1.01f, std::numeric_limits<float>::quiet_NaN()}) {
    config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
    config.hybrid_v2_pitch_strength = strength;
    require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                  "invalid hybrid V2 pitch strength was accepted");
  }

  config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_model_path = missing.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "missing V2 model did not fail construction");

  auto engine_config = itu::barrot_vcp_hybrid_v2_config();
  engine_config.vehicle_ns.hybrid_v2_model_path = model.string();
  engine_config.vehicle_ns.student_model_path = model.string();
  require_throw([&] { itu::AudioEngine invalid(engine_config); },
                "AudioEngine accepted hybrid V2 plus legacy model");

  engine_config = itu::barrot_vcp_hybrid_v2_config();
  engine_config.vehicle_ns.hybrid_v2_model_path = model.string();
  engine_config.sample_rate = 24000;
  engine_config.frame_size = 240;
  require_throw([&] { itu::AudioEngine invalid(engine_config); },
                "AudioEngine accepted hybrid V2 at 24 kHz");
}

void pitch_contract() {
  auto config = itu::barrot_vcp_hybrid_v2_config().vehicle_ns;
  config.hybrid_v2_pitch_strength = 1.f;
  itu::VehicleNsProcessor processor(16000, config);
  std::array<float, 160> frame{};
  for (unsigned block = 0; block < 40; ++block) {
    for (unsigned i = 0; i < frame.size(); ++i) {
      const auto sample = block * frame.size() + i;
      const double phase = 6.283185307179586 * 220.0 * sample / 16000.0;
      frame[i] = .05f * float(std::sin(phase)) +
                 .025f * float(std::sin(2.0 * phase)) +
                 .002f * float(std::sin(.731 * sample));
    }
    require(processor.process(frame.data(), frame.size()),
            "pitch-only hybrid V2 frame failed");
  }
  const auto stats = processor.statistics();
  require(processor.delay_samples() == 160,
          "pitch-only hybrid V2 changed the 160-sample delay");
  require(stats.hybrid_v2_state == 2 && stats.hybrid_v2_inferences == 0 &&
              stats.hybrid_v2_fallback_frames == 0 &&
              stats.hybrid_v2_parameters == 0 && stats.hybrid_v2_input_dim == 0 &&
              stats.hybrid_v2_raw_mix == 0.f,
          "pitch-only DSP telemetry was attributed to the neural model");
  require(stats.hybrid_v2_pitch_lag_samples >= 40 &&
              stats.hybrid_v2_pitch_lag_samples <= 228 &&
              stats.hybrid_v2_pitch_periodicity >= .60f &&
              stats.hybrid_v2_pitch_active_frames > 0,
          "pitch-only DSP statistics did not report active/lag/periodicity");
}

void engine_contract(const std::filesystem::path& model256,
                     const std::filesystem::path& model320) {
  std::array<float, 160> input{};
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = .02f * std::sin(.09f * float(i)) +
               .005f * std::sin(.37f * float(i));
  std::array<float, 160> baseline{}, mix0{}, applied{};
  itu::FrameInfo info;

  auto config = itu::barrot_vcp_hybrid_v2_config();
  config.automatic_gain = false;
  config.limiter = false;
  config.tx_drc.enabled = false;
  itu::AudioEngine baseline_engine(config);
  require(baseline_engine.process(input.data(), input.size(), nullptr, 0,
                                  baseline.data(), baseline.size(), info) ==
              itu::Status::ok,
          "hybrid V2 disabled baseline failed");
  require(info.hybrid_v2_state == 0 && info.hybrid_v2_inferences == 0 &&
              info.hybrid_v2_parameters == 0 && info.ns_delay_samples == 160 &&
              info.latency_samples == 160,
          "hybrid V2 disabled telemetry/delay is wrong");

  auto mix0_config = config;
  mix0_config.vehicle_ns.hybrid_v2_model_path = model256.string();
  mix0_config.vehicle_ns.hybrid_v2_mix = 0.f;
  itu::AudioEngine mix0_engine(mix0_config);
  require(mix0_engine.process(input.data(), input.size(), nullptr, 0,
                              mix0.data(), mix0.size(), info) == itu::Status::ok,
          "hybrid V2 mix=0 engine failed");
  require(mix0 == baseline && info.hybrid_v2_state == 1 &&
              info.hybrid_v2_inferences == 1 &&
              info.hybrid_v2_fallback_frames == 0 &&
              info.hybrid_v2_applied_frames == 0 && !info.hybrid_v2_applied &&
              info.hybrid_v2_parameters == 45568 &&
              info.hybrid_v2_input_dim == 256 && info.hybrid_v2_raw_mix == 0.f &&
              info.ns_delay_samples == 160 && info.latency_samples == 160,
          "mix=0 engine changed PCM or telemetry");

  auto applied_config = config;
  applied_config.vehicle_ns.hybrid_v2_model_path = model256.string();
  itu::AudioEngine applied_engine(applied_config);
  require(applied_engine.process(input.data(), input.size(), nullptr, 0,
                                 applied.data(), applied.size(), info) ==
              itu::Status::ok,
          "hybrid V2 enabled engine failed");
  require(info.hybrid_v2_state == 1 && info.hybrid_v2_inferences == 1 &&
              info.hybrid_v2_fallback_frames == 0 &&
              info.hybrid_v2_applied_frames == 1 && info.hybrid_v2_applied &&
              info.hybrid_v2_parameters == 45568 &&
              info.hybrid_v2_input_dim == 256 && info.latency_samples == 160,
          "enabled hybrid V2 telemetry is wrong");

  auto model320_config = config;
  model320_config.vehicle_ns.hybrid_v2_model_path = model320.string();
  itu::AudioEngine model320_engine(model320_config);
  require(model320_engine.process(input.data(), input.size(), nullptr, 0,
                                  applied.data(), applied.size(), info) ==
              itu::Status::ok,
          "320-dimensional hybrid V2 engine failed");
  require(info.hybrid_v2_parameters == 49664 && info.hybrid_v2_input_dim == 320 &&
              info.hybrid_v2_fallback_frames == 0,
          "320-dimensional hybrid V2 engine telemetry is wrong");
}

void cli_contract(const std::filesystem::path& cli,
                  const std::filesystem::path& directory,
                  const std::filesystem::path& model256,
                  const std::filesystem::path& model320,
                  const std::filesystem::path& missing) {
  const auto wav = directory / "fixture-16k.wav";
  const auto wav24 = directory / "fixture-24k.wav";
  write_fixture_wav(wav, 16000);
  write_fixture_wav(wav24, 24000);
  const std::vector<std::string> common = {
      "--profile", "vcp_hybrid_v2", "--mic", wav.string(),
      "--no-agc", "--no-tx-drc", "--diagnostic-stages"};

  auto run = [&](const std::string& name,
                 std::vector<std::string> extra) {
    auto args = common;
    args.insert(args.end(), extra.begin(), extra.end());
    args.insert(args.end(), {"--out", (directory / name).string()});
    run_cli_ok(cli, std::move(args));
    return directory / name;
  };

  const auto base = run("base", {});
  const auto mix0 = run("mix0", {"--hybrid-v2-model", model256.string(),
                                  "--hybrid-v2-mix", "0"});
  const auto mix1 = run("mix1", {"--hybrid-v2-model", model256.string(),
                                  "--hybrid-v2-mix", "1"});
  const auto model320_run = run("model320", {"--hybrid-v2-model", model320.string()});
  const auto pitch0 = run("pitch0", {"--hybrid-v2-pitch-strength", "0"});
  const auto pitch1 = run("pitch1", {"--hybrid-v2-pitch-strength", "1"});

  const auto base_hash = file_hash(base.string() + ".tx_ns_input.float.wav");
  for (const auto* prefix : {&mix0, &mix1, &model320_run, &pitch0, &pitch1})
    require(file_hash(prefix->string() + ".tx_ns_input.float.wav") == base_hash,
            "actual tx_ns_input hash changed with V2 model or pitch processing");
  require(read_bytes(base.string() + ".after.wav") ==
              read_bytes(mix0.string() + ".after.wav"),
          "mix=0 V2 inference changed the final PCM");
  require(read_bytes(base.string() + ".after.wav") ==
              read_bytes(pitch0.string() + ".after.wav"),
          "pitch strength 0 changed the final PCM");

  const auto metadata = read_text(pitch1.string() + ".stages.json");
  const auto ns_input = line_with(metadata, "\"id\":\"tx_ns_input\"");
  const auto output = line_with(metadata, "\"id\":\"tx_output\"");
  require(contains(ns_input, "\"samples_per_channel\":16000") &&
              contains(ns_input, "\"fixed_delay_samples\":0"),
          "pitch changed actual NS-input tap extent or delay metadata");
  require(contains(output, "\"samples_per_channel\":16160") &&
              contains(output, "\"fixed_delay_samples\":160"),
          "pitch changed output tap metadata beyond the 160-sample NS delay");

  const std::vector<std::pair<std::string, std::vector<std::string>>> rejected = {
      {"legacy-student", {"--hybrid-ns-model", model256.string()}},
      {"legacy-residual", {"--residual-ns-model", model256.string()}},
      {"legacy-mix", {"--residual-ns-mix", "0.5"}},
      {"legacy-policy", {"--vehicle-ns-policy", "legacy"}},
      {"invalid-pitch", {"--hybrid-v2-pitch-strength", "1.1"}},
      {"wrong-backend", {"--ns-backend", "webrtc"}},
      {"no-ns", {"--no-ns"}},
  };
  for (const auto& [name, extra] : rejected) {
    auto args = common;
    args.insert(args.end(), extra.begin(), extra.end());
    args.insert(args.end(), {"--out", (directory / ("reject-" + name)).string()});
    run_cli_rejected(cli, std::move(args));
  }

  auto wrong_rate = common;
  wrong_rate[3] = wav24.string();
  wrong_rate.insert(wrong_rate.end(), {"--out", (directory / "reject-24k").string()});
  run_cli_rejected(cli, std::move(wrong_rate));

  auto missing_model = common;
  missing_model.insert(missing_model.end(),
                       {"--hybrid-v2-model", missing.string(),
                        "--out", (directory / "reject-missing-model").string()});
  run_cli_rejected(cli, std::move(missing_model));

  std::cout << "tx_ns_input_fnv1a64=" << std::hex << base_hash << std::dec
            << " model256=45568/256 model320=49664/320\n";
}
}  // namespace

int main(int argc, char** argv) {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("itu-hybrid-v2-vcp-" + std::to_string(::getpid()));
  try {
    std::filesystem::create_directories(directory);
    const auto model256 = write_zero_model(directory, 256);
    const auto model320 = write_zero_model(directory, 320);
    const auto missing = directory / "missing.itunsr02.bin";
    const auto executable = std::filesystem::absolute(argv[0]);
    const auto cli = argc > 1 ? std::filesystem::path(argv[1])
                              : executable.parent_path() / "itu_audio_cli";
    require(std::filesystem::exists(cli), "itu_audio_cli is required");

    profile_contract();
    control_contract(model256, model320);
    processor_contract(model256, model320);
    validation_contract(model256, missing);
    pitch_contract();
    engine_contract(model256, model320);
    cli_contract(cli, directory, model256, model320, missing);
    std::filesystem::remove_all(directory);
    std::cout << "hybrid V2 VCP profile/validation/mix0/model/pitch/hash/taps checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
