#include "itu/audio_engine.hpp"
#include "itu/vehicle_ns.hpp"
#include "residual_ns_control.hpp"
#include "tiny_ns_residual.hpp"

#include <algorithm>
#include <array>
#include <bit>
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
using itu::detail::ResidualNsControl;
using itu::detail::TinyNsResidual;

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

void write_fixture_wav(const std::filesystem::path& path) {
  constexpr unsigned rate = 16000;
  constexpr unsigned samples = 3200;
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
    const float value = .03f * std::sin(.07f * float(i));
    const std::int16_t pcm = static_cast<std::int16_t>(std::lround(value * 32767.f));
    put_u16(file, static_cast<std::uint16_t>(pcm));
  }
  if (!file) throw std::runtime_error("cannot finish CLI fixture WAV");
}

std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot read stage export: " + path.string());
  return std::vector<char>(std::istreambuf_iterator<char>(file),
                           std::istreambuf_iterator<char>());
}

void run_cli(const std::filesystem::path& cli, std::vector<std::string> args) {
  std::vector<std::string> owned;
  owned.push_back(cli.string());
  owned.insert(owned.end(), args.begin(), args.end());
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
  if (::waitpid(child, &status, 0) != child || !WIFEXITED(status) ||
      WEXITSTATUS(status) != 0)
    throw std::runtime_error("itu_audio_cli failed for residual VCP stage export");
}

std::filesystem::path write_zero_model(const std::filesystem::path& directory) {
  const auto path = directory / ("c64-zero-" + std::to_string(::getpid()) + ".bin");
  std::vector<unsigned char> bytes(TinyNsResidual::file_size, 0);
  std::memcpy(bytes.data(), "ITUNSR01", 8);
  const std::array<std::uint32_t, 14> metadata = {
      1, 16000, 64, 256, 64, 64, 64, 45568, 1, 1, 512, 160, 320, 0};
  for (std::size_t i = 0; i < metadata.size(); ++i)
    put_u32(bytes, 8 + i * 4, metadata[i]);
  const std::uint64_t hash = TinyNsResidual::fnv1a64(
      bytes.data(), bytes.size() - TinyNsResidual::checksum_size);
  put_u64(bytes, bytes.size() - TinyNsResidual::checksum_size, hash);
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  if (!file) throw std::runtime_error("cannot write residual fixture");
  file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  if (!file) throw std::runtime_error("cannot finish residual fixture");
  return path;
}

void profile_contract() {
  const auto c64 = itu::barrot_vcp_c64_config();
  const auto adaptive = itu::barrot_vcp_adaptive_config();
  require(c64.sample_rate == 16000 && c64.frame_size == 160 &&
              c64.microphones == 1 && !c64.aec && !c64.beamforming,
          "vcp_c64 is not native 16 kHz mono with AEC off");
  require(c64.noise_suppression &&
              c64.noise_backend == itu::NoiseBackend::omlsa_imcra &&
              c64.noise_level == 0 && c64.vehicle_ns_control,
          "vcp_c64 did not select native strength-0 OM-LSA");
  require(c64.vehicle_ns.backend == itu::VehicleNsBackend::omlsa_imcra &&
              c64.vehicle_ns.strength == 0 &&
              c64.vehicle_ns.policy == "aligned",
          "vcp_c64 vehicle backend/strength/policy differs");
  require(!c64.vehicle_ns.unified_raw_budget &&
              c64.vehicle_ns.dry_max == 0.f &&
              c64.vehicle_ns.startup_ms == 0 &&
              c64.vehicle_ns.quiet_floor_db == -300.f &&
              c64.vehicle_ns.noise_gain == 1.f,
          "vcp_c64 still has a raw-mix budget");
  require(c64.vehicle_ns.repair_mode == 0 &&
              c64.vehicle_ns.rumble_hz == 0.f &&
              !c64.vehicle_ns.adaptive_rumble &&
              c64.vehicle_ns.rumble_recovery == 0.f &&
              c64.vehicle_ns.rumble_adaptive_hz == 0.f &&
              c64.vehicle_ns.spectral_dry_guard == 0.f,
          "vcp_c64 legacy repair/rumble/spectral guard is enabled");
  require(c64.vehicle_ns.weak_speech_recovery == 0.f &&
              c64.vehicle_ns.speech_band_recovery == 0.f &&
              c64.vehicle_ns.noise_step_recovery == 0.f &&
              c64.vehicle_ns.transient_dry_guard == 0.f &&
              c64.vehicle_ns.transient_spectral_guard_db == 0.f &&
              c64.vehicle_ns.residual_max_db == 0.f,
          "vcp_c64 recovery or legacy residual stage is enabled");
  const auto& tuning = c64.vehicle_ns.omlsa_tuning;
  require(tuning.noise_update_scale == 1.f && tuning.speech_recovery == 0.f &&
              tuning.release_ms == 60.f && tuning.speech_floor_db == -40.f &&
              !tuning.adaptive_noise_tracking &&
              !tuning.high_snr_transparency && !tuning.low_band_protection,
          "vcp_c64 did not reset to the original OM-LSA tuning");
  require(!c64.mild_near_only_ns,
          "vcp_c64 retained legacy mild-near-only routing");
  require(c64.vehicle_ns.residual_model_path.empty() &&
              c64.vehicle_ns.residual_model_mix == 1.f &&
              c64.vehicle_ns.residual_model_floor_db == -9.f,
          "vcp_c64 must not hardcode or enable a residual model");
  require(c64.eq_before_agc == adaptive.eq_before_agc &&
              c64.automatic_gain == adaptive.automatic_gain &&
              c64.limiter == adaptive.limiter &&
              c64.tx_drc.enabled == adaptive.tx_drc.enabled &&
              c64.tx_drc.threshold_dbfs == adaptive.tx_drc.threshold_dbfs &&
              c64.tx_drc.ratio == adaptive.tx_drc.ratio &&
              c64.bounded_speech_gain == adaptive.bounded_speech_gain,
          "vcp_c64 changed the upstream/downstream VCP frontend contract");

  itu::VehicleNsProcessor processor(16000, c64.vehicle_ns);
  require(processor.delay_samples() == 160,
          "vcp_c64 VehicleNs delay is not 160 samples");
  std::array<float, 160> frame{};
  for (unsigned i = 0; i < frame.size(); ++i)
    frame[i] = .01f * std::sin(.07f * float(i));
  require(processor.process(frame.data(), frame.size()),
          "vcp_c64 disabled-model frame failed");
  const auto stats = processor.statistics();
  require(stats.residual_model_state == 0 &&
              stats.residual_model_inferences == 0 &&
              stats.residual_model_fallback_frames == 0 &&
              stats.residual_model_applied_frames == 0 &&
              stats.residual_model_parameters == 0,
          "disabled residual model emitted active statistics");
}

void validation_contract(const std::filesystem::path& model) {
  auto config = itu::barrot_vcp_c64_config().vehicle_ns;
  config.residual_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(24000, config); },
                "residual model accepted a non-16-kHz stream");

  config = itu::barrot_vcp_c64_config().vehicle_ns;
  config.backend = itu::VehicleNsBackend::webrtc;
  config.residual_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "residual model accepted a non-OM-LSA backend");

  config = itu::barrot_vcp_c64_config().vehicle_ns;
  config.residual_model_path = model.string();
  config.student_model_path = model.string();
  require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                "residual and legacy student models were both accepted");

  for (float mix : {-0.01f, 1.01f,
                    std::numeric_limits<float>::quiet_NaN()}) {
    config = itu::barrot_vcp_c64_config().vehicle_ns;
    config.residual_model_mix = mix;
    require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                  "invalid residual mix was accepted");
  }
  for (float floor : {-40.01f, 0.01f,
                      std::numeric_limits<float>::quiet_NaN()}) {
    config = itu::barrot_vcp_c64_config().vehicle_ns;
    config.residual_model_floor_db = floor;
    require_throw([&] { itu::VehicleNsProcessor invalid(16000, config); },
                  "invalid residual floor was accepted");
  }
}

void fallback_contract() {
  ResidualNsControl control(16000, 512, "", -9.f, 1.f);
  std::array<float, 257> power{}, noise{}, xi{}, gamma{}, probability{}, gain{};
  power.fill(1.f);
  noise.fill(1.f);
  xi.fill(1.f);
  gamma.fill(1.f);
  gain.fill(1.f);
  const NsSpectralFrame frame{16000, 512, 257, power.data(), noise.data(),
                              xi.data(), gamma.data(), probability.data(),
                              gain.data()};
  control.observe(frame);
  require(!control.model_ready() && control.fallback_frames() == 1 &&
              control.inferences() == 0 && control.applied_frames() == 0 &&
              !control.applied(),
          "disabled/fallback residual control did not stay explicit");
  require(std::all_of(gain.begin(), gain.end(), [](float value) {
            return value == 1.f;
          }),
          "fallback residual control changed base gain");
}

void engine_telemetry(const std::filesystem::path& model) {
  std::vector<float> input(160), baseline(160), disabled_mix(160), applied(160);
  for (unsigned i = 0; i < input.size(); ++i)
    input[i] = .01f * std::sin(.07f * float(i));
  itu::FrameInfo info;

  auto no_model = itu::barrot_vcp_c64_config();
  no_model.automatic_gain = false;
  no_model.limiter = false;
  itu::AudioEngine baseline_engine(no_model);
  require(baseline_engine.process(input.data(), input.size(), nullptr, 0,
                                  baseline.data(), baseline.size(), info) ==
              itu::Status::ok,
          "vcp_c64 no-model engine failed");
  require(info.latency_samples == 160 && info.residual_model_state == 0 &&
              info.hybrid_ns_state == 0 &&
              info.residual_model_parameters == 0,
          "vcp_c64 no-model telemetry is not disabled");

  auto zero_mix = no_model;
  zero_mix.vehicle_ns.residual_model_path = model.string();
  zero_mix.vehicle_ns.residual_model_mix = 0.f;
  itu::AudioEngine zero_mix_engine(zero_mix);
  require(zero_mix_engine.process(input.data(), input.size(), nullptr, 0,
                                  disabled_mix.data(), disabled_mix.size(), info) ==
              itu::Status::ok,
          "vcp_c64 mix=0 engine failed");
  require(disabled_mix == baseline,
          "mix=0 residual inference changed frontend PCM");
  require(info.residual_model_state == 1 && info.residual_model_inferences == 1 &&
              info.residual_model_applied_frames == 0 &&
              info.residual_model_parameters == TinyNsResidual::parameter_count,
          "mix=0 residual inference telemetry differs");

  auto enabled = no_model;
  enabled.vehicle_ns.residual_model_path = model.string();
  itu::AudioEngine enabled_engine(enabled);
  require(enabled_engine.process(input.data(), input.size(), nullptr, 0,
                                 applied.data(), applied.size(), info) ==
              itu::Status::ok,
          "vcp_c64 enabled residual engine failed");
  require(info.residual_model_state == 1 && info.residual_model_inferences == 1 &&
              info.residual_model_applied_frames == 1 &&
              info.residual_model_parameters == TinyNsResidual::parameter_count,
          "enabled residual model telemetry differs");
}

void cli_preboundary_contract(const std::filesystem::path& cli,
                              const std::filesystem::path& directory,
                              const std::filesystem::path& model) {
  const auto wav = directory / "fixture.wav";
  const auto disabled = directory / "cli-disabled";
  const auto enabled = directory / "cli-enabled";
  write_fixture_wav(wav);
  const std::vector<std::string> common = {
      "--profile", "vcp_c64", "--mic", wav.string(), "--diagnostic-stages"};
  auto disabled_args = common;
  disabled_args.insert(disabled_args.end(), {"--out", disabled.string()});
  run_cli(cli, std::move(disabled_args));
  auto enabled_args = common;
  enabled_args.insert(enabled_args.end(), {
      "--residual-ns-model", model.string(), "--residual-ns-mix", "1",
      "--residual-ns-floor-db", "-9", "--out", enabled.string()});
  run_cli(cli, std::move(enabled_args));

  const auto disabled_input = disabled.string() + ".tx_ns_input.float.wav";
  const auto enabled_input = enabled.string() + ".tx_ns_input.float.wav";
  require(std::filesystem::exists(disabled_input),
          "disabled CLI did not export tx_ns_input");
  require(std::filesystem::exists(enabled_input),
          "enabled CLI did not export tx_ns_input");
  require(read_bytes(disabled_input) == read_bytes(enabled_input),
          "tx_ns_input changed when residual model was enabled");
}
}  // namespace

int main(int argc, char** argv) {
  const auto directory = std::filesystem::temp_directory_path() /
                         ("itu-residual-vcp-" + std::to_string(::getpid()));
  try {
    std::filesystem::create_directories(directory);
    const auto model = write_zero_model(directory);
    profile_contract();
    validation_contract(model);
    fallback_contract();
    engine_telemetry(model);
    const auto executable = std::filesystem::absolute(argv[0]);
    const auto cli = argc > 1 ? std::filesystem::path(argv[1])
                              : executable.parent_path() / "itu_audio_cli";
    require(std::filesystem::exists(cli), "itu_audio_cli is required for tx_ns_input test");
    cli_preboundary_contract(cli, directory, model);
    std::filesystem::remove_all(directory);
    std::cout << "residual VCP structural validation/disable/fallback/profile/tx_ns_input checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::filesystem::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
