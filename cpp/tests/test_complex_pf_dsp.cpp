#include "complex_pf_dsp.hpp"
#include "itu/audio_engine.hpp"
#include "itu/audio_engine_c.h"
#include "itu/complex_pf_dsp_provider.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char kModelId[] = "complex_pf_stream_v1";
constexpr char kOtherModelId[] = "complex_pf_other_v1";
constexpr char kHashA[] =
    "0123456789abcdef0123456789abcdef"
    "0123456789abcdef0123456789abcdef";
constexpr char kHashB[] =
    "fedcba9876543210fedcba9876543210"
    "fedcba9876543210fedcba9876543210";
constexpr std::uint32_t kSamples = 160;
constexpr std::uint32_t kDelaySamples = 480;

static_assert(sizeof(kHashA) == 65, "test provider model hash must be 64 hex characters");
static_assert(sizeof(kHashB) == 65, "test provider changed hash must be 64 hex characters");
static_assert(sizeof(itu_audio_frame_info) == 64, "legacy C frame_info ABI must remain 64 bytes");

void require(bool condition, const std::string& reason) {
  if (!condition) throw std::runtime_error(reason);
}

template <std::size_t N>
void copy_text(char (&destination)[N], const char* source) noexcept {
  std::memset(destination, 0, N);
  const auto length = std::min(N - 1, std::strlen(source));
  std::memcpy(destination, source, length);
}

void set_error(char* error, std::size_t capacity, const char* message) noexcept {
  if (!error || capacity == 0) return;
  const auto length = std::min(capacity - 1, std::strlen(message));
  std::memcpy(error, message, length);
  error[length] = '\0';
}

enum class ProcessFault {
  none,
  nan_output,
  nonzero_return,
  wrong_frame_index,
  fallback_frame,
};

struct ProviderState {
  bool host_simulator = false;
  bool wrong_architecture = false;
  bool missing_hash = false;
  bool wrong_info_abi = false;
  bool probe_model_mismatch = false;
  bool create_hash_mismatch = false;
  ProcessFault process_fault = ProcessFault::none;

  unsigned probe_calls = 0;
  unsigned create_calls = 0;
  unsigned process_calls = 0;
  unsigned destroy_calls = 0;
  bool create_args_valid = false;
  bool frame_protocol_valid = false;
  bool pointer_match = false;
  bool input_values_match = false;
  unsigned last_samples = 0;
  std::uint64_t last_frame_index = 0;
  const float* expected_residual = nullptr;
  const float* expected_farend = nullptr;
};

ProviderState* active_state = nullptr;

void fill_info(itu_complex_pf_dsp_info_v1& info) {
  info = {};
  info.struct_size = sizeof(info);
  info.version = ITU_COMPLEX_PF_DSP_ABI;
  info.available = 1;
  info.sample_rate = 16000;
  info.callback_samples = kSamples;
  info.delay_samples = kDelaySamples;
  info.working_memory_bytes = 4096;
  copy_text(info.execution_provider, "hifi_dsp");
  copy_text(info.architecture, "mdf_complex_pf");
  copy_text(info.device_id, "hifi0");
  copy_text(info.model_id, kModelId);
  copy_text(info.model_sha256, kHashA);
  copy_text(info.frontend_sha256, kHashA);
  copy_text(info.provider_sha256, kHashA);
}

int fake_probe(const char* model_id, itu_complex_pf_dsp_info_v1* info,
               char* error, std::size_t error_capacity) {
  if (!active_state || !info) {
    set_error(error, error_capacity, "test probe missing state/info");
    return 1;
  }
  auto& state = *active_state;
  ++state.probe_calls;
  if (!model_id || std::strcmp(model_id, kModelId) != 0) {
    set_error(error, error_capacity, "unexpected model id");
    return 1;
  }
  fill_info(*info);
  if (state.wrong_info_abi) info->version = ITU_COMPLEX_PF_DSP_ABI + 1u;
  if (state.host_simulator) copy_text(info->execution_provider, "host_simulator");
  if (state.wrong_architecture) copy_text(info->architecture, "host_stft_complex_pf");
  if (state.missing_hash) info->provider_sha256[0] = '\0';
  if (state.probe_model_mismatch) copy_text(info->model_id, kOtherModelId);
  return 0;
}

void* fake_create(const char* model_id, std::uint32_t rate, std::uint32_t frame_samples,
                  itu_complex_pf_dsp_info_v1* info, char* error,
                  std::size_t error_capacity) {
  if (!active_state || !info) {
    set_error(error, error_capacity, "test create missing state/info");
    return nullptr;
  }
  auto& state = *active_state;
  ++state.create_calls;
  state.create_args_valid = model_id && std::strcmp(model_id, kModelId) == 0 &&
                            rate == 16000 && frame_samples == kSamples;
  if (!state.create_args_valid) {
    set_error(error, error_capacity, "unexpected create arguments");
    return nullptr;
  }
  fill_info(*info);
  if (state.create_hash_mismatch) copy_text(info->provider_sha256, kHashB);
  return &state;
}

int fake_process(void* handle, const float* residual, const float* farend,
                 float* output, std::uint32_t samples,
                 itu_complex_pf_dsp_frame_v1* frame) {
  auto* state = static_cast<ProviderState*>(handle);
  if (!state || !residual || !farend || !output || !frame || samples != kSamples) return 1;

  ++state->process_calls;
  state->last_samples = samples;
  state->last_frame_index = frame->frame_index;
  state->frame_protocol_valid =
      frame->struct_size == sizeof(*frame) &&
      frame->version == ITU_COMPLEX_PF_DSP_ABI &&
      frame->frame_index == state->process_calls - 1;
  state->pointer_match =
      residual == state->expected_residual && farend == state->expected_farend;
  state->input_values_match = state->expected_residual && state->expected_farend;
  for (std::uint32_t i = 0; state->input_values_match && i < samples; ++i) {
    state->input_values_match =
        residual[i] == state->expected_residual[i] && farend[i] == state->expected_farend[i];
  }

  for (std::uint32_t i = 0; i < samples; ++i) output[i] = .5f * residual[i];
  frame->inferences = frame->frame_index + 2;
  frame->fallback_frames = 0;
  frame->cycles = 40 + frame->frame_index;

  switch (state->process_fault) {
    case ProcessFault::none:
      return 0;
    case ProcessFault::nan_output:
      output[samples / 2] = std::numeric_limits<float>::quiet_NaN();
      return 0;
    case ProcessFault::nonzero_return:
      return 17;
    case ProcessFault::wrong_frame_index:
      ++frame->frame_index;
      return 0;
    case ProcessFault::fallback_frame:
      frame->fallback_frames = 1;
      return 0;
  }
  return 1;
}

void fake_destroy(void* handle) {
  auto* state = static_cast<ProviderState*>(handle);
  if (state) ++state->destroy_calls;
}

itu_complex_pf_dsp_provider_v1 make_provider() {
  itu_complex_pf_dsp_provider_v1 provider{};
  provider.struct_size = sizeof(provider);
  provider.version = ITU_COMPLEX_PF_DSP_ABI;
  provider.probe = &fake_probe;
  provider.create = &fake_create;
  provider.process = &fake_process;
  provider.destroy = &fake_destroy;
  return provider;
}

itu::Config valid_config() {
  itu::Config config;
  config.sample_rate = 16000;
  config.frame_size = kSamples;
  config.microphones = 1;
  config.reference_channels = 1;
  config.aec_backend = itu::AecBackend::speex_fused_dsp;
  config.speex_fused_dsp_model_id = kModelId;
  config.speex_fused_dsp_deadline_us = 10000;
  return config;
}

void route_identity_and_default_host_capability() {
  require(itu::AecBackend::speex_fused_cpu != itu::AecBackend::speex_fused_dsp,
          "CPU and DSP AEC enum values must be distinct");
  require(std::strcmp(itu::aec_backend_id(itu::AecBackend::speex_fused_cpu),
                      "speex_fused_cpu") == 0,
          "CPU AEC id changed");
  require(std::strcmp(itu::aec_backend_id(itu::AecBackend::speex_fused_dsp),
                      "speex_fused_dsp") == 0,
          "DSP AEC id changed");
  require(std::strcmp(itu::aec_backend_report_name(itu::AecBackend::speex_fused_cpu),
                      "speex_mdf_nlms_fused_cpu") == 0,
          "CPU AEC report mapping changed");
  require(std::strcmp(itu::aec_backend_report_name(itu::AecBackend::speex_fused_dsp),
                      "speex_mdf_complex_pf_dsp") == 0,
          "DSP AEC report mapping changed");
  require(std::strcmp(itu::aec_backend_report_name(itu::AecBackend::speex_fused_cpu),
                      itu::aec_backend_report_name(itu::AecBackend::speex_fused_dsp)) != 0,
          "CPU and DSP report names must remain distinct");

  const auto cpu = itu::aec_backend_capability(itu::AecBackend::speex_fused_cpu, kModelId);
  require(cpu.available && cpu.execution_provider == "cpu" && cpu.device_id == "host_cpu",
          "CPU fused AEC capability should remain available on the host");

  const auto direct_host = itu::complex_pf_dsp_capability(kModelId, nullptr);
  require(!direct_host.available, "DSP capability must be unavailable without a provider");
  require(direct_host.id == "speex_fused_dsp" &&
              direct_host.architecture == "mdf_complex_pf",
          "DSP capability identity changed");
  require(direct_host.execution_provider.empty() && direct_host.device_id.empty(),
          "unavailable DSP capability must not claim a CPU executor/device");
  require(direct_host.reason.find("HiFi SDK/provider is not linked") != std::string::npos,
          "unavailable DSP capability must explain the missing HiFi provider");

  const auto routed = itu::aec_backend_capability(itu::AecBackend::speex_fused_dsp, kModelId);
  require(!routed.available, "default host route unexpectedly claims DSP availability");
  require(routed.id == "speex_fused_dsp" && routed.architecture == "mdf_complex_pf",
          "default DSP route identity changed");
  require(routed.execution_provider != "cpu" && routed.execution_provider != "host_cpu",
          "default DSP route must not masquerade as a CPU backend");
  require(routed.execution_provider.empty() && routed.device_id.empty(),
          "unavailable default DSP route must not expose a synthetic device");
}

void expect_capability_rejection(ProviderState state, const char* expected_reason) {
  active_state = &state;
  auto provider = make_provider();
  const auto capability = itu::complex_pf_dsp_capability(kModelId, &provider);
  require(!capability.available, std::string("invalid provider was accepted: ") + expected_reason);
  require(capability.reason.find(expected_reason) != std::string::npos,
          std::string("unexpected capability rejection reason: ") + capability.reason);
  require(capability.execution_provider.empty() && capability.device_id.empty(),
          "rejected provider leaked execution identity");
  active_state = nullptr;
}

void capability_rejections_and_probe_create_identity() {
  {
    ProviderState state;
    state.host_simulator = true;
    expect_capability_rejection(state, "real hifi_dsp provider");
  }
  {
    ProviderState state;
    state.wrong_architecture = true;
    expect_capability_rejection(state, "MDF + Complex_PF architecture");
  }
  {
    ProviderState state;
    state.missing_hash = true;
    expect_capability_rejection(state, "SHA-256 identities");
  }
  {
    ProviderState state;
    state.wrong_info_abi = true;
    expect_capability_rejection(state, "metadata ABI mismatch");
  }
  {
    ProviderState state;
    active_state = &state;
    auto provider = make_provider();
    provider.version = ITU_COMPLEX_PF_DSP_ABI + 1u;
    const auto capability = itu::complex_pf_dsp_capability(kModelId, &provider);
    require(!capability.available, "provider with wrong ABI was accepted");
    require(capability.reason.find("provider ABI") != std::string::npos,
            "wrong provider ABI did not produce an ABI rejection");
    require(state.probe_calls == 0, "wrong provider ABI reached probe");
    active_state = nullptr;
  }
  {
    ProviderState state;
    state.create_hash_mismatch = true;
    active_state = &state;
    auto provider = make_provider();
    const auto config = valid_config();
    bool rejected = false;
    std::string error;
    try {
      itu::ComplexPfDsp dsp(config, &provider);
      (void)dsp;
    } catch (const std::runtime_error& exception) {
      rejected = true;
      error = exception.what();
    }
    require(rejected, "probe/create hash mismatch was accepted");
    require(error.find("identity changed between probe and create") != std::string::npos,
            "probe/create hash mismatch reported the wrong failure: " + error);
    require(state.probe_calls == 1 && state.create_calls == 1,
            "probe/create mismatch did not execute both identity checks");
    require(state.destroy_calls == 1,
            "probe/create mismatch did not destroy the rejected handle");
    active_state = nullptr;
  }
}

void injected_provider_process_and_diagnostics() {
  ProviderState state;
  active_state = &state;
  auto provider = make_provider();
  const auto config = valid_config();

  std::vector<float> residual(kSamples);
  std::vector<float> farend(kSamples);
  for (std::uint32_t i = 0; i < kSamples; ++i) {
    residual[i] = -.72f + .009f * static_cast<float>(i);
    farend[i] = .61f - .005f * static_cast<float>(i);
  }
  const auto residual_before = residual;
  const auto farend_before = farend;

  {
    itu::ComplexPfDsp dsp(config, &provider);
    require(state.probe_calls == 1 && state.create_calls == 1,
            "constructor did not probe/create exactly once");
    require(state.create_args_valid, "constructor passed wrong model/rate/frame to create");
    require(dsp.delay() == kDelaySamples, "declared whole-callback delay was not preserved");
    require(dsp.delay() % config.frame_size == 0,
            "declared delay must span an integer number of callbacks");

    std::vector<float> output(kSamples, .75f);
    state.expected_residual = residual.data();
    state.expected_farend = farend.data();
    auto status = dsp.process(residual.data(), farend.data(), output.data());
    require(status == itu::Status::ok, "valid injected provider process failed");
    require(state.frame_protocol_valid && state.pointer_match && state.input_values_match,
            "raw MDF residual/farend were not passed through unchanged");
    require(state.last_samples == kSamples && state.last_frame_index == 0,
            "first callback shape/index contract changed");
    require(residual == residual_before && farend == farend_before,
            "provider processing mutated its const inputs");

    bool nonidentity = false;
    for (std::uint32_t i = 0; i < kSamples; ++i) {
      require(std::abs(output[i] - .5f * residual[i]) < 1e-7f,
              "controlled provider output contract was not preserved");
      nonidentity = nonidentity || output[i] != residual[i];
    }
    require(nonidentity, "controlled provider output was accidentally an identity transform");

    auto diagnostics = dsp.diagnostics();
    require(diagnostics.execution_provider == "hifi_dsp" &&
                diagnostics.device_id == "hifi0" &&
                diagnostics.model_id == kModelId,
            "route diagnostics lost provider identity");
    require(diagnostics.model_sha256 == kHashA &&
                diagnostics.frontend_sha256 == kHashA &&
                diagnostics.provider_sha256 == kHashA,
            "route diagnostics lost fixed SHA-256 identity");
    require(diagnostics.neural_delay_samples == kDelaySamples,
            "route diagnostics lost declared delay");
    require(diagnostics.processed_frames == 1 && diagnostics.inferences == 2 &&
                diagnostics.cycles == 40,
            "first callback counters are wrong");

    std::vector<float> residual2(kSamples);
    std::vector<float> farend2(kSamples);
    for (std::uint32_t i = 0; i < kSamples; ++i) {
      residual2[i] = .34f - .006f * static_cast<float>(i);
      farend2[i] = -.55f + .004f * static_cast<float>(i);
    }
    std::vector<float> output2(kSamples, .75f);
    state.expected_residual = residual2.data();
    state.expected_farend = farend2.data();
    status = dsp.process(residual2.data(), farend2.data(), output2.data());
    require(status == itu::Status::ok, "second valid injected provider process failed");
    require(state.frame_protocol_valid && state.pointer_match && state.input_values_match,
            "second callback raw MDF residual/farend pass-through failed");
    require(state.last_frame_index == 1, "second callback frame index did not advance");

    diagnostics = dsp.diagnostics();
    require(diagnostics.processed_frames == 2 && diagnostics.inferences == 5 &&
                diagnostics.cycles == 81,
            "callback inference/cycle counters did not accumulate");
  }

  require(state.destroy_calls == 1, "provider handle was not destroyed exactly once");
  active_state = nullptr;
}

void expect_sticky_process_failure(ProcessFault fault, const char* label) {
  ProviderState state;
  state.process_fault = fault;
  active_state = &state;
  auto provider = make_provider();
  const auto config = valid_config();
  std::vector<float> residual(kSamples, .2f);
  std::vector<float> farend(kSamples, -.1f);
  state.expected_residual = residual.data();
  state.expected_farend = farend.data();

  {
    itu::ComplexPfDsp dsp(config, &provider);
    std::vector<float> output(kSamples, .75f);
    auto status = dsp.process(residual.data(), farend.data(), output.data());
    require(status == itu::Status::backend_error,
            std::string(label) + " did not produce backend_error");
    require(std::all_of(output.begin(), output.end(), [](float sample) { return sample == 0.f; }),
            std::string(label) + " did not clear the rejected output frame");

    auto diagnostics = dsp.diagnostics();
    require(diagnostics.processed_frames == 0 && diagnostics.inferences == 0 &&
                diagnostics.cycles == 0,
            std::string(label) + " advanced counters after failure");

    state.process_fault = ProcessFault::none;
    std::vector<float> retry_output(kSamples, 0.f);
    status = dsp.process(residual.data(), farend.data(), retry_output.data());
    require(status == itu::Status::backend_error,
            std::string(label) + " allowed a later successful callback");
    require(state.process_calls == 1,
            std::string(label) + " invoked the provider again after sticky failure");
  }

  require(state.destroy_calls == 1, std::string(label) + " leaked the provider handle");
  active_state = nullptr;
}

void process_failure_contracts() {
  expect_sticky_process_failure(ProcessFault::nan_output, "NaN output");
  expect_sticky_process_failure(ProcessFault::nonzero_return, "nonzero provider return");
  expect_sticky_process_failure(ProcessFault::wrong_frame_index, "wrong frame_index");
  expect_sticky_process_failure(ProcessFault::fallback_frame, "fallback_frames");
}

void expect_config_rejection(itu::Config config, const char* label) {
  ProviderState state;
  active_state = &state;
  auto provider = make_provider();
  bool rejected = false;
  std::string error;
  try {
    itu::ComplexPfDsp dsp(config, &provider);
    (void)dsp;
  } catch (const std::invalid_argument& exception) {
    rejected = true;
    error = exception.what();
  }
  require(rejected, std::string(label) + " configuration was accepted");
  require(error.find("16 kHz / 160 samples / mono capture and playback") !=
              std::string::npos,
          std::string(label) + " configuration reported the wrong constraint");
  require(state.probe_calls == 0 && state.create_calls == 0,
          std::string(label) + " configuration reached the provider");
  active_state = nullptr;
}

void configuration_constraints() {
  auto config = valid_config();
  config.sample_rate = 48000;
  expect_config_rejection(config, "non-16-kHz");

  config = valid_config();
  config.frame_size = 320;
  expect_config_rejection(config, "non-160-sample");

  config = valid_config();
  config.microphones = 2;
  expect_config_rejection(config, "multi-microphone capture");

  config = valid_config();
  config.reference_channels = 2;
  expect_config_rejection(config, "multi-channel reference");
}

void c_aec_route_api() {
  itu_audio_config config{};
  itu_audio_default_config(&config);
  require(config.struct_size == sizeof(config) && config.abi_version == ITU_AUDIO_ABI_VERSION,
          "default C config ABI changed");

  itu_audio_extensions_v2 extensions{};
  itu_audio_default_extensions_v2(&extensions);
  require(extensions.struct_size == sizeof(extensions) && extensions.version == 2 &&
              extensions.base.struct_size == sizeof(extensions.base) &&
              extensions.base.version == 1,
          "default C extensions v2 ABI changed");

  extensions.aec_backend = ITU_AEC_FUSED_CPU;
  std::array<char, 512> error{};
  auto* engine = itu_audio_create_extended_v2(
      &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_WEBRTC, nullptr, &extensions,
      error.data(), error.size());
  require(engine != nullptr, std::string("C fused CPU route creation failed: ") + error.data());

  itu_audio_aec_route_info_v1 route{};
  route.struct_size = sizeof(route);
  route.version = 1;
  const auto route_status = itu_audio_get_aec_route_info_v1(engine, &route);
  require(route_status == ITU_AUDIO_OK,
          std::string("C route info query failed with status ") +
              std::to_string(route_status));
  require(route.requested_backend == ITU_AEC_FUSED_CPU,
          "C route info lost the requested fused CPU backend");
  require(std::strcmp(route.backend_id, "speex_fused_cpu") == 0,
          std::string("C route info returned wrong backend id: ") + route.backend_id);
  require(std::strcmp(route.effective_backend, "speex_mdf_nlms_fused_cpu") == 0,
          std::string("C route info returned wrong effective backend: ") +
              route.effective_backend);
  require(std::strcmp(route.execution_provider, "cpu") == 0,
          std::string("C route info returned wrong execution provider: ") +
              route.execution_provider);
  require(std::strcmp(route.device_id, "host_cpu") == 0,
          std::string("C route info returned wrong device id: ") + route.device_id);
  itu_audio_destroy(engine);

  extensions = {};
  itu_audio_default_extensions_v2(&extensions);
  extensions.aec_backend = ITU_AEC_LOCALVQE_AEC200K;
  error.fill(0);
  engine = itu_audio_create_extended_v2(
      &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_WEBRTC, nullptr, &extensions,
      error.data(), error.size());
  require(engine != nullptr,
          std::string("C LocalVQE route creation failed: ") + error.data());
  route = {};
  route.struct_size = sizeof(route);
  route.version = 1;
  require(itu_audio_get_aec_route_info_v1(engine, &route) == ITU_AUDIO_OK,
          "C LocalVQE route info query failed");
  require(route.requested_backend == ITU_AEC_LOCALVQE_AEC200K,
          "C route info lost the requested LocalVQE backend");
  require(std::strcmp(route.backend_id, "localvqe_aec200k") == 0,
          std::string("C LocalVQE backend id mismatch: ") + route.backend_id);
  require(std::strcmp(route.effective_backend, "localvqe_aec200k_cpu") == 0,
          std::string("C LocalVQE effective backend mismatch: ") + route.effective_backend);
  itu_audio_destroy(engine);

  extensions = {};
  itu_audio_default_extensions_v2(&extensions);
  extensions.aec_backend = ITU_AEC_MDF_COMPLEX_PF_DSP;
  error.fill(0);
  engine = itu_audio_create_extended_v2(
      &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_WEBRTC, nullptr, &extensions,
      error.data(), error.size());
  require(engine == nullptr, "C DSP route was created without a real HiFi provider");
  const std::string dsp_error(error.data());
  require(dsp_error.find("speex_fused_dsp") != std::string::npos &&
              dsp_error.find("HiFi") != std::string::npos,
          "C DSP route rejection did not identify speex_fused_dsp/HiFi: " + dsp_error);

  extensions = {};
  itu_audio_default_extensions_v2(&extensions);
  extensions.aec_backend = 999;
  error.fill(0);
  engine = itu_audio_create_extended_v2(
      &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_WEBRTC, nullptr, &extensions,
      error.data(), error.size());
  require(engine == nullptr, "unknown C AEC enum was accepted");
  require(std::string(error.data()).find("unknown AEC backend selection") != std::string::npos,
          std::string("unknown C AEC enum reported wrong error: ") + error.data());
  require(itu_audio_aec_backend_name_v2(999) == nullptr,
          "unknown C AEC enum received a backend name");
  require(itu_audio_aec_backend_available_v2(999) == -1,
          "unknown C AEC enum received an availability result");

  extensions = {};
  itu_audio_default_extensions_v2(&extensions);
  extensions.aec_backend = ITU_AEC_FUSED_CPU;
  extensions.version = 1;
  error.fill(0);
  engine = itu_audio_create_extended_v2(
      &config, ITU_AUDIO_PROFILE_GENERAL, ITU_AUDIO_NS_WEBRTC, nullptr, &extensions,
      error.data(), error.size());
  require(engine == nullptr, "C extensions v2 with wrong version was accepted");
  require(std::string(error.data()).find("AEC extensions v2 ABI mismatch") !=
              std::string::npos,
          std::string("C extensions v2 wrong version reported wrong error: ") + error.data());
}

void c_frame_info_abi() {
  require(sizeof(itu_audio_frame_info) == 64, "legacy C frame_info ABI changed");
}

}  // namespace

int main() {
  try {
    route_identity_and_default_host_capability();
    capability_rejections_and_probe_create_identity();
    injected_provider_process_and_diagnostics();
    process_failure_contracts();
    configuration_constraints();
    c_aec_route_api();
    c_frame_info_abi();
    std::cout << "[PASS] Complex_PF DSP route, provider identity, process and C ABI contracts\n";
    return 0;
  } catch (const std::exception& exception) {
    active_state = nullptr;
    std::cerr << "[FAIL] " << exception.what() << '\n';
    return 1;
  }
}
