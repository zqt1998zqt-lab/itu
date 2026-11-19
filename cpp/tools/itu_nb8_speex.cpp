#include "speex/speex_echo.h"
#include "wav.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr unsigned kSampleRate = 8000;
constexpr unsigned kFrameSamples = 80;
constexpr std::size_t kMaximumWavBytes = 512u * 1024u * 1024u;

class Sha256 {
 public:
  Sha256()
      : state_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
               0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

  void update(const std::uint8_t* data, std::size_t size) {
    total_bytes_ += size;
    while (size != 0) {
      const std::size_t take = std::min(size, block_.size() - block_size_);
      std::memcpy(block_.data() + block_size_, data, take);
      block_size_ += take;
      data += take;
      size -= take;
      if (block_size_ == block_.size()) {
        transform(block_.data());
        block_size_ = 0;
      }
    }
  }

  std::string finish_hex() {
    const std::uint64_t bit_length = total_bytes_ * 8u;
    block_[block_size_++] = 0x80u;
    if (block_size_ > 56) {
      std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                block_.end(), 0u);
      transform(block_.data());
      block_size_ = 0;
    }
    std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
              block_.begin() + 56, 0u);
    for (unsigned i = 0; i < 8; ++i) {
      block_[56 + i] =
          static_cast<std::uint8_t>(bit_length >> ((7u - i) * 8u));
    }
    transform(block_.data());

    std::ostringstream text;
    text << std::hex << std::setfill('0');
    for (const std::uint32_t word : state_) text << std::setw(8) << word;
    return text.str();
  }

 private:
  static std::uint32_t rotate_right(std::uint32_t value, unsigned count) {
    return (value >> count) | (value << (32u - count));
  }

  void transform(const std::uint8_t* block) {
    static constexpr std::array<std::uint32_t, 64> constants{
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
        0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
        0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
        0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
        0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
        0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
        0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
        0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

    std::array<std::uint32_t, 64> words{};
    for (unsigned i = 0; i < 16; ++i) {
      words[i] = static_cast<std::uint32_t>(block[i * 4]) << 24u |
                 static_cast<std::uint32_t>(block[i * 4 + 1]) << 16u |
                 static_cast<std::uint32_t>(block[i * 4 + 2]) << 8u |
                 static_cast<std::uint32_t>(block[i * 4 + 3]);
    }
    for (unsigned i = 16; i < words.size(); ++i) {
      const std::uint32_t s0 = rotate_right(words[i - 15], 7) ^
                               rotate_right(words[i - 15], 18) ^
                               (words[i - 15] >> 3u);
      const std::uint32_t s1 = rotate_right(words[i - 2], 17) ^
                               rotate_right(words[i - 2], 19) ^
                               (words[i - 2] >> 10u);
      words[i] = words[i - 16] + s0 + words[i - 7] + s1;
    }

    std::uint32_t a = state_[0];
    std::uint32_t b = state_[1];
    std::uint32_t c = state_[2];
    std::uint32_t d = state_[3];
    std::uint32_t e = state_[4];
    std::uint32_t f = state_[5];
    std::uint32_t g = state_[6];
    std::uint32_t h = state_[7];
    for (unsigned i = 0; i < words.size(); ++i) {
      const std::uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                                 rotate_right(e, 25);
      const std::uint32_t choice = (e & f) ^ (~e & g);
      const std::uint32_t temp1 = h + sum1 + choice + constants[i] + words[i];
      const std::uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                                 rotate_right(a, 22);
      const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
      const std::uint32_t temp2 = sum0 + majority;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
  }

  std::array<std::uint32_t, 8> state_;
  std::array<std::uint8_t, 64> block_{};
  std::size_t block_size_ = 0;
  std::uint64_t total_bytes_ = 0;
};

std::uint16_t little_u16(const std::uint8_t* bytes) {
  return static_cast<std::uint16_t>(bytes[0]) |
         static_cast<std::uint16_t>(bytes[1]) << 8u;
}

std::uint32_t little_u32(const std::uint8_t* bytes) {
  return static_cast<std::uint32_t>(bytes[0]) |
         static_cast<std::uint32_t>(bytes[1]) << 8u |
         static_cast<std::uint32_t>(bytes[2]) << 16u |
         static_cast<std::uint32_t>(bytes[3]) << 24u;
}

std::vector<std::uint8_t> read_file_bytes(const std::string& path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("cannot open WAV: " + path);
  const std::streamoff length = file.tellg();
  if (length < 12 ||
      static_cast<std::uint64_t>(length) > kMaximumWavBytes) {
    throw std::runtime_error("WAV size must be 12 bytes..512 MiB: " + path);
  }
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(length));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  if (!file) throw std::runtime_error("cannot read complete WAV: " + path);
  return bytes;
}

struct StrictWav {
  unsigned sample_rate = 0;
  unsigned channels = 0;
  unsigned bits_per_sample = 0;
  std::string format;
  std::vector<float> samples;
  std::vector<std::int16_t> pcm16;
  std::string sha256;
  std::size_t clipped_samples = 0;
};

StrictWav read_strict_wav(const std::string& path) {
  const std::vector<std::uint8_t> bytes = read_file_bytes(path);
  if (std::memcmp(bytes.data(), "RIFF", 4) != 0 ||
      std::memcmp(bytes.data() + 8, "WAVE", 4) != 0) {
    throw std::runtime_error("expected little-endian RIFF WAVE: " + path);
  }
  const std::size_t riff_end =
      static_cast<std::size_t>(little_u32(bytes.data() + 4)) + 8u;
  if (riff_end < 12 || riff_end > bytes.size()) {
    throw std::runtime_error("truncated RIFF WAVE: " + path);
  }

  bool has_fmt = false;
  bool has_data = false;
  unsigned format_tag = 0;
  unsigned channels = 0;
  unsigned sample_rate = 0;
  unsigned block_align = 0;
  unsigned bits_per_sample = 0;
  std::size_t data_offset = 0;
  std::size_t data_size = 0;

  for (std::size_t offset = 12; offset + 8 <= riff_end;) {
    const std::size_t chunk_size =
        static_cast<std::size_t>(little_u32(bytes.data() + offset + 4));
    const std::size_t payload = offset + 8;
    if (chunk_size > riff_end - payload) {
      throw std::runtime_error("truncated WAV chunk: " + path);
    }
    if (std::memcmp(bytes.data() + offset, "fmt ", 4) == 0) {
      if (has_fmt || chunk_size < 16) {
        throw std::runtime_error("invalid or duplicate fmt chunk: " + path);
      }
      format_tag = little_u16(bytes.data() + payload);
      channels = little_u16(bytes.data() + payload + 2);
      sample_rate = little_u32(bytes.data() + payload + 4);
      block_align = little_u16(bytes.data() + payload + 12);
      bits_per_sample = little_u16(bytes.data() + payload + 14);
      has_fmt = true;
    } else if (std::memcmp(bytes.data() + offset, "data", 4) == 0) {
      if (has_data) {
        throw std::runtime_error("multiple WAV data chunks unsupported: " +
                                 path);
      }
      data_offset = payload;
      data_size = chunk_size;
      has_data = true;
    }
    const std::size_t advance = chunk_size + (chunk_size & 1u);
    if (advance > riff_end - payload) {
      throw std::runtime_error("missing RIFF chunk padding: " + path);
    }
    offset = payload + advance;
  }

  if (!has_fmt || !has_data) {
    throw std::runtime_error("WAV requires fmt and data chunks: " + path);
  }
  const bool pcm16 = format_tag == 1 && bits_per_sample == 16;
  const bool float32 = format_tag == 3 && bits_per_sample == 32;
  if (!pcm16 && !float32) {
    throw std::runtime_error(
        "input WAV must be PCM16 or IEEE FLOAT32: " + path);
  }
  if (channels != 1) {
    throw std::runtime_error("input WAV must be mono: " + path);
  }
  if (sample_rate != kSampleRate) {
    throw std::runtime_error("input WAV must be 8000 Hz: " + path);
  }
  const unsigned bytes_per_sample = bits_per_sample / 8u;
  if (block_align != bytes_per_sample || data_size == 0 ||
      data_size % block_align != 0) {
    throw std::runtime_error("invalid WAV data alignment or empty data: " +
                             path);
  }

  StrictWav wav;
  wav.sample_rate = sample_rate;
  wav.channels = channels;
  wav.bits_per_sample = bits_per_sample;
  wav.format = pcm16 ? "PCM16" : "FLOAT32";
  const std::size_t sample_count = data_size / bytes_per_sample;
  wav.samples.resize(sample_count);
  wav.pcm16.resize(sample_count);

  for (std::size_t i = 0; i < sample_count; ++i) {
    const std::uint8_t* source =
        bytes.data() + data_offset + i * bytes_per_sample;
    if (pcm16) {
      const std::uint16_t raw = little_u16(source);
      const int signed_value =
          raw < 0x8000u ? static_cast<int>(raw)
                        : static_cast<int>(raw) - 65536;
      wav.pcm16[i] = static_cast<std::int16_t>(signed_value);
      wav.samples[i] = static_cast<float>(signed_value) / 32768.0f;
    } else {
      const std::uint32_t raw = little_u32(source);
      float value = 0.0f;
      std::memcpy(&value, &raw, sizeof(value));
      if (!std::isfinite(value)) {
        throw std::runtime_error("input WAV contains non-finite samples: " +
                                 path);
      }
      wav.samples[i] = value;
      constexpr float kMaximum = 32767.0f / 32768.0f;
      if (value > kMaximum || value < -1.0f) ++wav.clipped_samples;
      const float bounded = std::clamp(value, -1.0f, kMaximum);
      const long quantized = std::lround(bounded * 32768.0f);
      wav.pcm16[i] = static_cast<std::int16_t>(
          std::clamp(quantized, -32768L, 32767L));
    }
  }

  Sha256 sha256;
  sha256.update(bytes.data(), bytes.size());
  wav.sha256 = sha256.finish_hex();
  return wav;
}

std::string sha256_file(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot hash output WAV: " + path);
  Sha256 sha256;
  std::array<char, 64 * 1024> buffer{};
  while (file) {
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = file.gcount();
    if (count > 0) {
      sha256.update(reinterpret_cast<const std::uint8_t*>(buffer.data()),
                    static_cast<std::size_t>(count));
    }
  }
  if (!file.eof()) throw std::runtime_error("failed while hashing output WAV");
  return sha256.finish_hex();
}

std::string json_string(const std::string& value) {
  std::ostringstream output;
  output << '"';
  for (const unsigned char ch : value) {
    if (ch == '"' || ch == '\\') {
      output << '\\' << static_cast<char>(ch);
    } else if (ch < 32) {
      output << "\\u00" << std::hex << std::setfill('0') << std::setw(2)
             << static_cast<unsigned>(ch) << std::dec;
    } else {
      output << static_cast<char>(ch);
    }
  }
  output << '"';
  return output.str();
}

struct Arguments {
  std::string mic;
  std::string echo;
  std::string prefix;
  int tail_ms = 500;
};

void usage(std::ostream& output) {
  output
      << "usage: itu_nb8_speex --mic PATH --echo PATH --out PREFIX "
         "[--tail-ms 500]\n"
      << "inputs: same-length mono 8000 Hz PCM16 or FLOAT32 WAV\n"
      << "outputs: PREFIX.wav (PCM16) and PREFIX.json\n"
      << "tail-ms range: 10..10000\n";
}

int parse_tail_ms(const std::string& text) {
  std::size_t used = 0;
  long long value = 0;
  try {
    value = std::stoll(text, &used);
  } catch (const std::exception&) {
    throw std::runtime_error("--tail-ms must be an integer");
  }
  if (used != text.size() || value < 10 || value > 10000) {
    throw std::runtime_error("--tail-ms must be an integer in 10..10000");
  }
  return static_cast<int>(value);
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  bool mic_seen = false;
  bool echo_seen = false;
  bool prefix_seen = false;
  bool tail_seen = false;
  for (int i = 1; i < argc; ++i) {
    const std::string option = argv[i];
    if (option == "--help" || option == "-h") {
      usage(std::cout);
      std::exit(0);
    }
    if (i + 1 >= argc) {
      throw std::runtime_error("missing value for " + option);
    }
    const std::string value = argv[++i];
    if (option == "--mic") {
      if (mic_seen) throw std::runtime_error("duplicate --mic");
      arguments.mic = value;
      mic_seen = true;
    } else if (option == "--echo") {
      if (echo_seen) throw std::runtime_error("duplicate --echo");
      arguments.echo = value;
      echo_seen = true;
    } else if (option == "--out") {
      if (prefix_seen) throw std::runtime_error("duplicate --out");
      arguments.prefix = value;
      prefix_seen = true;
    } else if (option == "--tail-ms") {
      if (tail_seen) throw std::runtime_error("duplicate --tail-ms");
      arguments.tail_ms = parse_tail_ms(value);
      tail_seen = true;
    } else {
      throw std::runtime_error("unknown option: " + option);
    }
  }
  if (!mic_seen || !echo_seen || !prefix_seen) {
    throw std::runtime_error("--mic, --echo and --out are required");
  }
  return arguments;
}

std::string absolute_normalized(const std::string& path) {
  return std::filesystem::absolute(path).lexically_normal().string();
}

struct EchoStateDeleter {
  void operator()(SpeexEchoState* state) const noexcept {
    speex_echo_state_destroy(state);
  }
};

using EchoState =
    std::unique_ptr<SpeexEchoState, EchoStateDeleter>;

EchoState create_echo_state(int tail_ms) {
  const int tail_samples = tail_ms * static_cast<int>(kSampleRate / 1000u);
  EchoState state(speex_echo_state_init_mc(
      static_cast<int>(kFrameSamples), tail_samples, 1, 1));
  if (!state) throw std::runtime_error("Speex MDF initialization failed");

  int sample_rate = static_cast<int>(kSampleRate);
  if (speex_echo_ctl(state.get(), SPEEX_ECHO_SET_SAMPLING_RATE,
                     &sample_rate) != 0) {
    throw std::runtime_error("Speex MDF rejected 8000 Hz");
  }
  int reported_rate = 0;
  if (speex_echo_ctl(state.get(), SPEEX_ECHO_GET_SAMPLING_RATE,
                     &reported_rate) != 0 ||
      reported_rate != static_cast<int>(kSampleRate)) {
    throw std::runtime_error("Speex MDF sample-rate verification failed");
  }
  int reported_frame = 0;
  if (speex_echo_ctl(state.get(), SPEEX_ECHO_GET_FRAME_SIZE,
                     &reported_frame) != 0 ||
      reported_frame != static_cast<int>(kFrameSamples)) {
    throw std::runtime_error("Speex MDF frame-size verification failed");
  }
  return state;
}

void write_report(const std::string& path, const Arguments& arguments,
                  const StrictWav& mic, const StrictWav& echo,
                  const std::string& output_wav,
                  const std::string& output_sha256,
                  std::size_t inference_frames,
                  std::size_t padded_tail_samples,
                  std::size_t output_samples,
                  std::size_t input_clipped_samples) {
  std::ofstream report(path);
  if (!report) throw std::runtime_error("cannot write JSON report: " + path);
  const int tail_samples =
      arguments.tail_ms * static_cast<int>(kSampleRate / 1000u);
  report
      << "{\n"
      << "  \"schema\": \"itu.nb8_speex.report.v1\",\n"
      << "  \"schema_version\": 1,\n"
      << "  \"sample_rate\": " << kSampleRate << ",\n"
      << "  \"channels\": 1,\n"
      << "  \"frame_samples\": " << kFrameSamples << ",\n"
      << "  \"tail_ms\": " << arguments.tail_ms << ",\n"
      << "  \"tail_samples\": " << tail_samples << ",\n"
      << "  \"aec_backend\": \"speex_mdf_nb8\",\n"
      << "  \"aec_inference_frames\": " << inference_frames << ",\n"
      << "  \"aec_padded_tail_samples\": " << padded_tail_samples << ",\n"
      << "  \"input_samples\": " << mic.samples.size() << ",\n"
      << "  \"mic_input_samples\": " << mic.samples.size() << ",\n"
      << "  \"echo_input_samples\": " << echo.samples.size() << ",\n"
      << "  \"output_samples\": " << output_samples << ",\n"
      << "  \"input_sha256\": {\"mic\": "
      << json_string(mic.sha256) << ", \"echo\": "
      << json_string(echo.sha256) << "},\n"
      << "  \"mic_input_sha256\": " << json_string(mic.sha256) << ",\n"
      << "  \"echo_input_sha256\": " << json_string(echo.sha256) << ",\n"
      << "  \"output_sha256\": " << json_string(output_sha256) << ",\n"
      << "  \"input_hash_scope\": \"raw input file bytes\",\n"
      << "  \"output_hash_scope\": \"PCM16 WAV file bytes\",\n"
      << "  \"mic_input_format\": " << json_string(mic.format) << ",\n"
      << "  \"echo_input_format\": " << json_string(echo.format) << ",\n"
      << "  \"output_format\": \"PCM16\",\n"
      << "  \"finite\": true,\n"
      << "  \"output_finite\": true,\n"
      << "  \"input_clipped_samples\": " << input_clipped_samples << ",\n"
      << "  \"no_resample\": true,\n"
      << "  \"resampling\": \"none\",\n"
      << "  \"fallback_frames\": 0,\n"
      << "  \"algorithm_delay_samples\": 0,\n"
      << "  \"evidence_scope\": "
      << json_string(
             "Direct speex_echo_cancellation has no explicit delay stage. "
             "Mic/reference are assumed sample-aligned; no acoustic-delay "
             "estimator or cross-correlation proof was run. The value 0 "
             "means no tool-inserted delay, not a measured acoustic delay.")
      << ",\n"
      << "  \"mic_file\": " << json_string(absolute_normalized(arguments.mic))
      << ",\n"
      << "  \"echo_file\": " << json_string(absolute_normalized(arguments.echo))
      << ",\n"
      << "  \"output_file\": " << json_string(absolute_normalized(output_wav))
      << "\n"
      << "}\n";
  if (!report) throw std::runtime_error("JSON report write failed");
}

int run(const Arguments& arguments) {
  const std::string output_wav = arguments.prefix + ".wav";
  const std::string output_json = arguments.prefix + ".json";
  const std::string mic_path = absolute_normalized(arguments.mic);
  const std::string echo_path = absolute_normalized(arguments.echo);
  if (absolute_normalized(output_wav) == mic_path ||
      absolute_normalized(output_wav) == echo_path) {
    throw std::runtime_error("output WAV must not overwrite an input WAV");
  }

  const StrictWav mic = read_strict_wav(arguments.mic);
  const StrictWav echo = read_strict_wav(arguments.echo);
  if (mic.samples.size() != echo.samples.size()) {
    throw std::runtime_error(
        "mic and echo WAVs must contain exactly the same sample count");
  }

  EchoState echo_state = create_echo_state(arguments.tail_ms);
  const std::size_t inference_frames =
      (mic.samples.size() + kFrameSamples - 1u) / kFrameSamples;
  const std::size_t padded_tail_samples =
      inference_frames * kFrameSamples - mic.samples.size();
  std::vector<float> output(mic.samples.size());

  for (std::size_t frame = 0; frame < inference_frames; ++frame) {
    const std::size_t source_offset = frame * kFrameSamples;
    const std::size_t count =
        std::min<std::size_t>(kFrameSamples, mic.samples.size() - source_offset);
    std::array<spx_int16_t, kFrameSamples> mic_frame{};
    std::array<spx_int16_t, kFrameSamples> echo_frame{};
    std::array<spx_int16_t, kFrameSamples> output_frame{};
    std::copy_n(mic.pcm16.data() + source_offset, count, mic_frame.data());
    std::copy_n(echo.pcm16.data() + source_offset, count, echo_frame.data());
    speex_echo_cancellation(echo_state.get(), mic_frame.data(),
                            echo_frame.data(), output_frame.data());
    for (std::size_t i = 0; i < count; ++i) {
      output[source_offset + i] =
          static_cast<float>(output_frame[i]) / 32768.0f;
    }
  }

  for (const float sample : output) {
    if (!std::isfinite(sample)) {
      throw std::runtime_error("Speex MDF produced a non-finite sample");
    }
  }
  itu::write_wav(output_wav, kSampleRate, 1, output);
  const std::string output_sha256 = sha256_file(output_wav);
  write_report(output_json, arguments, mic, echo, output_wav, output_sha256,
               inference_frames, padded_tail_samples, output.size(),
               mic.clipped_samples + echo.clipped_samples);
  std::cout << "processed_samples=" << output.size()
            << " frames=" << inference_frames << " output=" << output_wav
            << " report=" << output_json << '\n';
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      usage(std::cerr);
      return 1;
    }
    return run(parse_arguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "itu_nb8_speex: " << error.what() << '\n';
    return 1;
  }
}
