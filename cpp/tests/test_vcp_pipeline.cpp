#include "drc.hpp"
#include "downlink_pipeline.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void test_drc_basic() {
  std::cout << "[RUN] test_drc_basic..." << std::endl;
  itu::DrcConfig cfg;
  cfg.threshold_dbfs = -12.0f;
  cfg.ratio = 4.0f;
  cfg.ceiling_dbfs = -1.0f;
  cfg.attack_ms = 2.0f;
  cfg.release_ms = 20.0f;

  itu::DynamicRangeCompressor drc(16000, cfg);

  // 1. Low level signal (-20 dBFS -> peak ~ 0.1): should not be compressed
  std::vector<float> low_level(160);
  for (int i = 0; i < 160; ++i) {
    low_level[i] = 0.1f * std::sin(2.0f * 3.14159f * 440.0f * i / 16000.0f);
  }
  std::vector<float> low_out = low_level;
  drc.process(low_out.data(), low_out.size());

  // Check output is close to input (no compression)
  for (size_t i = 0; i < low_out.size(); ++i) {
    assert(std::abs(low_out[i] - low_level[i]) < 0.05f);
  }

  // 2. High level signal (+2 dBFS peak ~ 1.25): should be compressed and clamped below ceiling (-1 dBFS = 0.891)
  std::vector<float> high_level(160 * 10);
  for (size_t i = 0; i < high_level.size(); ++i) {
    high_level[i] = 1.25f * std::sin(2.0f * 3.14159f * 1000.0f * i / 16000.0f);
  }
  std::vector<float> high_out = high_level;
  drc.process(high_out.data(), high_out.size());

  float max_val = 0.0f;
  for (float v : high_out) {
    max_val = std::max(max_val, std::abs(v));
    // Must never exceed ceiling
    assert(std::abs(v) <= 0.895f);
  }
  std::cout << "  High level input peak 1.25 -> compressed peak " << max_val << " (ceiling: 0.891)" << std::endl;
  std::cout << "[PASS] test_drc_basic" << std::endl;
}

void test_downlink_pipeline() {
  std::cout << "[RUN] test_downlink_pipeline..." << std::endl;
  itu::DownlinkConfig cfg;
  cfg.enabled = true;
  cfg.enable_nr = true;
  cfg.drc.enabled = true;
  cfg.drc.threshold_dbfs = -6.0f;
  cfg.drc.ceiling_dbfs = -0.5f;

  itu::DownlinkPipeline pipeline(16000, 160, cfg);

  std::vector<float> in(160 * 20);
  std::vector<float> out_speaker(160 * 20);
  std::vector<float> out_ref(160 * 20);

  // Generate test signal with loud burst
  for (size_t i = 0; i < in.size(); ++i) {
    float amp = (i > 160 * 5 && i < 160 * 15) ? 1.2f : 0.2f;
    in[i] = amp * std::sin(2.0f * 3.14159f * 800.0f * i / 16000.0f);
  }

  // Process frame by frame
  for (size_t frame = 0; frame < 20; ++frame) {
    pipeline.process(&in[frame * 160], &out_speaker[frame * 160], &out_ref[frame * 160], 160);
  }

  // Verify speaker output and ref match
  for (size_t i = 0; i < out_speaker.size(); ++i) {
    assert(out_speaker[i] == out_ref[i]);
    assert(std::abs(out_speaker[i]) <= 0.95f); // No clipping
  }

  std::cout << "[PASS] test_downlink_pipeline" << std::endl;
}

int main() {
  test_drc_basic();
  test_downlink_pipeline();
  std::cout << "All Barrot VCP pipeline unit tests passed successfully!" << std::endl;
  return 0;
}
