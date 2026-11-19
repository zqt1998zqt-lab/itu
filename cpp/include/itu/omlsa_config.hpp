#pragma once

namespace itu {
// Opt-in causal OM-LSA experiment. Defaults preserve the original estimator
// and synthesis path exactly; the gain floor selected by noise_level is intact.
struct OmlsaTuningConfig {
  float noise_update_scale = 1.f;
  float speech_recovery = 0.f;
  float release_ms = 60.f;
  // -40 disables the additional evidence-qualified synthesis floor.
  float speech_floor_db = -40.f;
  // Opt-in adaptive OM-LSA experiments. All defaults preserve the legacy path.
  bool adaptive_noise_tracking = false;
  bool high_snr_transparency = false;
  bool low_band_protection = false;
};
}
