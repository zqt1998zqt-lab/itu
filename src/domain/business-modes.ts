// Service reference rates used by the acceptance datasets. A selected mode
// does not resample a case: its WAV rate and the chosen algorithm set the PCM rate.
const BUSINESS_MODE_PRESENTATION = {
  "NB-wired": { label: "窄带 · 有线", referenceRateHz: 8_000 },
  "NB-wireless": { label: "窄带 · 无线", referenceRateHz: 8_000 },
  "NB-bt": { label: "窄带 · 蓝牙", referenceRateHz: 8_000 },
  "WB-wired": { label: "宽带 · 有线", referenceRateHz: 16_000 },
  "WB-wireless": { label: "宽带 · 无线", referenceRateHz: 16_000 },
  "WB-bt": { label: "宽带 · 蓝牙", referenceRateHz: 16_000 },
  "SWB-wired": { label: "超宽带 · 有线", referenceRateHz: 48_000 },
  "SWB-wireless": { label: "超宽带 · 无线", referenceRateHz: 48_000 },
} as const;

type BusinessMode = keyof typeof BUSINESS_MODE_PRESENTATION;
export type BusinessSampleRateHz = 8_000 | 16_000 | 48_000;

export function businessModeReferenceRateHz(mode: string): BusinessSampleRateHz | null {
  if (!Object.hasOwn(BUSINESS_MODE_PRESENTATION, mode)) return null;
  return BUSINESS_MODE_PRESENTATION[mode as BusinessMode].referenceRateHz;
}

export function businessModeReferenceRateLabel(mode: string): string {
  const rate = businessModeReferenceRateHz(mode);
  if (rate === null) return "未定义";
  return `${rate / 1000} kHz`;
}

export function businessModeLabel(mode: string): string {
  if (!Object.hasOwn(BUSINESS_MODE_PRESENTATION, mode)) return mode;
  const info = BUSINESS_MODE_PRESENTATION[mode as BusinessMode];
  return `${info.label} · ${businessModeReferenceRateLabel(mode)} 参考 · ${mode}`;
}
