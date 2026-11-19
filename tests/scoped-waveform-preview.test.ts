import assert from "node:assert/strict";
import { test } from "node:test";

import {
  readSceneConditionPreview,
  readSceneConditionWaveformPreview,
  SCENE_CONDITION_SCHEMA_VERSION,
  type ScopedInputSignalRole,
  type ScopedPreviewWaveformTrack,
  type ScopedSceneConditionWaveformPreview,
} from "../src/features/scoped-acoustic-model.ts";
import {
  CONVERSATION_MODES,
  LEGACY_SCENE_CONDITION_PROFILES,
  SCENE_CONDITION_PROFILES,
} from "../src/features/scene-condition-profiles.ts";

const SCENE = "sce1" as const;
const PROFILE = SCENE_CONDITION_PROFILES[SCENE];
const LEGACY_PROFILE = LEGACY_SCENE_CONDITION_PROFILES[SCENE];
const ROLES = [
  "microphone",
  "reference",
  "far_end",
  "noise",
] as const satisfies readonly ScopedInputSignalRole[];
const WAVE = "A".repeat(1_920);
const MEL = "A".repeat(25_600);

function hash(seed: number): string {
  return (seed % 256).toString(16).padStart(2, "0").repeat(32);
}

function waveformPreviewFixture(): Record<string, any> {
  const cases: Record<string, any>[] = [];
  for (const [modeIndex, conversationMode] of CONVERSATION_MODES.entries()) {
    const noiseOnly = conversationMode === "noise_only";
    for (let offset = 0; offset < 20; offset += 1) {
      const sequence = modeIndex * 20 + offset;
      const hashes = Object.fromEntries(
        ROLES.map((role, roleIndex) => [
          role,
          hash(sequence * ROLES.length + roleIndex + 1),
        ]),
      );
      cases.push({
        case_id:
          SCENE + "-" + conversationMode + "-" +
          String(offset + 1).padStart(2, "0"),
        conditions: {
          mode: "WB-wired",
          direction: "send",
          scene: SCENE,
          volume: "nominal",
          window_ms: null,
          input_level_db: 0,
        },
        families: noiseOnly
          ? ["cng_level", "cng_spectrum"]
          : ["g_mos", "n_mos", "s_mos"],
        signals: [...ROLES],
        signals_sha256: hashes,
        contract_item_ids: noiseOnly
          ? PROFILE.contractItemIds.slice(3)
          : PROFILE.contractItemIds.slice(0, 3),
        provenance: {
          engineering_preview: true,
          actual_vehicle_recording: false,
          conversation_mode: conversationMode,
        },
      });
    }
  }
  const visualCases = Object.fromEntries(
    cases.map((previewCase) => [
      previewCase.case_id,
      Object.fromEntries(
        ROLES.map((role) => [
          role,
          {
            source_sha256: previewCase.signals_sha256[role],
            duration_s: 12,
            sample_rate: 16_000,
            channels: role === "microphone" ? 2 : 1,
            peak: 0.25,
            wave_scale: 1,
            wave_min_i16: WAVE,
            wave_max_i16: WAVE,
          },
        ]),
      ),
    ]),
  );
  return {
    dataset_id: PROFILE.datasetId,
    dataset_name: PROFILE.datasetName,
    dataset_manifest_sha256: "d".repeat(64),
    dataset_assets_sha256: "e".repeat(64),
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: SCENE,
    cases,
    visuals: {
      schema: "scoped-acoustic-visual-v1",
      waveform_bins: 720,
      mel_bands: 40,
      mel_frames: 240,
      mel_fft_size: 512,
      mel_hop_ms: 10,
      mel_dynamic_range_db: 80,
      mel_db_offset_db: -200,
      mel_db_step_db: 0.1,
      detail: "waveform",
      cases: visualCases,
    },
    visual_detail: "waveform",
    preview_condition_group_count: 100,
    contract_item_count: 5,
    observation_count: 280,
    conversation_counts: { ...PROFILE.conversationCounts },
    standard_compliance: false,
    formal_test_status: "not_measured",
    dataset_kind: PROFILE.datasetKind,
    schema_version: PROFILE.schemaVersion,
  };
}

function withFullVisuals(value: Record<string, any>): Record<string, any> {
  delete value.visual_detail;
  delete value.visuals.detail;
  for (const tracks of Object.values(value.visuals.cases) as Record<
    string,
    any
  >[]) {
    for (const role of ROLES) {
      tracks[role].mel_i16 = MEL;
      tracks[role].mel_ceiling_db = -5.5;
    }
  }
  return value;
}

function fullPreviewFixture(): Record<string, any> {
  return withFullVisuals(waveformPreviewFixture());
}

function legacyWaveformPreviewFixture(): Record<string, any> {
  const cases: Record<string, any>[] = [];
  for (let index = 0; index < 20; index += 1) {
    const noiseOnly = index >= 10;
    const caseId = "legacy-preview-" + String(index + 1).padStart(2, "0");
    const hashes = Object.fromEntries(
      ROLES.map((role, roleIndex) => [
        role,
        hash(1_000 + index * ROLES.length + roleIndex),
      ]),
    );
    cases.push({
      case_id: caseId,
      conditions: {
        mode: "WB-wired",
        direction: "send",
        scene: SCENE,
        volume: "nominal",
        window_ms: null,
        input_level_db: 0,
      },
      families: noiseOnly
        ? ["cng_level", "cng_spectrum"]
        : ["g_mos", "n_mos", "s_mos"],
      signals: [...ROLES],
      signals_sha256: hashes,
      contract_item_ids: noiseOnly
        ? LEGACY_PROFILE.contractItemIds.slice(3)
        : LEGACY_PROFILE.contractItemIds.slice(0, 3),
      provenance: {
        engineering_preview: true,
        actual_vehicle_recording: false,
      },
    });
  }
  const visualCases = Object.fromEntries(
    cases.map((previewCase) => [
      previewCase.case_id,
      Object.fromEntries(
        ROLES.map((role) => [
          role,
          {
            source_sha256: previewCase.signals_sha256[role],
            duration_s: 12,
            sample_rate: 16_000,
            channels: role === "microphone" ? 2 : 1,
            peak: 0.25,
            wave_scale: 1,
            wave_min_i16: WAVE,
            wave_max_i16: WAVE,
          },
        ]),
      ),
    ]),
  );
  return {
    dataset_id: LEGACY_PROFILE.datasetId,
    dataset_name: LEGACY_PROFILE.datasetName,
    dataset_manifest_sha256: "d".repeat(64),
    dataset_assets_sha256: "e".repeat(64),
    test_scope: "background_noise",
    mode: "WB-wired",
    scene: SCENE,
    cases,
    visuals: {
      schema: "scoped-acoustic-visual-v1",
      waveform_bins: 720,
      mel_bands: 40,
      mel_frames: 240,
      mel_fft_size: 512,
      mel_hop_ms: 10,
      mel_dynamic_range_db: 80,
      mel_db_offset_db: -200,
      mel_db_step_db: 0.1,
      detail: "waveform",
      cases: visualCases,
    },
    visual_detail: "waveform",
    preview_condition_group_count: 20,
    contract_item_count: 5,
    observation_count: 50,
    conversation_counts: {},
    standard_compliance: false,
    formal_test_status: "not_measured",
    dataset_kind: LEGACY_PROFILE.datasetKind,
    schema_version: SCENE_CONDITION_SCHEMA_VERSION,
  };
}

function legacyFullPreviewFixture(): Record<string, any> {
  return withFullVisuals(legacyWaveformPreviewFixture());
}

function readWaveform(value: Record<string, any>) {
  return readSceneConditionWaveformPreview(
    value,
    value.dataset_id,
    "WB-wired",
    SCENE,
  );
}

function readFull(value: Record<string, any>) {
  return readSceneConditionPreview(
    value,
    value.dataset_id,
    "WB-wired",
    SCENE,
  );
}

test("waveform preview parses 100 cases with lightweight far-end tracks", () => {
  const parsed: ScopedSceneConditionWaveformPreview = readWaveform(
    waveformPreviewFixture(),
  );
  assert.equal(parsed.cases.length, 100);
  assert.equal(parsed.preview_condition_group_count, 100);
  assert.equal(parsed.visual_detail, "waveform");
  assert.equal(parsed.visuals.detail, "waveform");
  assert.equal(Object.keys(parsed.visuals.cases).length, 100);

  const previewCase = parsed.cases[0];
  const farEnd: ScopedPreviewWaveformTrack =
    parsed.visuals.cases[previewCase.case_id].far_end;
  assert.equal(farEnd.duration_s, 12);
  assert.equal(farEnd.sample_rate, 16_000);
  assert.equal(farEnd.channels, 1);
  assert.equal(Object.hasOwn(farEnd, "mel_i16"), false);
  assert.equal(Object.hasOwn(farEnd, "mel_ceiling_db"), false);
  assert.deepEqual(
    Object.keys(farEnd).sort(),
    [
      "channels",
      "duration_s",
      "peak",
      "sample_rate",
      "source_sha256",
      "wave_max_i16",
      "wave_min_i16",
      "wave_scale",
    ],
  );
});

test("full preview reader still requires mel data while accepting full tracks", () => {
  const lightweight = waveformPreviewFixture();
  assert.throws(
    () =>
      readSceneConditionPreview(
        lightweight,
        lightweight.dataset_id,
        "WB-wired",
        SCENE,
      ),
    /梅尔频谱/,
  );

  const full = fullPreviewFixture();
  const parsed = readSceneConditionPreview(
    full,
    full.dataset_id,
    "WB-wired",
    SCENE,
  );
  const track = parsed.visuals.cases[parsed.cases[0].case_id].far_end;
  assert.equal(track.mel_i16, MEL);
  assert.equal(track.mel_ceiling_db, -5.5);
});

test("waveform preview keeps scene metadata, SHA, provenance, and visual markers strict", () => {
  const mutations: Array<[string, (value: Record<string, any>) => void]> = [
    [
      "asset SHA",
      (value) => {
        value.dataset_assets_sha256 = "not-a-sha256";
      },
    ],
    [
      "signal SHA",
      (value) => {
        value.cases[0].signals_sha256.microphone = "not-a-sha256";
      },
    ],
    [
      "source SHA",
      (value) => {
        value.visuals.cases[value.cases[0].case_id].far_end.source_sha256 =
          "f".repeat(64);
      },
    ],
    [
      "metadata",
      (value) => {
        value.schema_version = "scene-condition-preview-v1";
      },
    ],
    [
      "provenance",
      (value) => {
        value.cases[0].provenance.actual_vehicle_recording = true;
      },
    ],
    [
      "missing waveform",
      (value) => {
        delete value.visuals.cases[value.cases[0].case_id].far_end.wave_min_i16;
      },
    ],
    [
      "waveform bin length",
      (value) => {
        value.visuals.cases[value.cases[0].case_id].far_end.wave_max_i16 =
          "A".repeat(1_916);
      },
    ],
    [
      "missing visuals",
      (value) => {
        delete value.visuals;
      },
    ],
    [
      "missing visual track",
      (value) => {
        delete value.visuals.cases[value.cases[0].case_id].noise;
      },
    ],
    [
      "unknown case",
      (value) => {
        value.visuals.cases.unknown = value.visuals.cases[value.cases[0].case_id];
      },
    ],
    [
      "99 cases",
      (value) => {
        const removed = value.cases.shift();
        delete value.visuals.cases[removed.case_id];
        value.preview_condition_group_count = 99;
      },
    ],
    [
      "top-level waveform marker",
      (value) => {
        delete value.visual_detail;
      },
    ],
    [
      "nested waveform marker",
      (value) => {
        delete value.visuals.detail;
      },
    ],
    [
      "mel leakage",
      (value) => {
        value.visuals.cases[value.cases[0].case_id].far_end.mel_i16 = MEL;
      },
    ],
  ];

  for (const [label, mutate] of mutations) {
    const value = waveformPreviewFixture();
    mutate(value);
    assert.throws(() => readWaveform(value), Error, label);
  }
});

test("legacy preview accepts missing or strict empty conversation counts", () => {
  for (const [label, make, read] of [
    ["legacy light empty", legacyWaveformPreviewFixture, readWaveform],
    ["legacy full empty", legacyFullPreviewFixture, readFull],
  ] as const) {
    const parsed = read(make());
    assert.deepEqual(parsed.conversation_counts, {}, label);
  }

  for (const [label, make, read] of [
    ["legacy light missing", legacyWaveformPreviewFixture, readWaveform],
    ["legacy full missing", legacyFullPreviewFixture, readFull],
  ] as const) {
    const value = make();
    delete value.conversation_counts;
    const parsed = read(value);
    assert.equal(parsed.conversation_counts, undefined, label);
  }
});

test("legacy preview rejects non-empty or fabricated conversation counts", () => {
  for (const [label, make, read] of [
    ["legacy light counts", legacyWaveformPreviewFixture, readWaveform],
    ["legacy full counts", legacyFullPreviewFixture, readFull],
  ] as const) {
    const value = make();
    value.conversation_counts = { near_single: 20 };
    assert.throws(() => read(value), Error, label);
  }
});

test("v2 preview still rejects missing or empty conversation counts", () => {
  for (const [label, make, read] of [
    ["v2 light missing", waveformPreviewFixture, readWaveform],
    ["v2 full missing", fullPreviewFixture, readFull],
  ] as const) {
    const value = make();
    delete value.conversation_counts;
    assert.throws(() => read(value), Error, label);
  }

  for (const [label, make, read] of [
    ["v2 light empty", waveformPreviewFixture, readWaveform],
    ["v2 full empty", fullPreviewFixture, readFull],
  ] as const) {
    const value = make();
    value.conversation_counts = {};
    assert.throws(() => read(value), Error, label);
  }
});
