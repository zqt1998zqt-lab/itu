import {
  FIVE_SIGNAL_MODES,
  type FiveSignalMode,
} from "./five-signal-flow-model.ts";
import type { BackgroundNoiseScene } from "./scoped-acoustic-model.ts";

export type SceneConditionCell =
  | "B24"
  | "B25"
  | "B26"
  | "B27"
  | "B28"
  | "B29"
  | "B30"
  | "B31"
  | "B32"
  | "B33"
  | "B34"
  | "B35"
  | "B36"
  | "B37"
  | "B38"
  | "B39"
  | "B40"
  | "B41"
  | "B42"
  | "B43";

export type SceneConditionItemId = `${FiveSignalMode}:${SceneConditionCell}`;

export type SceneConditionProfile = {
  scene: BackgroundNoiseScene;
  mode?: FiveSignalMode;
  sampleRateHz?: number;
  datasetId: string;
  datasetName: string;
  datasetKind:
    | "background_noise_idle_preview"
    | "background_noise_scene_preview"
    | "background_noise_conversation_preview";
  schemaVersion?: "scene-condition-preview-v1" | "scene-condition-preview-v2";
  caseCount?: number;
  speechCaseCount?: number;
  cngCaseCount?: number;
  observationCount?: number;
  conversationCounts?: Record<ConversationMode, number>;
  sceneLabel: string;
  contractItemIds: readonly [
    SceneConditionItemId,
    SceneConditionItemId,
    SceneConditionItemId,
    SceneConditionItemId,
    SceneConditionItemId,
  ];
};

const SCENE_CELLS: Record<
  BackgroundNoiseScene,
  readonly [
    SceneConditionCell,
    SceneConditionCell,
    SceneConditionCell,
    SceneConditionCell,
    SceneConditionCell,
  ]
> = {
  sce1: ["B24", "B25", "B26", "B27", "B28"],
  sce2: ["B29", "B30", "B31", "B32", "B33"],
  sce3: ["B34", "B35", "B36", "B37", "B38"],
  sce4: ["B39", "B40", "B41", "B42", "B43"],
};

function itemIdsFor(
  scene: BackgroundNoiseScene,
  mode: FiveSignalMode,
): readonly [
  SceneConditionItemId,
  SceneConditionItemId,
  SceneConditionItemId,
  SceneConditionItemId,
  SceneConditionItemId,
] {
  const [c0, c1, c2, c3, c4] = SCENE_CELLS[scene];
  return [
    `${mode}:${c0}`,
    `${mode}:${c1}`,
    `${mode}:${c2}`,
    `${mode}:${c3}`,
    `${mode}:${c4}`,
  ];
}

export const LEGACY_SCENE_CONDITION_PROFILES = {
  sce1: {
    scene: "sce1",
    mode: "WB-wired",
    datasetId: "edea6f9a51df87d0a4aefb2303dd8e1f",
    datasetName:
      "background-noise-sce1-wb-wired-20-condition-preview-v1",
    datasetKind: "background_noise_idle_preview",
    sceneLabel: "怠速 / 静止",
    contractItemIds: [
      "WB-wired:B24",
      "WB-wired:B25",
      "WB-wired:B26",
      "WB-wired:B27",
      "WB-wired:B28",
    ],
  },
  sce2: {
    scene: "sce2",
    mode: "WB-wired",
    datasetId: "d71bf7153f51f098fdead44059f65389",
    datasetName:
      "background-noise-sce2-wb-wired-20-condition-preview-v1",
    datasetKind: "background_noise_scene_preview",
    sceneLabel: "60 km/h 市区",
    contractItemIds: [
      "WB-wired:B29",
      "WB-wired:B30",
      "WB-wired:B31",
      "WB-wired:B32",
      "WB-wired:B33",
    ],
  },
  sce3: {
    scene: "sce3",
    mode: "WB-wired",
    datasetId: "73034293a0a2738f7b367e11676f16bf",
    datasetName:
      "background-noise-sce3-wb-wired-20-condition-preview-v1",
    datasetKind: "background_noise_scene_preview",
    sceneLabel: "120 km/h 高速",
    contractItemIds: [
      "WB-wired:B34",
      "WB-wired:B35",
      "WB-wired:B36",
      "WB-wired:B37",
      "WB-wired:B38",
    ],
  },
  sce4: {
    scene: "sce4",
    mode: "WB-wired",
    datasetId: "8783aac64dfaf4746e4798ec7c636e15",
    datasetName:
      "background-noise-sce4-wb-wired-20-condition-preview-v1",
    datasetKind: "background_noise_scene_preview",
    sceneLabel: "空调大风档",
    contractItemIds: [
      "WB-wired:B39",
      "WB-wired:B40",
      "WB-wired:B41",
      "WB-wired:B42",
      "WB-wired:B43",
    ],
  },
} as const satisfies Record<BackgroundNoiseScene, SceneConditionProfile>;

export const CONVERSATION_MODE_LABELS = {
  near_single: "近端单讲",
  far_single: "远端单讲",
  double_talk: "持续双讲",
  transition: "单讲 / 双讲切换",
  noise_only: "CNG 噪声观测",
} as const;
export type ConversationMode = keyof typeof CONVERSATION_MODE_LABELS;
export const CONVERSATION_MODES = Object.keys(CONVERSATION_MODE_LABELS) as ConversationMode[];

export const CONVERSATION_DATASET_IDS_BY_SCENE_AND_MODE: Record<
  BackgroundNoiseScene,
  Record<FiveSignalMode, string>
> = {
  sce1: {
    "NB-bt": "63cdcf3d85c0fc2921d33bd5134e0040",
    "NB-wired": "9a54fe8e537a4465c77ba8d9816a8aa5",
    "NB-wireless": "04c36654a2cf4263659165744445b779",
    "SWB-wired": "2734b9dd598958407482129759d2b084",
    "SWB-wireless": "d23816f19e6ca2c321a3cbf5c09e47c5",
    "WB-bt": "117d693b9a0e57e81872dc3ccec838e9",
    "WB-wired": "cf29475e343052eaffa73dfebad1b0b0",
    "WB-wireless": "5c2a785b3fcda0f4a58898280c6399bc",
  },
  sce2: {
    "NB-bt": "a6e54248f27dcfb56c4d763f914579d2",
    "NB-wired": "d1269cb2b82e0e4125464dc8127f4725",
    "NB-wireless": "c6531722512d5bbe1d215acee385b0b1",
    "SWB-wired": "14b49c91dff564434b6e420cfb8c2672",
    "SWB-wireless": "3106063f53b5b8052a804c2a0279ad39",
    "WB-bt": "8523151c73c203dd586db24d49b3d2e8",
    "WB-wired": "1f5451c6dbb3af6245dd7e6fb3d7338f",
    "WB-wireless": "09ec22a9d7e169aea589192d7b34fedd",
  },
  sce3: {
    "NB-bt": "b732505147ab4631d4c1b5ca3e63e584",
    "NB-wired": "0f76ed94cdf8eb90fa6b4073df2f64d0",
    "NB-wireless": "5882b0b11c9132fc927684866c842714",
    "SWB-wired": "f9e54460c56b052260e5ec20f180150f",
    "SWB-wireless": "ff864b4d23e6df9f97214dae65622adf",
    "WB-bt": "efab1281678b2e8ef35900eea8cdda24",
    "WB-wired": "093445541b9d5273419d844b18d209c8",
    "WB-wireless": "7b679e6f585a5c4e4f1f09a5fbb5eb97",
  },
  sce4: {
    "NB-bt": "7dc30ee3a8483225d4a26b2d1346479f",
    "NB-wired": "0394227334de08d73430312db8d7398d",
    "NB-wireless": "93e6dc3b3a63cff70371c29ccfc4a0d6",
    "SWB-wired": "989b5b158312aa121de2271f6206a9a8",
    "SWB-wireless": "cd111a1194d9ad8a94119012fe4cad4c",
    "WB-bt": "31ed5bb5da1ce0421a0ba8f295b7898b",
    "WB-wired": "f7f69b9900f87bea7546365b49d23af3",
    "WB-wireless": "d26302b1393546f71181b1c75f8a8d6e",
  },
};

export function sceneConditionSampleRateHz(
  mode: FiveSignalMode = "WB-wired",
): number {
  if (mode.startsWith("NB-")) return 8_000;
  if (mode.startsWith("SWB-")) return 48_000;
  return 16_000;
}

export const SCENE_MODE_CONDITION_PROFILES = Object.fromEntries(
  (Object.entries(LEGACY_SCENE_CONDITION_PROFILES) as [BackgroundNoiseScene, SceneConditionProfile][]).map(
    ([scene, legacy]) => [
      scene,
      Object.fromEntries(
        FIVE_SIGNAL_MODES.map((mode) => [
          mode,
          {
            ...legacy,
            scene,
            mode,
            sampleRateHz: sceneConditionSampleRateHz(mode),
            datasetId: CONVERSATION_DATASET_IDS_BY_SCENE_AND_MODE[scene][mode],
            datasetName: `background-noise-${scene}-${mode.toLowerCase()}-100-conversation-v2`,
            datasetKind: "background_noise_conversation_preview",
            schemaVersion: "scene-condition-preview-v2",
            caseCount: 100,
            speechCaseCount: 80,
            cngCaseCount: 20,
            observationCount: 280,
            conversationCounts: Object.fromEntries(
              CONVERSATION_MODES.map((convMode) => [convMode, 20]),
            ) as Record<ConversationMode, number>,
            contractItemIds: itemIdsFor(scene, mode),
          } satisfies SceneConditionProfile,
        ]),
      ) as Record<FiveSignalMode, SceneConditionProfile>,
    ],
  ),
) as Record<BackgroundNoiseScene, Record<FiveSignalMode, SceneConditionProfile>>;

export const SCENE_CONDITION_PROFILES = Object.fromEntries(
  (Object.keys(LEGACY_SCENE_CONDITION_PROFILES) as BackgroundNoiseScene[]).map(
    (scene) => [scene, SCENE_MODE_CONDITION_PROFILES[scene]["WB-wired"]],
  ),
) as Record<BackgroundNoiseScene, SceneConditionProfile>;

const ALL_SCENE_CONDITION_PROFILES: readonly SceneConditionProfile[] = [
  ...Object.values(SCENE_MODE_CONDITION_PROFILES).flatMap((byMode) =>
    Object.values(byMode),
  ),
  ...Object.values(LEGACY_SCENE_CONDITION_PROFILES),
];

export function sceneConditionCounts(profile: SceneConditionProfile) {
  return {
    cases: profile.caseCount ?? 20,
    speech: profile.speechCaseCount ?? 10,
    cng: profile.cngCaseCount ?? 10,
    observations: profile.observationCount ?? 50,
  };
}

export function conversationModeOf(value: { provenance?: Record<string, unknown> }): ConversationMode | null {
  const mode = value.provenance?.conversation_mode;
  return typeof mode === "string" && Object.hasOwn(CONVERSATION_MODE_LABELS, mode)
    ? mode as ConversationMode : null;
}

export const SCENE_CONDITION_ITEM_IDS: readonly SceneConditionItemId[] =
  Object.values(SCENE_CONDITION_PROFILES).flatMap(
    (profile) => profile.contractItemIds,
  );

export function sceneConditionProfile(
  scene: BackgroundNoiseScene,
  mode: FiveSignalMode = "WB-wired",
): SceneConditionProfile {
  return (
    SCENE_MODE_CONDITION_PROFILES[scene]?.[mode]
    ?? SCENE_CONDITION_PROFILES[scene]
  );
}

export function sceneConditionProfileForDatasetId(
  datasetId: string,
): SceneConditionProfile | null {
  return (
    ALL_SCENE_CONDITION_PROFILES.find(
      (profile) => profile.datasetId === datasetId,
    ) ?? null
  );
}

// Register new immutable dataset identities while preserving historical lookup.
import { registerSceneDatasetRevision } from "./register-scene-dataset-revision.ts";
registerSceneDatasetRevision(
  SCENE_MODE_CONDITION_PROFILES,
  ALL_SCENE_CONDITION_PROFILES,
  SCENE_CONDITION_PROFILES,
  false,
);

import datasetRevisionR2 from "./scene-dataset-revision-r2-ids.json" with { type: "json" };
registerSceneDatasetRevision(
  SCENE_MODE_CONDITION_PROFILES,
  ALL_SCENE_CONDITION_PROFILES,
  SCENE_CONDITION_PROFILES,
  false,
  datasetRevisionR2,
);

// R2 was validated in all 48 stores before publishing ACTIVATION-r2.json.
registerSceneDatasetRevision(
  SCENE_MODE_CONDITION_PROFILES,
  ALL_SCENE_CONDITION_PROFILES,
  SCENE_CONDITION_PROFILES,
  true,
  datasetRevisionR2,
);
