import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import {
  chainCommand, chainFlags, chainPresets, chainEvidencePresets, chainSelections, chainStages, changeStage, matchingEvidence,
  parseChainReport, restoreConfiguration,
} from "../src/features/enhancement-chain-model.ts";

const report = () => parseChainReport(JSON.parse(readFileSync(new URL("../public/evaluation/enhancement-chain/report.json", import.meta.url), "utf8")));
const serial = chainPresets.find((item) => item.id === "all_serial")!;

test("every published route matches the exact selectable CLI flags and a complete paired matrix", () => {
  const actual = report();
  assert.equal(actual.case_count, 72);
  assert.equal(actual.cli_runs, 72 * chainEvidencePresets.length);
  for (const preset of chainEvidencePresets) {
    const evidence = matchingEvidence(preset, actual);
    assert.ok(evidence, preset.id);
    assert.deepEqual(evidence.flags, chainFlags(preset));
    assert.equal(evidence.caseGroups.length, 2);
    assert.deepEqual(new Set(evidence.caseGroups.map((group) => group.id)), new Set(["pass", "fail"]));
    for (const stage of chainStages) assert.ok(stage.options.some((option) => option.id === preset.selection[stage.id]));
  }
  const recommended = chainPresets.find((item) => item.recommended);
  assert.equal(recommended?.id, "super_gaussian_optimal");
  assert.equal(matchingEvidence(recommended!, actual)?.pass, 60);
});

test("every algorithm cluster exposes its seven-stage actual execution selection", () => {
  for (const preset of chainPresets) {
    const selections = chainSelections(preset);
    assert.equal(selections.length, 7, preset.id);
    assert.deepEqual(
      selections.map(({ stage }) => stage.id),
      ["bf", "aec", "ns", "eq", "agc", "drc", "limiter"],
    );
    assert.ok(
      selections.every(({ stage, option }) =>
        Boolean(option) &&
        stage.options.some((candidate) => candidate.id === option?.id),
      ),
      preset.id,
    );
  }

  const unified = chainPresets.find((preset) => preset.id === "vcp_unified")!;
  const unifiedStages = Object.fromEntries(
    chainSelections(unified).map(({ stage, option }) => [stage.id, option?.id]),
  );
  assert.deepEqual(unifiedStages, {
    bf: "das",
    aec: "speex_fused_cpu",
    ns: "unified",
    eq: "off",
    agc: "guarded5",
    drc: "default",
    limiter: "on",
  });

  const missing = chainSelections({
    profile: "vcp",
    selection: { ...unified.selection, aec: "not-a-real-backend" },
  });
  assert.equal(
    missing.find(({ stage }) => stage.id === "aec")?.option,
    null,
  );
});

test("changing a real algorithm invalidates the old scores and case figures", () => {
  const actual = report();
  assert.equal(matchingEvidence(serial, actual)?.id, "all_serial");
  const draft = changeStage(serial, "ns", "super_gaussian");
  assert.equal(matchingEvidence(draft, actual), undefined);
  assert.ok(chainFlags(draft).includes("super_gaussian"));
  assert.equal(matchingEvidence(changeStage(draft, "ns", "webrtc"), actual)?.id, "all_serial");
});

test("frozen CPU AEC evidence is canonicalized to the renamed backend", () => {
  const actual = report();
  for (const id of ["vcp_unified", "vcp_unified_eq_fused"]) {
    const variant = actual.variants.find((item) => item.id === id);
    assert.ok(variant);
    assert.equal(variant.selection.aec, "speex_fused_cpu");
    const backendIndex = variant.flags.indexOf("--aec-backend");
    assert.ok(backendIndex >= 0);
    assert.equal(variant.flags[backendIndex + 1], "speex_fused_cpu");
  }
});

test("CPU and HiFi DSP fused AEC routes are distinct catalog requests without frozen evidence", () => {
  const actual = report();
  const cpu = chainPresets.find((item) => item.id === "speex_fused_cpu")!;
  const dsp = chainPresets.find((item) => item.id === "speex_fused_dsp")!;
  const expected = {
    bf: "das", aec: "", ns: "webrtc", eq: "off", agc: "guarded5", drc: "default", limiter: "on",
  };
  assert.equal(cpu.profile, "vcp");
  assert.equal(dsp.profile, "vcp");
  assert.deepEqual(cpu.selection, { ...expected, aec: "speex_fused_cpu" });
  assert.deepEqual(dsp.selection, { ...expected, aec: "speex_fused_dsp" });
  const dspOption = chainStages.find((stage) => stage.id === "aec")!.options.find((option) => option.id === "speex_fused_dsp")!;
  assert.deepEqual(dspOption.flags, ["--aec-backend", "speex_fused_dsp"]);
  assert.deepEqual(dspOption.requires, ["hifi_dsp", "complex_pf_model"]);
  assert.equal(dspOption.candidate, true);
  assert.notDeepEqual(chainFlags(cpu), chainFlags(dsp));
  assert.ok(chainCommand(cpu).includes("'speex_fused_cpu'"));
  assert.ok(!chainCommand(cpu).includes("'speex_fused_dsp'"));
  assert.ok(chainCommand(dsp).includes("'speex_fused_dsp'"));
  assert.ok(!chainCommand(dsp).includes("'speex_fused_cpu'"));
  assert.ok(!chainEvidencePresets.some((preset) => preset.id === "speex_fused_cpu" || preset.id === "speex_fused_dsp"));
  assert.equal(matchingEvidence(cpu, actual), undefined);
  assert.equal(matchingEvidence(dsp, actual), undefined);
});

test("Complex_PF CPU is a fifth offline experimental route without frozen evidence", () => {
  const actual = report();
  const legacyCpu = chainPresets.find((item) => item.id === "speex_fused_cpu")!;
  const complexCpu = chainPresets.find((item) => item.id === "computer_pf_cpu")!;
  const option = chainStages.find((stage) => stage.id === "aec")!.options.find((item) => item.id === "computer_pf_cpu")!;
  assert.equal(complexCpu.profile, "vcp");
  assert.deepEqual(complexCpu.selection, { ...legacyCpu.selection, aec: "computer_pf_cpu" });
  assert.deepEqual(option.flags, ["--aec-backend", "computer_pf_cpu"]);
  assert.deepEqual(option.requires, ["complex_pf_model"]);
  assert.equal(option.candidate, true);
  assert.match(option.label, /CPU/);
  assert.match(option.label, /离线实验/);
  assert.match(option.note, /本地训练/);
  assert.match(option.note, /不是官方预训练权重/);
  assert.match(option.note, /不保证优于现有 CPU/);
  assert.ok(chainCommand(complexCpu).includes("'computer_pf_cpu'"));
  assert.ok(!chainCommand(complexCpu).includes("'speex_fused_cpu'"));
  assert.ok(!chainCommand(complexCpu).includes("'speex_fused_dsp'"));
  assert.ok(!chainEvidencePresets.some((preset) => preset.id === "computer_pf_cpu"));
  assert.equal(matchingEvidence(complexCpu, actual), undefined);

  const restored = restoreConfiguration(JSON.stringify({
    ...complexCpu,
    extra_flags: ["--aec-backend", "speex_fused_cpu"],
  }));
  assert.equal(restored.profile, "vcp");
  assert.deepEqual(restored.selection, complexCpu.selection);
  assert.equal(restored.selection.aec, "computer_pf_cpu");
  assert.equal(restored.extra_flags, undefined);
  assert.ok(chainFlags(restored).includes("computer_pf_cpu"));
  assert.ok(!chainFlags(restored).includes("speex_fused_cpu"));
});

test("LocalVQE 203K is a real VCP AEC route without borrowed evidence", () => {
  const actual = report();
  const preset = chainPresets.find((item) => item.id === "localvqe_aec200k")!;
  const option = chainStages
    .find((stage) => stage.id === "aec")!
    .options.find((item) => item.id === "localvqe_aec200k")!;
  assert.equal(preset.profile, "vcp");
  assert.deepEqual(preset.selection, {
    bf: "das",
    aec: "localvqe_aec200k",
    ns: "webrtc",
    eq: "three_band",
    agc: "guarded5",
    drc: "default",
    limiter: "on",
  });
  assert.deepEqual(option.flags, ["--aec-backend", "localvqe_aec200k"]);
  assert.deepEqual(option.requires, ["localvqe_cpu", "localvqe_aec200k"]);
  assert.equal(option.candidate, true);
  assert.deepEqual(preset.extra_flags, [
    "--localvqe-reference-channel",
    "0",
  ]);
  assert.match(option.label, /LocalVQE/);
  assert.match(option.note, /DAF/);
  assert.match(option.note, /不是正式 ITU-T 合规/);
  assert.ok(chainCommand(preset).includes("'localvqe_aec200k'"));
  assert.ok(chainCommand(preset).includes("--localvqe-reference-channel"));
  assert.ok(chainCommand(preset).includes("'0'"));
  assert.ok(!chainCommand(preset).includes("'speex_fused_cpu'"));
  assert.ok(!chainEvidencePresets.some((item) => item.id === "localvqe_aec200k"));
  assert.equal(matchingEvidence(preset, actual), undefined);

  const restored = restoreConfiguration(JSON.stringify({
    ...preset,
    extra_flags: ["--aec-backend", "speex_fused_cpu"],
  }));
  assert.equal(restored.profile, "vcp");
  assert.deepEqual(restored.selection, preset.selection);
  assert.deepEqual(restored.extra_flags, preset.extra_flags);
  assert.ok(chainFlags(restored).includes("localvqe_aec200k"));
});

test("server-executed NS clusters keep the LocalVQE AEC tail and never expose boundary capture as a full CLI", () => {
  const actual = report();
  const base = chainPresets.find((item) => item.id === "localvqe_aec200k")!;
  const nsStage = chainStages.find((stage) => stage.id === "ns")!;
  const routes = [
    {
      ns: "cofilite",
      label: "CoFi-Lite · ONNX CPU 流式",
      presetId: "localvqe_cofilite",
      presetLabel: "VCP · LocalVQE 203K AEC + CoFi-Lite · ONNX CPU 流式",
    },
    {
      ns: "dpdfnet2",
      label: "DPDFNet-2 · CPU",
      presetId: "localvqe_dpdfnet2",
      presetLabel: "VCP · LocalVQE 203K AEC + DPDFNet-2 · CPU",
    },
    {
      ns: "hybrid_v2_pitch",
      label: "V2＋谐波 · CPU",
      presetId: "localvqe_hybrid_v2_pitch",
      presetLabel: "VCP · LocalVQE 203K AEC + V2＋谐波 · CPU",
    },
  ];

  for (const route of routes) {
    const option = nsStage.options.find((item) => item.id === route.ns)!;
    const preset = chainPresets.find((item) => item.id === route.presetId)!;
    assert.equal(option.label, route.label);
    assert.deepEqual(option.flags, ["--ns-backend", "webrtc"]);
    assert.equal(option.candidate, true);
    if (route.ns === "cofilite") {
      assert.match(option.note, /16 kHz CPU 流式降噪/);
      assert.match(option.note, /原始 CoFi-Lite FP32 权重/);
      assert.match(option.note, /ONNX Runtime 单线程执行/);
      assert.match(option.note, /状态跨 10 ms 回调保留/);
      assert.match(option.note, /30 ms/);
      assert.doesNotMatch(option.note, /整段.*离线/);
    } else {
      assert.match(option.note, /16 kHz CPU 文件降噪/);
    }
    if (route.ns === "dpdfnet2") {
      assert.match(option.note, /DPDFNet-2 流式模型/);
      assert.match(option.note, /固定延时 40 ms/);
    }
    if (route.ns === "hybrid_v2_pitch") {
      assert.match(option.note, /OM-LSA \+ V2 增益控制 \+ 谐波保护/);
      assert.match(option.note, /固定延时 10 ms/);
    }

    assert.equal(preset.label, route.presetLabel);
    assert.equal(preset.profile, "vcp");
    assert.equal(preset.candidate, true);
    assert.equal(preset.recommended, undefined);
    assert.deepEqual(preset.selection, { ...base.selection, ns: route.ns });
    assert.deepEqual(preset.extra_flags, base.extra_flags);
    assert.deepEqual(chainFlags(preset), chainFlags(base));
    assert.ok(chainFlags(preset).includes("webrtc"));
    assert.ok(!chainEvidencePresets.some((item) => item.id === preset.id));
    assert.equal(matchingEvidence(preset, actual), undefined);

    const command = chainCommand(preset);
    assert.match(command, /服务器 wrapper/);
    if (route.ns === "cofilite") {
      assert.match(command, /持久模型进程/);
      assert.match(command, /160 样本在线处理/);
      assert.doesNotMatch(command, /--post-ns-pcm/);
    } else {
      assert.match(command, /--post-ns-pcm/);
    }
    assert.match(command, /请使用界面运行此算法簇/);
    assert.doesNotMatch(command, /--ns-backend|'webrtc'/);

    const restored = restoreConfiguration(JSON.stringify({
      ...preset,
      extra_flags: ["--ns-backend", "webrtc"],
    }));
    assert.equal(restored.profile, "vcp");
    assert.deepEqual(restored.selection, preset.selection);
    assert.deepEqual(restored.extra_flags, base.extra_flags);
  }

  assert.equal(chainCommand(base), "build/host/cpp/itu_audio_cli \\\n" +
    "  --profile 'vcp' --no-rx-processing --aec-backend 'localvqe_aec200k' --ns-backend 'webrtc' " +
    "--eq-preamp-db '-2' --eq-band 'low_shelf,180,-2,0.707' --eq-band 'peak,1800,2,1' " +
    "--eq-band 'high_shelf,4500,-1,0.707' " +
    "--tx-drc --localvqe-reference-channel '0' --mic 'capture.wav' --echo 'actual-playback.wav' " +
    "--beam-delays '<逐通道标定延迟，以逗号分隔>' --out 'build/chain-replay/result'\n");
});

test("LocalVQE defaults enable the same three-band EQ in the summary and execution flags", () => {
  for (const preset of chainPresets.filter((item) => item.selection.aec === "localvqe_aec200k")) {
    const eq = chainSelections(preset).find(({ stage }) => stage.id === "eq")!.option!;
    assert.equal(eq.id, "three_band", preset.id);
    assert.equal(eq.label, "三段语音形状", preset.id);
    const flags = chainFlags(preset);
    assert.equal(flags[flags.indexOf("--eq-preamp-db") + 1], "-2", preset.id);
    assert.deepEqual(
      flags.flatMap((flag, index) => flag === "--eq-band" ? [flags[index + 1]] : []),
      ["low_shelf,180,-2,0.707", "peak,1800,2,1", "high_shelf,4500,-1,0.707"],
      preset.id,
    );
    const bypass = changeStage(preset, "eq", "off");
    assert.equal(chainSelections(bypass).find(({ stage }) => stage.id === "eq")!.option!.id, "off");
    assert.ok(!chainFlags(bypass).includes("--eq-band"));
    assert.ok(!chainFlags(bypass).includes("--eq-preamp-db"));
    assert.equal(matchingEvidence(preset, report()), undefined);
  }
});

test("DSP drafts round-trip without becoming CPU and trusted unified profiles are restored", () => {
  const dsp = chainPresets.find((item) => item.id === "speex_fused_dsp")!;
  const restoredDsp = restoreConfiguration(JSON.stringify({
    ...dsp,
    extra_flags: ["--aec-backend", "speex_fused_cpu"],
  }));
  assert.equal(restoredDsp.profile, "vcp");
  assert.deepEqual(restoredDsp.selection, dsp.selection);
  assert.equal(restoredDsp.selection.aec, "speex_fused_dsp");
  assert.ok(chainFlags(restoredDsp).includes("speex_fused_dsp"));
  assert.ok(!chainFlags(restoredDsp).includes("speex_fused_cpu"));
  assert.equal(restoredDsp.extra_flags, undefined);

  const unified = chainPresets.find((item) => item.id === "vcp_unified")!;
  const restoredUnified = restoreConfiguration(JSON.stringify({ ...unified, extra_flags: ["--no-aec"] }));
  assert.equal(restoredUnified.profile, "vcp_unified");
  assert.deepEqual(restoredUnified.selection, unified.selection);
  assert.equal(restoredUnified.extra_flags, undefined);
});

test("same route name with different executor flags cannot borrow frozen evidence", () => {
  const actual = report();
  actual.variants.find((item) => item.id === "all_serial")!.flags.push("--no-agc");
  assert.equal(matchingEvidence(serial, actual), undefined);
});

test("invalid browser drafts and injected cached flags never alter the executor template", () => {
  assert.equal(restoreConfiguration("{malformed").selection.eq, serial.selection.eq);
  assert.equal(restoreConfiguration(JSON.stringify({ profile: "vcp", selection: { ...serial.selection, ns: "missing-backend" } })).selection.ns, serial.selection.ns);
  const restored = restoreConfiguration(JSON.stringify({ ...serial, extra_flags: ["--no-agc"] }));
  assert.ok(!chainFlags(restored).includes("--no-agc"));
  assert.equal(matchingEvidence(restored, report())?.id, "all_serial");
});

test("editing the DAS control creates a VCP draft with no hidden control-only flags", () => {
  const draft = changeStage(chainPresets[0], "ns", "webrtc");
  assert.equal(draft.profile, "vcp");
  assert.ok(!chainFlags(draft).includes("--no-low-cut"));
  assert.equal(matchingEvidence(draft, report()), undefined);
});

test("partial pilot reports and missing negative controls are rejected", () => {
  const partial = report();
  partial.case_count = 2;
  assert.throws(() => parseChainReport(partial), /不完整/);
  const vacuous = report();
  vacuous.negative_control.rejected = 0;
  assert.throws(() => parseChainReport(vacuous), /负对照/);
});

test("joint route is executable without borrowing the older matrix", () => {
  const joint = chainPresets.find((item) => item.id === "vcp_joint")!;
  assert.equal(joint.profile, "vcp_joint");
  assert.equal(chainFlags(joint)[1], "vcp_joint");
  assert.equal(matchingEvidence(joint, report()), undefined);
  const restored = restoreConfiguration(JSON.stringify({ ...joint, extra_flags: ["--no-aec"] }));
  assert.equal(restored.profile, "vcp_joint");
  assert.deepEqual(restored.selection, joint.selection);
  assert.ok(!chainFlags(restored).includes("--no-aec"));
  assert.equal(changeStage(restored, "eq", "three_band").profile, "vcp_joint");
  assert.equal(chainFlags(restored).at(chainFlags(restored).lastIndexOf("--tx-drc-threshold") + 1), "-6");
});
