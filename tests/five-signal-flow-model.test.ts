import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import type { NV200SheetData, NV200TestItem } from "../src/domain/nv200-table.ts";
import {
  buildDnsmosMapping,
  buildMetricMapping,
  buildReceiveProxyMapping,
  FIVE_SIGNAL_CONTRACT_DIRECTION_TOTALS,
  FIVE_SIGNAL_CONTRACT_GROUPS,
  FIVE_SIGNAL_MODES,
  deriveFiveSignalFlowSnapshot,
  type FiveSignalMode,
  type FiveSignalSummary,
} from "../src/features/five-signal-flow-model.ts";

test("the 42-item contract uses seven secondary classes across three directions", () => {
  assert.equal(FIVE_SIGNAL_CONTRACT_GROUPS.length, 7);
  assert.equal(
    new Set(FIVE_SIGNAL_CONTRACT_GROUPS.map(group => group.label)).size,
    7,
  );

  const directionTotals = FIVE_SIGNAL_CONTRACT_GROUPS.reduce(
    (totals, group) => {
      totals[group.direction] += group.count;
      return totals;
    },
    { uplink: 0, downlink: 0, overall: 0 },
  );

  assert.deepEqual(directionTotals, FIVE_SIGNAL_CONTRACT_DIRECTION_TOTALS);
  assert.equal(
    FIVE_SIGNAL_CONTRACT_GROUPS.reduce(
      (total, group) => total + group.count,
      0,
    ),
    42,
  );
  assert.equal(
    FIVE_SIGNAL_CONTRACT_GROUPS.filter(group => group.direction === "uplink")
      .length,
    3,
  );
  assert.equal(
    FIVE_SIGNAL_CONTRACT_GROUPS.filter(group => group.direction === "downlink")
      .length,
    1,
  );
  assert.equal(
    FIVE_SIGNAL_CONTRACT_GROUPS.filter(group => group.direction === "overall")
      .length,
    3,
  );
});

const validSummary = (
  overrides: Partial<FiveSignalSummary> = {},
): FiveSignalSummary => ({
  total: 42,
  measured: 21,
  pass: 3,
  fail: 2,
  review_required: 16,
  not_measured: 21,
  invalid: 0,
  ...overrides,
});

const report = (
  mode: string,
  summary: unknown,
  evaluationId = `${mode}-latest`,
) => ({
  evaluation_id: evaluationId,
  created_at: "2026-10-28T00:00:00Z",
  mode,
  summary,
});

test("empty evaluation list remains pending at zero of eight and zero of 336", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([]);
  assert.equal(snapshot.state, "pending");
  assert.equal(snapshot.availableModes, 0);
  assert.equal(snapshot.generatedItems, 0);
  assert.deepEqual(snapshot.counts, {
    pass: 0,
    fail: 0,
    review_required: 0,
    not_measured: 0,
    invalid: 0,
  });
});

test("one valid mode is partial and contributes exactly 42 rows", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([
    report("NB-wired", validSummary()),
  ]);
  assert.equal(snapshot.state, "partial");
  assert.equal(snapshot.availableModes, 1);
  assert.equal(snapshot.generatedItems, 42);
  assert.equal(snapshot.counts.pass, 3);
  assert.equal(snapshot.counts.not_measured, 21);
});

test("the first server-ordered evaluation wins for each mode", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([
    report(
      "NB-wired",
      validSummary({
        measured: 20,
        pass: 5,
        fail: 1,
        review_required: 14,
        not_measured: 22,
      }),
      "newest",
    ),
    report(
      "NB-wired",
      validSummary({
        measured: 20,
        pass: 0,
        fail: 6,
        review_required: 14,
        not_measured: 22,
      }),
      "older",
    ),
  ]);
  assert.equal(snapshot.latestByMode["NB-wired"]?.evaluation_id, "newest");
  assert.equal(snapshot.counts.pass, 5);
  assert.equal(snapshot.counts.fail, 1);
  assert.equal(snapshot.generatedItems, 42);
});

test("all eight valid modes produce a complete 336-row ledger", () => {
  const snapshot = deriveFiveSignalFlowSnapshot(
    FIVE_SIGNAL_MODES.map(mode => report(mode, validSummary())),
  );
  assert.equal(snapshot.state, "complete");
  assert.equal(snapshot.availableModes, 8);
  assert.equal(snapshot.generatedItems, 336);
  assert.deepEqual(snapshot.counts, {
    pass: 24,
    fail: 16,
    review_required: 128,
    not_measured: 168,
    invalid: 0,
  });
});

test("unknown modes cannot be counted toward the NV200 eight-mode contract", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([
    report("FB-wired" as FiveSignalMode, validSummary()),
    report("NB-wired", validSummary()),
  ]);
  assert.equal(snapshot.state, "invalid");
  assert.equal(snapshot.availableModes, 1);
  assert.equal(snapshot.ignoredModeCount, 1);
  assert.equal(snapshot.invalidModeCount, 1);
  assert.equal(snapshot.generatedItems, 42);
});

test("a mode with other than 42 rows is reported as invalid", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([
    report(
      "NB-wired",
      validSummary({
        total: 41,
        measured: 20,
        review_required: 15,
      }),
    ),
  ]);
  assert.equal(snapshot.state, "invalid");
  assert.equal(snapshot.availableModes, 0);
  assert.equal(snapshot.generatedItems, 0);
  assert.equal(snapshot.invalidModeCount, 1);
  assert.match(
    snapshot.latestByMode["NB-wired"]?.problem ?? "",
    /单模式应为 42 项，实际 41 项/,
  );
});

test("malformed or internally contradictory summaries never create false passes", () => {
  const snapshot = deriveFiveSignalFlowSnapshot([
    report("NB-wired", { total: 42, pass: 42 }),
    report("NB-wireless", validSummary({ total: 43 })),
  ]);
  assert.equal(snapshot.state, "invalid");
  assert.equal(snapshot.availableModes, 0);
  assert.equal(snapshot.generatedItems, 0);
  assert.equal(snapshot.invalidModeCount, 2);
});

// Negative cases keep row coverage separate from usable signal evidence.
import {
  contractGroupsForMode, FIVE_SIGNAL_ROLES, METRIC_VALUE_KEYS,
  parseFiveSignalDetail, analysisAvailability,
} from "../src/features/five-signal-flow-model.ts";

const detailFixture = () => ({
  evaluation_id: "detail-a", mode: "NB-wired", created_at: "2026-10-28T00:00:00Z",
  summary: validSummary(),
  inputs: Object.fromEntries(FIVE_SIGNAL_ROLES.map(role => [role, {
    name: `${role}.wav`, sha256: "a".repeat(64), sample_rate: 16000, duration_s: 5,
  }])),
  analysis: {
    values: Object.fromEntries(METRIC_VALUE_KEYS.flat().map(key => [key, 1])),
    segments: { near_only_seconds: 1, far_only_seconds: 1, double_talk_seconds: 1, noise_only_seconds: 1 },
    alignments: {
      microphone: { delay_ms: 10, correlation: 0.95, available: true, reliable: true },
      near_end: { delay_ms: 20, correlation: 0.95, available: true, reliable: true },
    },
  },
});

test("all eight display classifications match the real contract, including SWB", () => {
  const contract = JSON.parse(readFileSync(new URL("../cpp/evaluation/nv200-test-contract.json", import.meta.url), "utf8"));
  for (const mode of FIVE_SIGNAL_MODES) {
    const rows = contract.items.filter((row: {sheet: string}) => row.sheet === mode);
    const totals = { uplink: 0, downlink: 0, overall: 0 };
    for (const row of rows) {
      const group = ["round_trip_delay", "tclw", "hats_snr"].includes(row.family)
        ? "overall" : row.direction === "receive" ? "downlink" : "uplink";
      totals[group]++;
    }
    const groups = contractGroupsForMode(mode);
    assert.deepEqual(groups.reduce((n, group) => { n[group.direction] += group.count; return n; }, {uplink: 0, downlink: 0, overall: 0}), totals, mode);
    assert.equal(groups.find(group => group.label === "回声耦合")?.count, rows.filter((row: {family: string}) => row.family === "tclw").length);
    assert.equal(groups.reduce((n, group) => n + group.count, 0), 42);
  }
});

const contract = JSON.parse(
  readFileSync(
    new URL("../cpp/evaluation/nv200-test-contract.json", import.meta.url),
    "utf8",
  ),
) as {
  items: Array<{
    sheet: string;
    source_cell: string;
    test_name: string;
    family: string;
    direction: string;
    unit: string;
    lower_limit: number | null;
    upper_limit: number | null;
  }>;
};

function contractSheet(mode: string): NV200SheetData {
  const items: NV200TestItem[] = contract.items
    .filter((row) => row.sheet === mode)
    .map((row) => ({
      row: Number(row.source_cell.slice(1)),
      category: "合同项",
      name: row.test_name,
      summary: row.test_name,
      unit: row.unit,
      upperLimit: row.upper_limit === null ? "" : String(row.upper_limit),
      lowerLimit: row.lower_limit === null ? "" : String(row.lower_limit),
      gValue: "",
      isNumeric: false,
      lowerNum: row.lower_limit,
      upperNum: row.upper_limit,
      gNum: null,
      hNum: null,
      verdict: "unmeasured",
      margin: null,
      excess: null,
    }));
  return {
    sheetId: mode,
    sheetLabel: mode,
    dates: [],
    items,
    stats: {
      total: items.length,
      measured: 0,
      pass: 0,
      fail: 0,
      pending: 0,
      unmeasured: items.length,
      retested: 0,
    },
  };
}

function expectedSource(family: string, direction: string) {
  if (family === "round_trip_delay") return "alignment";
  if (family === "double_talk_classification") return "doubleTalk";
  if (family === "tclw") return "echo";
  if (direction !== "send") return null;
  if (family === "listening_quality") return "stoi";
  if (["slr", "idle_noise_level", "hats_snr", "cng_level"].includes(family)) {
    return "level";
  }
  if (
    [
      "frequency_response",
      "idle_noise_peaks",
      "out_of_band",
      "cng_spectrum",
    ].includes(family)
  ) {
    return "spectrum";
  }
  return null;
}

test("metric mapping covers all eight modes with 42 unique rows each", () => {
  for (const mode of FIVE_SIGNAL_MODES) {
    const mapping = buildMetricMapping(contractSheet(mode));
    assert.equal(mapping.valid, true, mode);
    assert.equal(mapping.rows.length, 42, mode);
    assert.deepEqual(
      mapping.rows.map((row) => row.row),
      Array.from({ length: 42 }, (_, index) => index + 2),
      mode,
    );
  }
});

test("NB and WB expose 21 mapped rows and SWB exposes 22", () => {
  for (const mode of FIVE_SIGNAL_MODES) {
    const mapping = buildMetricMapping(contractSheet(mode));
    const expected = mode.startsWith("SWB-")
      ? {
          mapped: 22,
          unmapped: 20,
          counts: {
            alignment: 1,
            sisdr: 0,
            stoi: 1,
            level: 8,
            spectrum: 7,
            doubleTalk: 2,
            echo: 3,
          },
        }
      : {
          mapped: 21,
          unmapped: 21,
          counts: {
            alignment: 1,
            sisdr: 0,
            stoi: 1,
            level: 7,
            spectrum: 8,
            doubleTalk: 2,
            echo: 2,
          },
        };
    assert.equal(mapping.mappedCount, expected.mapped, mode);
    assert.equal(mapping.unmappedCount, expected.unmapped, mode);
    assert.deepEqual(mapping.sourceCounts, expected.counts, mode);
  }
});

test("metric sources match the contract family and direction for every mode", () => {
  for (const mode of FIVE_SIGNAL_MODES) {
    const mapping = buildMetricMapping(contractSheet(mode));
    const contractRows = contract.items.filter((row) => row.sheet === mode);
    assert.equal(mapping.rows.length, contractRows.length, mode);
    for (const row of contractRows) {
      const mapped = mapping.rows.find(
        (candidate) => candidate.row === Number(row.source_cell.slice(1)),
      );
      assert.ok(mapped, `${mode}:${row.source_cell}`);
      assert.equal(
        mapped.source,
        expectedSource(row.family, row.direction),
        `${mode}:${row.source_cell}:${row.family}:${row.direction}`,
      );
    }
  }
});

test("metric mapping rejects a non-42-row sheet without inventing evidence", () => {
  const sheet = contractSheet("WB-wired");
  const mapping = buildMetricMapping({
    ...sheet,
    items: sheet.items.slice(0, 41),
  });
  assert.equal(mapping.valid, false);
  assert.equal(mapping.rows.length, 0);
  assert.equal(mapping.mappedCount, 0);
  assert.match(mapping.problem ?? "", /B2–B43 共 42 项/);
});

test("DNSMOS mapping covers all eight modes with 12 direct proxy rows", () => {
  const expectedRows = [
    [24, "ovrl"],
    [25, "bak"],
    [26, "sig"],
    [29, "ovrl"],
    [30, "bak"],
    [31, "sig"],
    [34, "ovrl"],
    [35, "bak"],
    [36, "sig"],
    [39, "ovrl"],
    [40, "bak"],
    [41, "sig"],
  ] as const;
  for (const mode of FIVE_SIGNAL_MODES) {
    const mapping = buildDnsmosMapping(contractSheet(mode));
    assert.equal(mapping.valid, true, mode);
    assert.equal(mapping.rows.length, 42, mode);
    assert.equal(mapping.mappedCount, 12, mode);
    assert.equal(mapping.unmappedCount, 30, mode);
    assert.deepEqual(mapping.outputCounts, { sig: 4, bak: 4, ovrl: 4 }, mode);
    assert.deepEqual(
      mapping.rows
        .filter((row) => row.output !== null)
        .map((row) => [row.row, row.output]),
      expectedRows,
      mode,
    );
    for (const row of mapping.rows.filter((candidate) => candidate.output)) {
      const name = row.item.name.toUpperCase();
      const expectedToken = {
        sig: "SMOS",
        bak: "NMOS",
        ovrl: "GMOS",
      }[row.output!];
      assert.ok(
        name.includes(expectedToken),
        `${mode}:B${row.row}:${row.output}:${row.item.name}`,
      );
    }
  }
});

test("DNSMOS mapping rejects an incomplete contract without inventing links", () => {
  const sheet = contractSheet("SWB-wired");
  const mapping = buildDnsmosMapping({
    ...sheet,
    items: sheet.items.slice(0, 41),
  });
  assert.equal(mapping.valid, false);
  assert.equal(mapping.rows.length, 0);
  assert.equal(mapping.mappedCount, 0);
  assert.equal(mapping.unmappedCount, 0);
  assert.match(mapping.problem ?? "", /B2–B43 共 42 项/);
});

test("receive proxy mapping covers B3–B11 with bandwidth-specific contract semantics", () => {
  const expected = {
    "NB-wired": {
      proxies: [3, 4, 5, 6, 7, 8, 9, 10, 11],
      gaps: [],
      counts: {
        binauralLevel: 2,
        frequencyResponse: 1,
        idleNoise: 1,
        noisePeaks: 1,
        outOfBand: 1,
        rlrLinearity: 2,
        binauralP835: 1,
      },
    },
    "WB-wired": {
      proxies: [3, 4, 5, 6, 7, 8, 9, 10, 11],
      gaps: [],
      counts: {
        binauralLevel: 2,
        frequencyResponse: 1,
        idleNoise: 1,
        noisePeaks: 1,
        outOfBand: 1,
        rlrLinearity: 2,
        binauralP835: 1,
      },
    },
    "SWB-wired": {
      proxies: [3, 4, 5, 6, 7, 8, 9, 10],
      gaps: [11],
      counts: {
        binauralLevel: 2,
        frequencyResponse: 1,
        idleNoise: 2,
        noisePeaks: 2,
        outOfBand: 0,
        rlrLinearity: 0,
        binauralP835: 1,
      },
    },
  } as const;

  for (const [mode, fixture] of Object.entries(expected)) {
    const mapping = buildReceiveProxyMapping(contractSheet(mode));
    assert.equal(mapping.valid, true, mode);
    assert.deepEqual(
      mapping.rows.map((row) => row.row),
      [3, 4, 5, 6, 7, 8, 9, 10, 11],
      mode,
    );
    assert.deepEqual(
      mapping.rows.filter((row) => row.proxy).map((row) => row.row),
      fixture.proxies,
      mode,
    );
    assert.deepEqual(
      mapping.rows.filter((row) => !row.proxy).map((row) => row.row),
      fixture.gaps,
      mode,
    );
    assert.deepEqual(mapping.proxyCounts, fixture.counts, mode);
    assert.equal(mapping.proxyCount, fixture.proxies.length, mode);
    assert.equal(mapping.unmappedCount, fixture.gaps.length, mode);
  }
});

test("receive proxy mapping adds linearity and P.835 proxies without claiming formal compliance", () => {
  const wb = buildReceiveProxyMapping(contractSheet("WB-wired"));
  assert.equal(
    wb.rows.find((row) => row.row === 3)?.proxy,
    "binauralLevel",
  );
  assert.equal(
    wb.rows.find((row) => row.row === 5)?.proxy,
    "rlrLinearity",
  );
  assert.equal(
    wb.rows.find((row) => row.row === 6)?.proxy,
    "rlrLinearity",
  );
  assert.match(wb.rows.find((row) => row.row === 5)?.gap ?? "", /三次/);
  assert.match(wb.rows.find((row) => row.row === 5)?.gap ?? "", /正式/);
  assert.match(wb.rows.find((row) => row.row === 8)?.gap ?? "", /dB\[Pa\]/);
  assert.equal(
    wb.rows.find((row) => row.row === 10)?.proxy,
    "binauralP835",
  );
  assert.match(wb.rows.find((row) => row.row === 10)?.gap ?? "", /不是 TMOS/);

  const swb = buildReceiveProxyMapping(contractSheet("SWB-wired"));
  assert.equal(
    swb.rows.find((row) => row.row === 6)?.proxy,
    "binauralP835",
  );
  assert.match(swb.rows.find((row) => row.row === 6)?.gap ?? "", /TMOS\/POLQA/);
  assert.match(swb.rows.find((row) => row.row === 11)?.gap ?? "", /发送/);
});

test("WB-wireless closes all 42 structural gaps with explicit proxy classes", () => {
  const sheet = contractSheet("WB-wireless");
  const engineering = buildMetricMapping(sheet);
  const dnsmos = buildDnsmosMapping(sheet);
  const receive = buildReceiveProxyMapping(sheet);
  const covered = engineering.rows.filter(
    (row, index) =>
      row.source !== null ||
      dnsmos.rows[index]?.output !== null ||
      receive.rows.some(
        (candidate) => candidate.row === row.row && candidate.proxy !== null,
      ),
  );

  assert.equal(covered.length, 42);
  assert.deepEqual(
    receive.rows
      .filter((row) => [5, 6, 10].includes(row.row))
      .map((row) => [row.row, row.proxy]),
    [
      [5, "rlrLinearity"],
      [6, "rlrLinearity"],
      [10, "binauralP835"],
    ],
  );
});

test("receive proxy mapping rejects an incomplete contract", () => {
  const sheet = contractSheet("WB-wired");
  const mapping = buildReceiveProxyMapping({
    ...sheet,
    items: sheet.items.slice(0, 41),
  });
  assert.equal(mapping.valid, false);
  assert.equal(mapping.rows.length, 0);
  assert.equal(mapping.proxyCount, 0);
  assert.equal(mapping.unmappedCount, 0);
  assert.match(mapping.problem ?? "", /B2–B43 共 42 项/);
});

test("measured must equal pass plus fail plus review_required", () => {
  const result = deriveFiveSignalFlowSnapshot([report("NB-wired", validSummary({ measured: 42 }))]);
  assert.equal(result.state, "invalid");
  assert.equal(result.measuredItems, 0);
  assert.equal(result.counts.invalid, 0, "report anomalies are not invalid items");
});

test("336 unmeasured rows are structurally complete without proving metric availability", () => {
  const snapshot = deriveFiveSignalFlowSnapshot(FIVE_SIGNAL_MODES.map(mode => report(mode, validSummary({measured: 0, pass: 0, fail: 0, review_required: 0, not_measured: 42}))));
  assert.equal(snapshot.state, "complete");
  assert.equal(snapshot.measuredItems, 0);
  assert.equal(analysisAvailability(null).state, "pending");
  const fixture = detailFixture();
  fixture.analysis.values = Object.fromEntries(METRIC_VALUE_KEYS.flat().map(key => [key, null])) as never;
  assert.deepEqual(analysisAvailability(parseFiveSignalDetail(fixture, "detail-a", "NB-wired")), {state: "pending", label: "无可用指标值"});
});

test("detail validation rejects mixed report identities and invalid metadata", () => {
  assert.throws(() => parseFiveSignalDetail(detailFixture(), "detail-b", "NB-wired"), /身份/);
  assert.throws(() => parseFiveSignalDetail(detailFixture(), "detail-a", "SWB-wired"), /身份/);
  const fixture = detailFixture();
  fixture.inputs.reference.sha256 = "missing";
  assert.throws(() => parseFiveSignalDetail(fixture, "detail-a", "NB-wired"), /元数据/);
});

test("detail validation binds an execution report to the selected algorithm cluster", () => {
  const fixture = detailFixture();
  (fixture as any).execution = {
    source: "alignment_cli",
    algorithm_cluster_id: "vcp_default",
    algorithm_cluster_label: "VCP default",
    output_sha256: fixture.inputs.near_end.sha256,
    cli_sha256: "b".repeat(64),
    catalog_sha256: "c".repeat(64),
    dataset_id: "field-p0-m2-multirender-snr15",
    configuration: { profile: "vcp", selection: { bf: "das" } },
    rx_processing: false,
  };
  assert.doesNotThrow(() => parseFiveSignalDetail(fixture, "detail-a", "NB-wired", "vcp_default"));
  assert.throws(() => parseFiveSignalDetail(fixture, "detail-a", "NB-wired", "vcp_unified"), /算法簇/);
  (fixture as any).execution.output_sha256 = "d".repeat(64);
  assert.throws(() => parseFiveSignalDetail(fixture, "detail-a", "NB-wired", "vcp_default"), /实际输出/);
});

test("missing, null and non-finite metric values never display complete availability", () => {
  const fixture = detailFixture();
  delete fixture.analysis.values.output_stoi;
  assert.equal(analysisAvailability(parseFiveSignalDetail(fixture, "detail-a", "NB-wired")).state, "partial");
  fixture.analysis.values.output_stoi = NaN;
  assert.throws(() => parseFiveSignalDetail(fixture, "detail-a", "NB-wired"), /非有限数/);
});

test("unreliable alignment, missing segments and report anomalies override readiness", () => {
  const fixture = detailFixture();
  assert.equal(analysisAvailability(parseFiveSignalDetail(fixture, "detail-a", "NB-wired")).state, "complete");
  fixture.analysis.alignments.microphone.reliable = false;
  assert.deepEqual(analysisAvailability(parseFiveSignalDetail(fixture, "detail-a", "NB-wired")), { state: "partial", label: "对齐需复核" });
  fixture.analysis.alignments.microphone.reliable = true;
  fixture.analysis.segments.double_talk_seconds = 0;
  const parsed = parseFiveSignalDetail(fixture, "detail-a", "NB-wired");
  assert.equal(analysisAvailability(parsed).state, "partial");
  assert.equal(analysisAvailability(parsed, true).state, "invalid");
});
