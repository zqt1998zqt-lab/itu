// Every stage starts after the preceding stage, so widening a stage also moves
// its background, children and connections together.
const GAP = 64;
const stage = (x: number, width: number) => ({ x, width, right: x + width });
const dataset = stage(32, 320);
const input = stage(dataset.right + GAP, 320);
const vcp = stage(input.right + GAP, 1204);
const alignment = stage(vcp.right + GAP, 336);
const analysis = stage(alignment.right + GAP, 560);
const mapping = stage(analysis.right + GAP, 680);
const modes = stage(mapping.right + GAP, 460);
const classify = stage(modes.right + GAP, 320);
const outcomes = stage(classify.right + GAP, 300);
const agents = stage(outcomes.right + GAP, 480);
const report = stage(agents.right + GAP, 360);
const boundary = stage(report.right + GAP, 360);

export const FIVE_SIGNAL_FLOW_LAYOUT = {
  dataset,
  input,
  vcp,
  alignment,
  analysis,
  mapping,
  modes,
  classify,
  outcomes,
  agents,
  report,
  boundary,
  width: boundary.right + 32,
  height: 1200,
  top: 80,
  stageHeight: 900,
  inputTop: 212,
  inputHeight: 280,
  mainY: 352,
  vcpCardWidth: 212,
  vcpCardHeight: 200,
  vcpCardGap: 24,
  uplinkY: 252,
  downlinkY: 596,
  algorithmClusterTop: 884,
  algorithmClusterHeight: 256,
  vcpGroupHeight: 1084,
  metricTop: 192,
  metricHeight: 108,
  metricStep: 128,
  modeTop: 176,
  modeHeight: 88,
  modeStep: 96,
  outcomeTop: 224,
  outcomeHeight: 128,
  outcomeStep: 176,
} as const;
