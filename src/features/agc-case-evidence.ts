import evidence from "../data/agc-case-evidence.json" with { type: "json" };
import type { PipelineCaseGroup } from "./PipelineEvaluation";

// Generated from the same report/results and WAVs as the figures.
export function agcCaseGroups(algorithm: string): PipelineCaseGroup[] {
  return evidence.filter(group => group.algorithm === algorithm).map(group => {
    if (group.id !== "pass" && group.id !== "fail") throw Error("无效的 AGC 案例状态");
    return {
      ...group,
      id: group.id,
      status: group.id === "pass" ? "通过" : "失败",
      caseAudio: { ...group.caseAudio, windowSeconds: [group.caseAudio.windowSeconds[0], group.caseAudio.windowSeconds[1]] },
    };
  });
}
