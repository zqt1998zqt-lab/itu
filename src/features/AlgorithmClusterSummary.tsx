import {
  chainSelections,
  type ChainPreset,
} from "./enhancement-chain-model";
import { AecRouteAvailability } from "./AecRouteAvailability";
import { isCapabilityAecRoute } from "./aec-route-capability";
import {
  algorithmClusterById,
  algorithmClusterRateSummary,
  type RateAlgorithmCluster,
} from "./rate-algorithm-clusters";

export default function AlgorithmClusterSummary({
  preset = null,
  cluster = null,
}: {
  preset?: ChainPreset | null;
  cluster?: RateAlgorithmCluster | null;
}) {
  const resolvedCluster =
    cluster ?? (preset ? algorithmClusterById(preset.id) : null);
  if (!resolvedCluster && !preset) return null;

  const label = resolvedCluster?.label ?? preset!.label;
  const description = resolvedCluster?.description ?? preset!.description;
  const id = resolvedCluster?.id ?? preset!.id;
  const profile = resolvedCluster?.profile ?? preset!.profile;
  const stages = resolvedCluster
    ? resolvedCluster.stages
    : chainSelections(preset!).map(({ ordinal, stage, option }) => ({
        ordinal,
        stageId: stage.id,
        stageLabel: stage.label,
        optionLabel: option?.label ?? "配置不可用",
        note: option?.note ?? "当前配置找不到该阶段的算法",
        bypassed: option?.id === "off",
        missing: option === null,
        candidate: Boolean(option?.candidate),
      }));
  const candidateCount = stages.filter((stage) => stage.candidate).length;
  const rateBadge = resolvedCluster ? algorithmClusterRateSummary(resolvedCluster.id) : null;
  const aecRouteId = resolvedCluster?.aecRouteId ?? preset?.selection.aec;
  const executionNote =
    resolvedCluster?.executionNote
    ?? "执行顺序固定为 BF → AEC → NS → EQ → AGC → DRC → Limiter。这里展示当前算法簇请求的完整配置；DSP、Complex_PF 或 LocalVQE 路由仍须通过后端能力、模型与库身份准入，实际执行身份以运行报告为准。";

  return (
    <section
      className="algorithm-cluster-summary"
      aria-label={`${label}实际执行算法`}
      aria-live="polite"
    >
      <header className="algorithm-cluster-summary-header">
        <div>
          <span>当前簇实际执行算法</span>
          <strong>{label}</strong>
          <small>{description}</small>
        </div>
        <div className="algorithm-cluster-summary-identity">
          <code>{id}</code>
          <span>{profile}</span>
          {rateBadge ? <span>{rateBadge}</span> : null}
          {candidateCount > 0 ? (
            <span className="is-candidate">{candidateCount} 个候选/实验级</span>
          ) : null}
        </div>
      </header>

      <ol
        className="algorithm-cluster-summary-stages"
        aria-label={`${label}的七级执行顺序`}
      >
        {stages.map((stage) => (
          <li
            key={stage.stageId}
            className={
              stage.missing
                ? "is-missing"
                : stage.bypassed
                  ? "is-bypassed"
                  : undefined
            }
            title={stage.note}
          >
            <span className="algorithm-cluster-summary-index">
              {String(stage.ordinal).padStart(2, "0")}
            </span>
            <span className="algorithm-cluster-summary-stage">
              {stage.stageLabel}
            </span>
            <strong>{stage.optionLabel}</strong>
          </li>
        ))}
      </ol>

      <p className="algorithm-cluster-summary-note">{executionNote}</p>
      {aecRouteId && isCapabilityAecRoute(aecRouteId) ? (
        <AecRouteAvailability routeId={aecRouteId} compact />
      ) : null}
    </section>
  );
}
