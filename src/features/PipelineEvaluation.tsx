import "./aec-pipeline.css";
import CaseGraphEngine from "./CaseGraphEngine";
import { resolvePipelineCase } from "./pipeline-case-metadata";
import type { AudioGraphicFocus } from "../domain/audio-graphics";

export type PipelineCaseAudio = {
  id: string;
  assetId?: string;
  detail: string;
  sampleRate: number;
  windowSeconds: [number, number];
  delaySamples: number;
  beforeLabel: string;
  afterLabel: string;
  beforeFile: string;
  afterFile: string;
  beforeChannel?: number;
  afterChannel?: number;
  waveformSrc: string;
  melSrc: string;
  analysis: string;
};

export type PipelineCaseGroup = {
  id: "pass" | "fail" | "representative" | "pending";
  variant?: string;
  source?: string;
  status: "通过" | "失败" | "代表案例" | "待测";
  headline: string;
  note: string;
  caseAudio: PipelineCaseAudio;
};

export type PipelineEvaluationData = {
  tone: "warning" | "pending";
  status: string;
  headline: string;
  scope: string;
  source: string;
  sources?: string[];
  note: string;
  metrics: Array<{ label: string; value: string; detail: string }>;
  caseAudio?: PipelineCaseAudio;
  caseGroups?: PipelineCaseGroup[];
};

function StaticCaseFigure({ caseAudio }: { caseAudio: PipelineCaseAudio }) {
  const [start, end] = caseAudio.windowSeconds;
  return (
    <div className="aec-case-figures">
      <figure>
        <figcaption>
          <div>
            <strong>波形</strong>
            <span>原幅度 · 输入/输出共享纵轴</span>
          </div>
          <small>{caseAudio.analysis}</small>
        </figcaption>
        <img
          src={caseAudio.waveformSrc}
          alt={`${caseAudio.id} 输入与输出的典型案例波形`}
          width={1200}
          height={570}
          loading="lazy"
          decoding="async"
        />
      </figure>
      <figure>
        <figcaption>
          <div>
            <strong>Mel 频谱</strong>
            <span>STFT → 64-band Mel · 80 dB 固定范围</span>
          </div>
          <small>同色标，上限为输入/输出联合 99.7 百分位</small>
        </figcaption>
        <img
          src={caseAudio.melSrc}
          alt={`${caseAudio.id} 输入与输出的典型案例 Mel 频谱`}
          width={1200}
          height={650}
          loading="lazy"
          decoding="async"
        />
      </figure>
      <div className="aec-case-tags aec-case-tags--figures">
        <span>{caseAudio.sampleRate / 1000} kHz</span>
        <span>
          {start.toFixed(1)}–{end.toFixed(1)} s
        </span>
        <span>delay {caseAudio.delaySamples} samples</span>
      </div>
      <details className="aec-case-evidence">
        <summary>查看案例音频来源</summary>
        <dl>
          <div>
            <dt>{caseAudio.beforeLabel}</dt>
            <dd>
              <code>{caseAudio.beforeFile}</code>
            </dd>
          </div>
          <div>
            <dt>{caseAudio.afterLabel}</dt>
            <dd>
              <code>{caseAudio.afterFile}</code>
            </dd>
          </div>
        </dl>
      </details>
    </div>
  );
}

function CaseFigure({
  caseAudio,
  graphFocus,
}: {
  caseAudio: PipelineCaseAudio;
  graphFocus?: AudioGraphicFocus;
}) {
  if (graphFocus && caseAudio.assetId) {
    return (
      <CaseGraphEngine
        assetId={caseAudio.assetId}
        beforeLabel={caseAudio.beforeLabel}
        afterLabel={caseAudio.afterLabel}
        focus={graphFocus}
      />
    );
  }
  return <StaticCaseFigure caseAudio={caseAudio} />;
}

function TypicalCaseVisualization({
  evaluation,
  graphFocus,
}: {
  evaluation: PipelineEvaluationData;
  graphFocus?: AudioGraphicFocus;
}) {
  const caseGroups = evaluation.caseGroups?.map(group => ({
    ...group, caseAudio: resolvePipelineCase(group.caseAudio),
  }));
  const caseAudio = !caseGroups?.length && evaluation.caseAudio
    ? resolvePipelineCase(evaluation.caseAudio) : undefined;
  if (caseGroups?.length) {
    return (
      <section className="aec-evaluation-case">
        <div className="aec-case-heading">
          <div>
            <span>PASS / FAIL EVIDENCE</span>
            <h3>通过、失败状态的典型案例</h3>
            <p>
              每个状态使用该技术自己的冻结输入/输出音频对；不借用其他算法案例，也不把同一场景的不同算法结果混成一组。
            </p>
          </div>
          <div className="aec-case-tags">
            <span>{caseGroups.length} 组案例</span>
          </div>
        </div>
        <div className="aec-case-groups">
          {caseGroups.map((group) => (
            <article
              key={`${group.variant ?? "default"}-${group.id}`}
              className={`aec-case-group aec-case-group--${group.id}`}
            >
              <div className="aec-case-group-head">
                <div>
                  <span>{group.status}</span>
                  <strong>{group.headline}</strong>
                </div>
                <code>{group.caseAudio.id}</code>
              </div>
              <p>{group.note}</p>
              {group.source && <p className="aec-case-source">判定来源：<code>{group.source}</code></p>}
              <div className="aec-case-identity">
                <div>
                  <span>案例</span>
                  <code>{group.caseAudio.id}</code>
                </div>
                <p>{group.caseAudio.detail}</p>
              </div>
              <CaseFigure caseAudio={group.caseAudio} graphFocus={graphFocus} />
            </article>
          ))}
        </div>
      </section>
    );
  }

  if (!caseAudio) {
    return (
      <section className="aec-evaluation-case is-missing">
        <div className="aec-case-heading">
          <div>
            <span>TYPICAL CASE VISUALS</span>
            <h3>典型案例波形与 Mel 频谱</h3>
          </div>
          <strong>无可视化案例</strong>
        </div>
        <p>
          当前评测没有可核验的冻结输入/输出音频对，因此未生成波形或 Mel
          频谱，也不借用其他算法的案例。
        </p>
      </section>
    );
  }

  return (
    <section className="aec-evaluation-case">
      <div className="aec-case-heading">
        <div>
          <span>TYPICAL CASE VISUALS</span>
          <h3>典型案例波形与 Mel 频谱</h3>
          <p>
            同一冻结案例、同一时间窗；输出已按声明的固定延迟对齐。
          </p>
        </div>
        <div className="aec-case-tags">
          <span>{caseAudio.sampleRate / 1000} kHz</span>
          <span>
            {caseAudio.windowSeconds[0].toFixed(1)}–
            {caseAudio.windowSeconds[1].toFixed(1)} s
          </span>
          <span>delay {caseAudio.delaySamples} samples</span>
        </div>
      </div>

      <div className="aec-case-identity">
        <div>
          <span>案例</span>
          <code>{caseAudio.id}</code>
        </div>
        <p>{caseAudio.detail}</p>
      </div>
      <CaseFigure caseAudio={caseAudio} graphFocus={graphFocus} />
    </section>
  );
}

export default function PipelineEvaluation({
  title,
  evaluation,
  graphFocus,
}: {
  title: string;
  evaluation: PipelineEvaluationData;
  graphFocus?: AudioGraphicFocus;
}) {
  return (
    <section
      className={`aec-evaluation aec-evaluation--${evaluation.tone}`}
      aria-live="polite"
    >
      <div className="aec-evaluation-head">
        <div>
          <span>EVALUATION EVIDENCE</span>
          <h2>{title} 评测结果</h2>
          <p>{evaluation.scope}</p>
        </div>
        <div className="aec-evaluation-result">
          <strong>{evaluation.headline}</strong>
          <span>{evaluation.status}</span>
        </div>
      </div>
      <div className="aec-evaluation-metrics">
        {evaluation.metrics.map((metric) => (
          <div key={metric.label}>
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
            <small>{metric.detail}</small>
          </div>
        ))}
      </div>
      <div className="aec-evaluation-evidence">
        <div>
          <span>证据入口</span>
          {(evaluation.sources ?? [evaluation.source]).map(source => <code key={source}>{source}</code>)}
        </div>
        <p>{evaluation.note}</p>
      </div>
      <TypicalCaseVisualization evaluation={evaluation} graphFocus={graphFocus} />
    </section>
  );
}
