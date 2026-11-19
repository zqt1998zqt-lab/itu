import { useState } from "react";
import { Pause, Play } from "lucide-react";
import "./output-audio-preview.css";

type Props = {
  caseId: string;
  label: string;
  url: string;
  playing: boolean;
  onToggle: () => void;
};

export default function OutputAudioPreview({
  caseId,
  label,
  url,
  playing,
  onToggle,
}: Props) {
  const [failed, setFailed] = useState(false);
  const accessibleName = `试听 ${caseId} ${label}`;
  return (
    <div className={`output-audio-preview${playing ? " is-playing" : ""}`}>
      <button
        type="button"
        className="output-audio-preview-toggle"
        aria-label={accessibleName}
        aria-pressed={playing}
        onClick={() => {
          setFailed(false);
          onToggle();
        }}
      >
        {playing ? (
          <Pause size={12} aria-hidden="true" />
        ) : (
          <Play size={12} fill="currentColor" aria-hidden="true" />
        )}
        {playing ? "停止试听" : "试听输出"}
      </button>
      {playing ? (
        <audio
          controls
          autoPlay
          preload="none"
          src={url}
          aria-label={accessibleName}
          onError={() => setFailed(true)}
        >
          当前浏览器不支持音频播放。
        </audio>
      ) : null}
      {failed ? (
        <span className="output-audio-preview-error" role="alert">
          输出音频加载失败，请检查批次输出是否完整。
        </span>
      ) : null}
    </div>
  );
}
