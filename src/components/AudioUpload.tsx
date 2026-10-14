import { useEffect, useRef, useState } from "react";
import { Check, FileAudio, Upload, X } from "lucide-react";
import { ROLE_COLOR, ROLE_LABEL } from "../domain/audio";
import type { AudioDescription, AudioRole } from "../domain/audio";
const ROLE_HINT = {
  reference: "原始干净信号，用于时延、保真与可懂度对照。",
  before: "算法处理前的麦克风录音，保留噪声与回声。",
  after: "算法处理后的输出，这是开始计算的必需输入。",
  echo: "送往扬声器的远端信号，用于回声片段核对。",
};
export default function AudioUpload({
  role,
  file,
  channel,
  info,
  disabled,
  onFile,
  onChannel,
}: {
  role: AudioRole;
  file: File | null;
  channel: number;
  info?: AudioDescription;
  disabled: boolean;
  onFile: (file: File | null) => void;
  onChannel: (channel: number) => void;
}) {
  const input = useRef<HTMLInputElement>(null);
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  useEffect(() => {
    setError("");
    if (!file) {
      setUrl("");
      return;
    }
    const next = URL.createObjectURL(file);
    setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [file]);
  function accept(next?: File) {
    if (!next || disabled) return;
    if (!/\.(wav|flac|mp3|ogg|m4a|aac|aif|aiff)$/i.test(next.name)) {
      setError(
        "请选择音频文件。支持 WAV / FLAC / MP3 / OGG / M4A / AAC / AIFF。",
      );
      return;
    }
    if (!next.size || next.size > 32 * 1024 * 1024) {
      setError("请选择非空、32 MB 以内的音频。");
      return;
    }
    setError("");
    onFile(next);
  }
  return (
    <section
      className={`audio-upload-card ${file ? "has-file" : ""} ${drag ? "dragging" : ""}`}
      style={{ "--role-color": ROLE_COLOR[role] } as React.CSSProperties}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        accept(e.dataTransfer.files[0]);
      }}
    >
      <div className="audio-role-heading">
        <span className="role-icon">
          <FileAudio size={19} />
        </span>
        <div>
          <h3>{ROLE_LABEL[role]}</h3>
          <span>{role === "after" ? "必需" : "按需上传"}</span>
        </div>
        {file ? <Check size={17} className="file-check" /> : null}
      </div>
      <p className="audio-role-hint">{ROLE_HINT[role]}</p>
      <input
        ref={input}
        type="file"
        className="sr-only"
        aria-label={`上传${ROLE_LABEL[role]}音频`}
        accept=".wav,.flac,.mp3,.ogg,.m4a,.aac,.aif,.aiff"
        disabled={disabled}
        onChange={(e) => {
          accept(e.target.files?.[0]);
          e.target.value = "";
        }}
      />
      {file ? (
        <>
          <div className="selected-audio">
            <FileAudio size={16} />
            <div>
              <strong title={file.name}>{file.name}</strong>
              <small>
                {(file.size / 1024 / 1024).toFixed(2)} MB
                {info
                  ? ` · ${info.duration.toFixed(2)} s · ${info.sample_rate / 1000} kHz · ${info.channels} 声道`
                  : " · 计算时读取真实音频信息"}
              </small>
            </div>
            <button
              disabled={disabled}
              className="icon-button"
              aria-label={`移除${ROLE_LABEL[role]}`}
              onClick={() => {
                setError("");
                onFile(null);
              }}
            >
              <X size={16} />
            </button>
          </div>
          <audio
            controls
            preload="metadata"
            src={url || undefined}
            aria-label={`${ROLE_LABEL[role]}原文件试听`}
          />
          <div className="audio-card-bottom">
            <label>
              分析声道
              <select
                aria-label={`${ROLE_LABEL[role]}分析声道`}
                disabled={disabled}
                value={channel}
                onChange={(e) => onChannel(Number(e.target.value))}
              >
                {Array.from({ length: info?.channels ?? 8 }, (_, i) => (
                  <option key={i} value={i}>
                    声道 {i + 1}
                  </option>
                ))}
              </select>
            </label>
            <button
              className="text-link"
              disabled={disabled}
              onClick={() => input.current?.click()}
            >
              重新选择
            </button>
          </div>
          <small className="preview-note">
            试听原文件；计算只使用所选声道。
          </small>
        </>
      ) : (
        <button
          className="audio-dropzone"
          disabled={disabled}
          onClick={() => input.current?.click()}
        >
          <Upload size={22} />
          <strong>选择音频或拖放到这里</strong>
          <span>每文件 ≤32 MB · 0.1–120 秒</span>
        </button>
      )}
      {error ? (
        <p className="field-error" role="alert">
          {error}
        </p>
      ) : null}
    </section>
  );
}
