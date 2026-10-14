import { AlertCircle, ArrowUpRight, ChevronDown } from "lucide-react";
import type { ReactNode } from "react";
import { STATUS_LABEL } from "../domain/model";
import type { Status, TestItem } from "../domain/model";
import { METRIC_MAP } from "../domain/metrics";

export function StatusBadge({ status }: { status: Status }) {
  return (
    <span className={`status-badge status-${status}`}>
      <span className="status-dot" />
      {STATUS_LABEL[status]}
    </span>
  );
}
export function SelectField({
  label,
  value,
  onChange,
  children,
  className = "",
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={`select-field ${className}`}>
      <span className="sr-only">{label}</span>
      <select
        aria-label={label}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {children}
      </select>
      <ChevronDown size={15} aria-hidden="true" />
    </label>
  );
}
export function Panel({
  title,
  subtitle,
  action,
  children,
  className = "",
}: {
  title: string;
  subtitle?: string;
  action?: ReactNode;
  children: ReactNode;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`}>
      <div className="panel-heading">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}
export function EmptyState({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <AlertCircle size={25} />
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      {action}
    </div>
  );
}
export function TestName({
  item,
  onClick,
}: {
  item: TestItem;
  onClick: () => void;
}) {
  return (
    <button className="test-name" onClick={onClick}>
      <span>
        {METRIC_MAP[item.metricId].name}
        <ArrowUpRight size={14} />
      </span>
      <small>{item.context || item.name}</small>
    </button>
  );
}
