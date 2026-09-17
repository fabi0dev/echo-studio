import { formatDuration } from "../lib/formatDuration";
import type { JobStatus } from "../types/studio";

type ProgressPanelProps = {
  job: JobStatus | null;
  visible: boolean;
};

function labelFor(job: JobStatus | null): string {
  if (!job) {
    return "Preparando…";
  }
  if (job.state === "loading_model") {
    return "Carregando modelo…";
  }
  if (job.state === "running") {
    return `Gerando · passo ${job.step}/${job.total_steps}`;
  }
  if (job.state === "queued") {
    return "Na fila…";
  }
  if (job.state === "done") {
    return job.elapsed > 0 ? `Concluído em ${formatDuration(job.elapsed)}` : "Concluído";
  }
  return "Preparando…";
}

export function ProgressPanel({ job, visible }: ProgressPanelProps) {
  if (!visible) {
    return null;
  }
  const progress = job?.progress ?? 0;
  return (
    <section className="progress-wrap">
      <div className="progress-head">
        <span>{labelFor(job)}</span>
        <span>{Math.round(progress * 100)}%</span>
      </div>
      <div className="progress-bar">
        <div className="progress-fill" style={{ width: `${Math.round(progress * 100)}%` }} />
      </div>
    </section>
  );
}
