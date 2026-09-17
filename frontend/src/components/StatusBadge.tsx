import type { Health } from "../types/studio";

type StatusBadgeProps = {
  health: Health | null;
  busy: boolean;
};

function gpuLabel(health: Health): string {
  const adapter = health.adapter?.trim();
  if (!adapter) {
    return health.device;
  }
  return adapter
    .replace(/^directml:\d+\s+/i, "")
    .replace(/^AMD\s+/i, "")
    .replace(/^NVIDIA\s+/i, "")
    .replace(/^Intel\(R\)\s+/i, "");
}

function labelFor(health: Health | null, busy: boolean): { text: string; tone: "ok" | "bad" | "busy" | "idle" } {
  if (health === null) {
    return { text: "servidor offline", tone: "bad" };
  }
  if (!health.gguf_present) {
    return { text: "modelo ausente", tone: "bad" };
  }
  if (!health.base_ready) {
    return { text: "baixando checkpoint…", tone: "busy" };
  }
  if (busy) {
    return { text: "gerando…", tone: "busy" };
  }
  if (health.model_loaded) {
    return { text: `pronto · ${gpuLabel(health)} · ${health.dtype}`, tone: "ok" };
  }
  return { text: "modelo não carregado", tone: "idle" };
}

export function StatusBadge({ health, busy }: StatusBadgeProps) {
  const { text, tone } = labelFor(health, busy);
  return (
    <div className="status">
      <span className={`dot ${tone === "idle" ? "" : tone}`.trim()} />
      <span>{text}</span>
    </div>
  );
}
