export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

function formatDetail(payload: unknown, status: number): string {
  if (!payload || typeof payload !== "object") {
    return `HTTP ${status}`;
  }
  const detail = (payload as { detail?: unknown }).detail;
  if (typeof detail === "string") {
    return detail;
  }
  if (Array.isArray(detail)) {
    return detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: unknown }).msg);
        }
        return JSON.stringify(item);
      })
      .join("\n");
  }
  return `HTTP ${status}`;
}

export async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  const payload: unknown = await response.json().catch(() => null);
  if (!response.ok) {
    throw new ApiError(formatDetail(payload, response.status), response.status);
  }
  return payload as T;
}
