import { requestJson } from "./client";
import type {
  GalleryImage,
  GeneratePayload,
  Health,
  JobStatus,
  StudioConfig,
} from "../types/studio";

export const studioApi = {
  health: (): Promise<Health> => requestJson<Health>("/api/health"),

  config: (): Promise<StudioConfig> => requestJson<StudioConfig>("/api/config"),

  generate: (payload: GeneratePayload): Promise<JobStatus> =>
    requestJson<JobStatus>("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    }),

  job: (id: string): Promise<JobStatus> => requestJson<JobStatus>(`/api/jobs/${id}`),

  gallery: (limit = 60): Promise<{ images: GalleryImage[] }> =>
    requestJson<{ images: GalleryImage[] }>(`/api/gallery?limit=${limit}`),

  deleteImage: (filename: string): Promise<{ ok: boolean; filename: string }> =>
    requestJson(`/api/gallery/${encodeURIComponent(filename)}`, { method: "DELETE" }),

  deleteGallery: (): Promise<{ ok: boolean; removed: number }> =>
    requestJson("/api/gallery", { method: "DELETE" }),
};
