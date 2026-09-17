export type JobState =
  | "queued"
  | "loading_model"
  | "running"
  | "done"
  | "error";

export type Health = {
  status: string;
  model_loaded: boolean;
  device: string;
  dtype: string;
  base_model: string;
  gguf: string;
  gguf_present: boolean;
  base_ready: boolean;
};

export type StudioConfig = {
  default_steps: number;
  max_steps: number;
  default_guidance: number;
  default_negative: string;
  max_side: number;
  max_batch: number;
};

export type JobImage = {
  url: string;
  seed: number;
  filename: string;
  elapsed?: number | null;
};

export type JobStatus = {
  id: string;
  state: JobState;
  progress: number;
  step: number;
  total_steps: number;
  prompt: string;
  images: JobImage[];
  error: string | null;
  created_at: number;
  elapsed: number;
};

export type GalleryImage = {
  url: string;
  filename: string;
  prompt: string | null;
  negative_prompt: string | null;
  seed: number | null;
  steps: number | null;
  guidance: number | null;
  elapsed: number | null;
};

export type GeneratePayload = {
  prompt: string;
  negative_prompt: string | null;
  width: number;
  height: number;
  steps: number;
  guidance: number;
  seed: number | null;
  num_images: number;
  init_image: string | null;
  strength?: number;
};

export type RatioPreset = {
  label: string;
  width: number;
  height: number;
};
