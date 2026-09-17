import { useCallback, useEffect, useRef, useState } from "react";
import { studioApi } from "../api/studioApi";
import type { GeneratePayload, JobStatus } from "../types/studio";

type GenerationController = {
  busy: boolean;
  job: JobStatus | null;
  error: string | null;
  generate: (payload: GeneratePayload) => Promise<void>;
  clearError: () => void;
};

export function useGeneration(onDone: () => void): GenerationController {
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<JobStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const onDoneRef = useRef(onDone);
  onDoneRef.current = onDone;

  const clearError = useCallback(() => setError(null), []);

  const generate = useCallback(async (payload: GeneratePayload) => {
    setError(null);
    setBusy(true);
    setJob({
      id: "",
      state: "queued",
      progress: 0,
      step: 0,
      total_steps: payload.steps * payload.num_images,
      prompt: payload.prompt,
      images: [],
      error: null,
      created_at: Date.now() / 1000,
      elapsed: 0,
    });
    try {
      const created = await studioApi.generate(payload);
      setJob(created);
    } catch (err) {
      setBusy(false);
      setJob(null);
      setError(err instanceof Error ? err.message : "Falha ao gerar");
    }
  }, []);

  useEffect(() => {
    if (!busy || !job?.id) {
      return;
    }
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const next = await studioApi.job(job.id);
        if (cancelled) {
          return;
        }
        setJob(next);
        if (next.state === "done") {
          window.clearInterval(timer);
          setBusy(false);
          onDoneRef.current();
        } else if (next.state === "error") {
          window.clearInterval(timer);
          setBusy(false);
          setError(next.error || "erro desconhecido");
        }
      } catch (err) {
        if (cancelled) {
          return;
        }
        window.clearInterval(timer);
        setBusy(false);
        setError(err instanceof Error ? err.message : "Falha ao consultar o job");
      }
    }, 700);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [busy, job?.id]);

  return { busy, job, error, generate, clearError };
}
