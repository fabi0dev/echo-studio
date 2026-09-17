import { useCallback, useEffect, useState } from "react";
import { studioApi } from "../api/studioApi";
import type { Health } from "../types/studio";

const POLL_MS = 10_000;

export function useHealth(): Health | null {
  const [health, setHealth] = useState<Health | null>(null);

  const refresh = useCallback(async () => {
    try {
      setHealth(await studioApi.health());
    } catch {
      setHealth(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => {
      void refresh();
    }, POLL_MS);
    return () => {
      window.clearInterval(timer);
    };
  }, [refresh]);

  return health;
}
