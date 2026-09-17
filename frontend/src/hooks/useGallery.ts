import { useCallback, useEffect, useState } from "react";
import { studioApi } from "../api/studioApi";
import type { GalleryImage } from "../types/studio";

export function useGallery(): {
  images: GalleryImage[];
  reload: () => Promise<void>;
  remove: (filename: string) => Promise<void>;
  removeAll: () => Promise<void>;
} {
  const [images, setImages] = useState<GalleryImage[]>([]);

  const reload = useCallback(async () => {
    try {
      const data = await studioApi.gallery();
      setImages(data.images);
    } catch {
      setImages([]);
    }
  }, []);

  const remove = useCallback(async (filename: string) => {
    await studioApi.deleteImage(filename);
    setImages((current) => current.filter((image) => image.filename !== filename));
  }, []);

  const removeAll = useCallback(async () => {
    await studioApi.deleteGallery();
    setImages([]);
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  return { images, reload, remove, removeAll };
}
