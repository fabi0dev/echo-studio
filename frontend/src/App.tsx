import { useCallback, useEffect, useState } from "react";
import { Composer, type ComposerValues } from "./components/Composer";
import { Gallery } from "./components/Gallery";
import { Lightbox } from "./components/Lightbox";
import { ProgressPanel } from "./components/ProgressPanel";
import { StatusBadge } from "./components/StatusBadge";
import { useGallery } from "./hooks/useGallery";
import { useGeneration } from "./hooks/useGeneration";
import { useHealth } from "./hooks/useHealth";
import type { GalleryImage, GeneratePayload } from "./types/studio";

const INITIAL: ComposerValues = {
  prompt: "",
  negative: "",
  width: 512,
  height: 512,
  count: 1,
  steps: 20,
  guidance: 7,
  seed: "",
  initImage: null,
  strength: 65,
  advanced: false,
};

function toPayload(values: ComposerValues): GeneratePayload {
  return {
    prompt: values.prompt.trim(),
    negative_prompt: values.negative.trim() || null,
    width: values.width,
    height: values.height,
    steps: values.steps,
    guidance: values.guidance,
    seed: values.seed === "" ? null : Number(values.seed),
    num_images: values.count,
    init_image: values.initImage,
    strength: values.initImage ? values.strength / 100 : undefined,
  };
}

function seedOf(image: GalleryImage): string {
  if (image.seed != null) {
    return String(image.seed);
  }
  return image.filename.match(/_(\d+)\.png$/)?.[1] ?? "";
}

export function App() {
  const health = useHealth();
  const { images, reload, remove, removeAll } = useGallery();
  const { busy, job, error, generate, clearError } = useGeneration(reload);
  const [values, setValues] = useState<ComposerValues>(INITIAL);
  const [flashing, setFlashing] = useState(false);
  const [lightbox, setLightbox] = useState<GalleryImage | null>(null);

  const patch = useCallback((next: Partial<ComposerValues>) => {
    setValues((current) => ({ ...current, ...next }));
  }, []);

  const flash = useCallback(() => {
    setFlashing(true);
    window.setTimeout(() => setFlashing(false), 700);
  }, []);

  const runGenerate = useCallback(() => {
    if (!values.prompt.trim()) {
      return;
    }
    void generate(toPayload(values));
  }, [generate, values]);

  const applyParams = useCallback(
    (image: GalleryImage, keepSeed: boolean) => {
      patch({
        prompt: image.prompt ?? values.prompt,
        negative: image.negative_prompt ?? values.negative,
        seed: keepSeed && image.seed != null ? String(image.seed) : "",
        steps: image.steps ?? values.steps,
        guidance: image.guidance ?? values.guidance,
      });
      window.scrollTo({ top: 0, behavior: "smooth" });
      flash();
    },
    [flash, patch, values.guidance, values.negative, values.prompt, values.steps],
  );

  const deleteImage = useCallback(
    async (image: GalleryImage) => {
      const confirmed = window.confirm("Apagar esta criação? Isso não tem volta.");
      if (!confirmed) {
        return;
      }
      try {
        await remove(image.filename);
        setLightbox((current) => (current?.filename === image.filename ? null : current));
      } catch (err) {
        window.alert(`Não consegui apagar: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [remove],
  );

  const deleteAll = useCallback(async () => {
    if (images.length === 0) {
      return;
    }
    const confirmed = window.confirm(
      `Apagar ${images.length} criação(ões) da galeria? Isso não tem volta.`,
    );
    if (!confirmed) {
      return;
    }
    try {
      await removeAll();
      setLightbox(null);
    } catch (err) {
      window.alert(`Não consegui apagar: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [images.length, removeAll]);

  const useAsReference = useCallback(
    async (url: string) => {
      try {
        const response = await fetch(url);
        const blob = await response.blob();
        const dataUrl = await new Promise<string>((resolve, reject) => {
          const reader = new FileReader();
          reader.onload = () => {
            if (typeof reader.result === "string") {
              resolve(reader.result);
              return;
            }
            reject(new Error("leitura inválida"));
          };
          reader.onerror = () => reject(reader.error ?? new Error("falha ao ler"));
          reader.readAsDataURL(blob);
        });
        patch({ initImage: dataUrl });
        setLightbox(null);
        window.scrollTo({ top: 0, behavior: "smooth" });
      } catch (err) {
        window.alert(`Não consegui carregar a imagem: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [patch],
  );

  useEffect(() => {
    const onPaste = (event: ClipboardEvent) => {
      const item = [...(event.clipboardData?.items ?? [])].find((entry) =>
        entry.type.startsWith("image/"),
      );
      const file = item?.getAsFile();
      if (!file) {
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          patch({ initImage: reader.result });
        }
      };
      reader.readAsDataURL(file);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setLightbox(null);
      }
    };
    document.addEventListener("paste", onPaste);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("paste", onPaste);
      document.removeEventListener("keydown", onKey);
    };
  }, [patch]);

  const pendingCount = busy ? values.count : 0;
  const showProgress = busy || job?.state === "done";

  return (
    <>
      <header className="topbar">
        <div className="brand">
          <span className="logo">✦</span>
          <div>
            <h1>Echo Studio</h1>
            <p className="tagline">
              imagine · local · <b>DreamShaper 8</b>
            </p>
          </div>
        </div>
        <StatusBadge health={health} busy={busy} />
      </header>

      <main>
        <Composer
          values={values}
          busy={busy}
          flashing={flashing}
          onChange={patch}
          onGenerate={runGenerate}
        />

        {error ? (
          <div className="error-banner" role="alert">
            <span>{error}</span>
            <button type="button" onClick={clearError}>
              fechar
            </button>
          </div>
        ) : null}

        <ProgressPanel job={job} visible={Boolean(showProgress && !error)} />

        <section className="gallery-head">
          <h2>Suas criações</h2>
          <div className="gallery-actions">
            {images.length > 0 ? (
              <button type="button" className="btn-ghost small danger" onClick={() => void deleteAll()}>
                ⌫ apagar todas
              </button>
            ) : null}
            <button type="button" className="btn-ghost small" onClick={() => void reload()}>
              ↻ atualizar
            </button>
          </div>
        </section>
        <Gallery
          images={images}
          pendingCount={pendingCount}
          onOpen={(image) => setLightbox(image)}
          onReuse={(image) => applyParams(image, true)}
          onVary={(image) => {
            applyParams(image, false);
            window.setTimeout(() => {
              void generate(
                toPayload({
                  ...values,
                  prompt: image.prompt ?? values.prompt,
                  negative: image.negative_prompt ?? values.negative,
                  seed: "",
                  steps: image.steps ?? values.steps,
                  guidance: image.guidance ?? values.guidance,
                }),
              );
            }, 0);
          }}
          onUseRef={(url) => {
            void useAsReference(url);
          }}
          onDelete={(image) => {
            void deleteImage(image);
          }}
        />
      </main>

      <Lightbox
        image={lightbox}
        seed={lightbox ? seedOf(lightbox) : ""}
        onClose={() => setLightbox(null)}
        onUseRef={(url) => {
          void useAsReference(url);
        }}
        onDelete={(image) => {
          void deleteImage(image);
        }}
      />

      <footer>
        <span>
          Modelo local <b>DreamShaper 8</b> (SD 1.5) · sem filtro · nada sai desta máquina.
        </span>
      </footer>
    </>
  );
}
