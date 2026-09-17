import { formatDuration } from "../lib/formatDuration";
import type { GalleryImage } from "../types/studio";

type GalleryProps = {
  images: GalleryImage[];
  pendingCount: number;
  onOpen: (image: GalleryImage) => void;
  onReuse: (image: GalleryImage) => void;
  onVary: (image: GalleryImage) => void;
  onUseRef: (url: string) => void;
  onDelete: (image: GalleryImage) => void;
};

function seedOf(image: GalleryImage): string {
  if (image.seed != null) {
    return String(image.seed);
  }
  const match = image.filename.match(/_(\d+)\.png$/);
  return match?.[1] ?? "";
}

export function Gallery({
  images,
  pendingCount,
  onOpen,
  onReuse,
  onVary,
  onUseRef,
  onDelete,
}: GalleryProps) {
  const empty = images.length === 0 && pendingCount === 0;
  return (
    <>
      <section className="gallery">
        {Array.from({ length: pendingCount }, (_, index) => (
          <div key={`pending-${index}`} className="card pending" />
        ))}
        {images.map((image) => {
          const seed = seedOf(image);
          const canReuse = Boolean(image.prompt) || image.seed != null;
          return (
            <div key={image.filename} className="card">
              <img
                loading="lazy"
                src={image.url}
                alt=""
                onClick={() => onOpen(image)}
              />
              {image.prompt ? <div className="card-prompt">{image.prompt}</div> : null}
              {seed || image.elapsed != null ? (
                <div className="card-meta">
                  {seed ? <span className="seed-tag">seed {seed}</span> : null}
                  {image.elapsed != null ? (
                    <span className="seed-tag" title="Tempo de geração">
                      {formatDuration(image.elapsed)}
                    </span>
                  ) : null}
                </div>
              ) : null}
              <div className="card-actions">
                {canReuse ? (
                  <button
                    type="button"
                    className="card-btn"
                    title="Reusar prompt e seed"
                    onClick={() => onReuse(image)}
                  >
                     ↶ reusar
                  </button>
                ) : null}
                {image.prompt ? (
                  <button
                    type="button"
                    className="card-btn"
                    title="Variação: mesmo prompt, seed aleatória"
                    onClick={() => onVary(image)}
                  >
                    ⚄ variar
                  </button>
                ) : null}
                <button
                  type="button"
                  className="card-btn"
                  title="Usar como referência (img2img)"
                  onClick={() => onUseRef(image.url)}
                >
                  ⧉ ref
                </button>
                <button
                  type="button"
                  className="card-btn danger"
                  title="Apagar esta criação"
                  onClick={() => onDelete(image)}
                >
                  ⌫ apagar
                </button>
              </div>
            </div>
          );
        })}
      </section>
      {empty ? (
        <p className="empty">
          Nada por aqui ainda. Escreva um prompt e clique em <b>Gerar</b>.
        </p>
      ) : null}
    </>
  );
}
