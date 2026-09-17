import { formatDuration } from "../lib/formatDuration";
import type { GalleryImage } from "../types/studio";

type LightboxProps = {
  image: GalleryImage | null;
  seed: string;
  onClose: () => void;
  onUseRef: (url: string) => void;
  onDelete: (image: GalleryImage) => void;
};

export function Lightbox({ image, seed, onClose, onUseRef, onDelete }: LightboxProps) {
  if (!image) {
    return null;
  }
  return (
    <div
      className="lightbox"
      onClick={(event) => {
        if (event.target === event.currentTarget) {
          onClose();
        }
      }}
    >
      <button type="button" className="lb-close" onClick={onClose}>
        ✕
      </button>
      <img src={image.url} alt="" />
      <div className="lb-meta">
        {seed ? (
          <span>
            seed <b>{seed}</b>
          </span>
        ) : null}
        {image.elapsed != null ? (
          <span>
            gerou em <b>{formatDuration(image.elapsed)}</b>
          </span>
        ) : null}
        <a
          href="#"
          onClick={(event) => {
            event.preventDefault();
            onUseRef(image.url);
          }}
        >
          ⧉ usar como referência
        </a>
        <a href={image.url} download={image.filename}>
          ⬇ baixar PNG
        </a>
        <button type="button" className="lb-delete" onClick={() => onDelete(image)}>
          ⌫ apagar
        </button>
      </div>
    </div>
  );
}
