import { useCallback, useRef, type ChangeEvent, type DragEvent, type KeyboardEvent } from "react";
import type { RatioPreset } from "../types/studio";

const RATIOS: RatioPreset[] = [
  { label: "1:1", width: 512, height: 512 },
  { label: "4:3", width: 640, height: 448 },
  { label: "3:2", width: 768, height: 512 },
  { label: "16:9", width: 768, height: 448 },
  { label: "2:3", width: 512, height: 768 },
  { label: "9:16", width: 448, height: 768 },
];

const COUNTS = [1, 2, 4] as const;
const MAX_UPLOAD_MB = 12;

export type ComposerValues = {
  prompt: string;
  negative: string;
  width: number;
  height: number;
  count: number;
  steps: number;
  guidance: number;
  seed: string;
  initImage: string | null;
  strength: number;
  advanced: boolean;
};

type ComposerProps = {
  values: ComposerValues;
  busy: boolean;
  flashing: boolean;
  onChange: (patch: Partial<ComposerValues>) => void;
  onGenerate: () => void;
};

export function Composer({ values, busy, flashing, onChange, onGenerate }: ComposerProps) {
  const fileRef = useRef<HTMLInputElement>(null);
  const isImg2Img = values.initImage !== null;

  const applyFile = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) {
        return;
      }
      if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
        window.alert(`Imagem muito grande (máx. ${MAX_UPLOAD_MB} MB).`);
        return;
      }
      const reader = new FileReader();
      reader.onload = () => {
        if (typeof reader.result === "string") {
          onChange({ initImage: reader.result });
        }
      };
      reader.readAsDataURL(file);
    },
    [onChange],
  );

  const onDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file) {
      applyFile(file);
    }
  };

  const onFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      applyFile(file);
    }
    event.target.value = "";
  };

  const onPromptKey = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      onGenerate();
    }
  };

  return (
    <section
      className={`composer${flashing ? " flash" : ""}`}
      onDragOver={(event) => event.preventDefault()}
      onDrop={onDrop}
    >
      <div className="composer-top">
        <div className="uploader">
          {values.initImage ? (
            <div className="drop-filled">
              <img src={values.initImage} alt="referência" />
              <button
                type="button"
                className="thumb-remove"
                title="Remover imagem"
                onClick={() => onChange({ initImage: null })}
              >
                ✕
              </button>
            </div>
          ) : (
            <button type="button" className="drop-empty" onClick={() => fileRef.current?.click()}>
              <span className="up-ico">⬆</span>
              <span className="up-title">Imagem</span>
              <span className="up-sub">arraste ou clique</span>
            </button>
          )}
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={onFile} />
        </div>
        <textarea
          rows={3}
          placeholder="Descreva o que você quer imaginar…  (ex.: uma raposa de neon numa floresta chuvosa à noite, cinematográfico)"
          value={values.prompt}
          onChange={(event) => onChange({ prompt: event.target.value })}
          onKeyDown={onPromptKey}
        />
      </div>

      {isImg2Img ? (
        <div className="strength-row">
          <label>
            Influência da imagem <b>{values.strength}%</b>
          </label>
          <input
            type="range"
            min={10}
            max={100}
            value={values.strength}
            onChange={(event) => onChange({ strength: Number(event.target.value) })}
          />
          <span className="hint">baixo = fiel à original · alto = mais criativo</span>
        </div>
      ) : null}

      <div className="composer-row">
        <div className="ratios">
          {RATIOS.map((ratio) => {
            const active = values.width === ratio.width && values.height === ratio.height;
            return (
              <button
                key={ratio.label}
                type="button"
                className={`chip${active ? " active" : ""}`}
                onClick={() => onChange({ width: ratio.width, height: ratio.height })}
              >
                {ratio.label}
              </button>
            );
          })}
        </div>
        <div className="grow" />
        <div className="count">
          <label>imagens</label>
          <div className="seg">
            {COUNTS.map((count) => (
              <button
                key={count}
                type="button"
                className={values.count === count ? "active" : ""}
                onClick={() => onChange({ count })}
              >
                {count}
              </button>
            ))}
          </div>
        </div>
        <button
          type="button"
          className="btn-ghost"
          title="Ajustes avançados"
          onClick={() => onChange({ advanced: !values.advanced })}
        >
          ⚙
        </button>
        <button type="button" className="btn-generate" disabled={busy} onClick={onGenerate}>
          <span className="gen-label">
            {busy ? (isImg2Img ? "Transformando…" : "Gerando…") : isImg2Img ? "Transformar" : "Gerar"}
          </span>
          {busy ? <span className="spinner" /> : null}
        </button>
      </div>

      {values.advanced ? (
        <div className="advanced">
          <div className="field">
            <label>Prompt negativo</label>
            <input
              type="text"
              placeholder="(usa o padrão se vazio)"
              value={values.negative}
              onChange={(event) => onChange({ negative: event.target.value })}
            />
          </div>
          <div className="field-inline">
            <div className="field">
              <label>
                Passos <b>{values.steps}</b>
              </label>
              <input
                type="range"
                min={8}
                max={60}
                value={values.steps}
                onChange={(event) => onChange({ steps: Number(event.target.value) })}
              />
            </div>
            <div className="field">
              <label>
                Guidance <b>{values.guidance.toFixed(1)}</b>
              </label>
              <input
                type="range"
                min={0}
                max={10}
                step={0.5}
                value={values.guidance}
                onChange={(event) => onChange({ guidance: Number(event.target.value) })}
              />
            </div>
            <div className="field seed-field">
              <label>Seed</label>
              <input
                type="number"
                min={0}
                placeholder="aleatória"
                value={values.seed}
                onChange={(event) => onChange({ seed: event.target.value })}
              />
            </div>
          </div>
        </div>
      ) : null}
    </section>
  );
}
