// Echo Studio — frontend logic (vanilla JS, no build step).
const $ = (id) => document.getElementById(id);

const state = {
  width: 1024,
  height: 1024,
  count: 1,
  polling: null,
  initImage: null, // base64 data URL of the reference image, or null
};

// ---- Status ---------------------------------------------------------------
async function refreshHealth() {
  try {
    const r = await fetch("/api/health");
    const h = await r.json();
    const dot = $("statusDot");
    const txt = $("statusText");
    if (!h.gguf_present) {
      dot.className = "dot bad";
      txt.textContent = "modelo Chroma Q4 ausente";
    } else if (h.model_loaded) {
      dot.className = "dot ok";
      txt.textContent = `pronto · ${h.device} · ${h.dtype}`;
    } else {
      dot.className = "dot";
      txt.textContent = "modelo não carregado";
    }
  } catch {
    $("statusDot").className = "dot bad";
    $("statusText").textContent = "servidor offline";
  }
}

// ---- Controls -------------------------------------------------------------
$("ratios").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  [...$("ratios").children].forEach((c) => c.classList.remove("active"));
  b.classList.add("active");
  state.width = +b.dataset.w;
  state.height = +b.dataset.h;
});

$("countSeg").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  [...$("countSeg").children].forEach((c) => c.classList.remove("active"));
  b.classList.add("active");
  state.count = +b.dataset.n;
});

$("advBtn").addEventListener("click", () => {
  const a = $("advanced");
  a.hidden = !a.hidden;
});

$("steps").addEventListener("input", (e) => ($("stepsVal").textContent = e.target.value));
$("guidance").addEventListener("input", (e) => ($("guidanceVal").textContent = (+e.target.value).toFixed(1)));

$("prompt").addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate();
});

$("generate").addEventListener("click", generate);
$("refresh").addEventListener("click", loadGallery);

$("strength").addEventListener("input", (e) => ($("strengthVal").textContent = e.target.value + "%"));

// ---- Image upload / drag & drop ------------------------------------------
const MAX_UPLOAD_MB = 12;

$("dropEmpty").addEventListener("click", () => $("fileInput").click());
$("fileInput").addEventListener("change", (e) => {
  if (e.target.files && e.target.files[0]) loadImageFile(e.target.files[0]);
  e.target.value = "";
});
$("removeImg").addEventListener("click", clearImage);

const composer = $("composer");
["dragenter", "dragover"].forEach((ev) =>
  composer.addEventListener(ev, (e) => {
    e.preventDefault();
    composer.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((ev) =>
  composer.addEventListener(ev, (e) => {
    e.preventDefault();
    if (ev === "dragleave" && composer.contains(e.relatedTarget)) return;
    composer.classList.remove("dragover");
  })
);
composer.addEventListener("drop", (e) => {
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) loadImageFile(f);
});

// Paste an image straight from the clipboard.
document.addEventListener("paste", (e) => {
  const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith("image/"));
  if (item) loadImageFile(item.getAsFile());
});

function loadImageFile(file) {
  if (!file.type.startsWith("image/")) return;
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    alert(`Imagem muito grande (máx. ${MAX_UPLOAD_MB} MB).`);
    return;
  }
  const reader = new FileReader();
  reader.onload = () => setReference(reader.result);
  reader.readAsDataURL(file);
}

// Apply a data URL as the active reference image (used by upload & gallery).
function setReference(dataUrl) {
  state.initImage = dataUrl;
  $("refThumb").src = dataUrl;
  $("dropEmpty").hidden = true;
  $("dropFilled").hidden = false;
  $("strengthRow").hidden = false;
  $("generate").querySelector(".gen-label").textContent = "Transformar";
}

// Apply prompt + steps/guidance/negative from an image. `keepSeed` controls
// whether the original seed is reused (reuse) or cleared for a random one (vary).
function applyParams(img, keepSeed) {
  if (img.prompt) $("prompt").value = img.prompt;
  if (img.negative_prompt != null) $("negative").value = img.negative_prompt;
  $("seed").value = keepSeed && img.seed != null ? img.seed : "";
  if (img.steps != null) {
    $("steps").value = img.steps;
    $("stepsVal").textContent = img.steps;
  }
  if (img.guidance != null) {
    $("guidance").value = img.guidance;
    $("guidanceVal").textContent = (+img.guidance).toFixed(1);
  }
}

// Reuse the exact params (same seed) — populates the composer to edit & run.
function reuseParams(img) {
  applyParams(img, true);
  window.scrollTo({ top: 0, behavior: "smooth" });
  $("prompt").focus();
  flashComposer();
}

// Variation: same prompt/params but a fresh random seed — runs immediately.
function varyParams(img) {
  applyParams(img, false);
  window.scrollTo({ top: 0, behavior: "smooth" });
  flashComposer();
  generate();
}

// Brief highlight so it's clear the composer was populated.
function flashComposer() {
  const el = $("composer");
  el.classList.remove("flash");
  void el.offsetWidth; // restart animation
  el.classList.add("flash");
}

// Load a gallery image (served URL) as the reference, then scroll up to compose.
async function useAsReference(url) {
  try {
    const resp = await fetch(url);
    const blob = await resp.blob();
    const reader = new FileReader();
    reader.onload = () => {
      setReference(reader.result);
      $("lightbox").hidden = true;
      window.scrollTo({ top: 0, behavior: "smooth" });
      $("prompt").focus();
    };
    reader.readAsDataURL(blob);
  } catch (e) {
    alert("Não consegui carregar a imagem: " + e.message);
  }
}

function clearImage() {
  state.initImage = null;
  $("refThumb").src = "";
  $("dropEmpty").hidden = false;
  $("dropFilled").hidden = true;
  $("strengthRow").hidden = true;
  $("generate").querySelector(".gen-label").textContent = "Gerar";
}

// ---- Generate -------------------------------------------------------------
function setBusy(busy) {
  const btn = $("generate");
  const img = !!state.initImage;
  btn.disabled = busy;
  btn.querySelector(".gen-label").textContent = busy
    ? (img ? "Transformando…" : "Gerando…")
    : (img ? "Transformar" : "Gerar");
  btn.querySelector(".spinner").hidden = !busy;
  $("statusDot").className = busy ? "dot busy" : "dot";
}

async function generate() {
  const prompt = $("prompt").value.trim();
  if (!prompt) {
    $("prompt").focus();
    return;
  }
  const body = {
    prompt,
    negative_prompt: $("negative").value.trim() || null,
    width: state.width,
    height: state.height,
    steps: +$("steps").value,
    guidance: +$("guidance").value,
    seed: $("seed").value === "" ? null : +$("seed").value,
    num_images: state.count,
    init_image: state.initImage,
    strength: state.initImage ? +$("strength").value / 100 : undefined,
  };

  setBusy(true);
  showPending(state.count);
  $("progressWrap").hidden = false;
  updateProgress(0, "Enfileirando…");

  let job;
  try {
    const r = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${r.status}`);
    }
    job = await r.json();
  } catch (e) {
    fail(e.message);
    return;
  }
  poll(job.id);
}

function poll(id) {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    try {
      const r = await fetch(`/api/jobs/${id}`);
      const j = await r.json();

      if (j.state === "loading_model") updateProgress(j.progress, "Carregando modelo Chroma Q4 (primeira vez)…");
      else if (j.state === "running") updateProgress(j.progress, `Gerando · passo ${j.step}/${j.total_steps}`);
      else if (j.state === "queued") updateProgress(0, "Na fila…");

      if (j.state === "done") {
        clearInterval(state.polling);
        updateProgress(1, "Concluído");
        setTimeout(() => ($("progressWrap").hidden = true), 800);
        setBusy(false);
        loadGallery();
      } else if (j.state === "error") {
        clearInterval(state.polling);
        fail(j.error || "erro desconhecido");
      }
    } catch (e) {
      clearInterval(state.polling);
      fail(e.message);
    }
  }, 700);
}

function fail(msg) {
  clearInterval(state.polling);
  setBusy(false);
  updateProgress(0, "");
  $("progressWrap").hidden = true;
  clearPending();
  alert("Falhou: " + msg);
  refreshHealth();
}

function updateProgress(p, label) {
  $("progressFill").style.width = Math.round(p * 100) + "%";
  $("progressPct").textContent = Math.round(p * 100) + "%";
  if (label !== undefined) $("progressLabel").textContent = label;
}

// ---- Gallery --------------------------------------------------------------
let pendingCards = [];

function showPending(n) {
  const g = $("gallery");
  $("empty").hidden = true;
  pendingCards = [];
  for (let i = 0; i < n; i++) {
    const c = document.createElement("div");
    c.className = "card pending";
    g.prepend(c);
    pendingCards.push(c);
  }
}
function clearPending() {
  pendingCards.forEach((c) => c.remove());
  pendingCards = [];
}

async function loadGallery() {
  clearPending();
  try {
    const r = await fetch("/api/gallery");
    const data = await r.json();
    const g = $("gallery");
    g.innerHTML = "";
    if (!data.images.length) {
      $("empty").hidden = false;
      return;
    }
    $("empty").hidden = true;
    for (const img of data.images) {
      const seed = img.seed ?? (img.filename.match(/_(\d+)\.png$/) || [])[1] ?? "";
      const c = document.createElement("div");
      c.className = "card";
      const canReuse = img.prompt || img.seed != null;
      c.innerHTML =
        `<img loading="lazy" src="${img.url}" alt="" />` +
        (img.prompt ? `<div class="card-prompt"></div>` : "") +
        (seed !== "" ? `<span class="seed-tag">seed ${seed}</span>` : "") +
        `<div class="card-actions">` +
          (canReuse ? `<button class="card-btn reuse" title="Reusar prompt e seed">⤺ reusar</button>` : "") +
          (img.prompt ? `<button class="card-btn vary" title="Variação: mesmo prompt, seed aleatória">⚄ variar</button>` : "") +
          `<button class="card-btn use-ref" title="Usar como referência (img2img)">⧉ ref</button>` +
        `</div>`;
      if (img.prompt) {
        // Use textContent so the prompt can't break the markup.
        c.querySelector(".card-prompt").textContent = img.prompt;
        c.title = img.prompt; // full prompt as a native tooltip
      }
      c.querySelector("img").addEventListener("click", () => openLightbox(img.url, seed));
      c.querySelector(".use-ref").addEventListener("click", (e) => {
        e.stopPropagation();
        useAsReference(img.url);
      });
      const reuseBtn = c.querySelector(".reuse");
      if (reuseBtn) reuseBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        reuseParams(img);
      });
      const varyBtn = c.querySelector(".vary");
      if (varyBtn) varyBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        varyParams(img);
      });
      g.appendChild(c);
    }
  } catch {
    /* ignore */
  }
}

// ---- Lightbox -------------------------------------------------------------
function openLightbox(url, seed) {
  $("lbImg").src = url;
  $("lbMeta").innerHTML =
    (seed ? `<span>seed <b>${seed}</b></span>` : "") +
    `<a href="#" id="lbUseRef">⧉ usar como referência</a>` +
    `<a href="${url}" download>⬇ baixar PNG</a>`;
  $("lbUseRef").addEventListener("click", (e) => {
    e.preventDefault();
    useAsReference(url);
  });
  $("lightbox").hidden = false;
}
$("lbClose").addEventListener("click", () => ($("lightbox").hidden = true));
$("lightbox").addEventListener("click", (e) => {
  if (e.target.id === "lightbox") $("lightbox").hidden = true;
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") $("lightbox").hidden = true;
});

// ---- Init -----------------------------------------------------------------
refreshHealth();
loadGallery();
setInterval(refreshHealth, 8000);
