// Echo Studio — frontend logic (vanilla JS, no build step).
const $ = (id) => document.getElementById(id);

const state = {
  width: 1024,
  height: 1024,
  count: 1,
  polling: null,
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

// ---- Generate -------------------------------------------------------------
function setBusy(busy) {
  const btn = $("generate");
  btn.disabled = busy;
  btn.querySelector(".gen-label").textContent = busy ? "Imaginando…" : "Imaginar";
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
      const seed = (img.filename.match(/_(\d+)\.png$/) || [])[1] || "";
      const c = document.createElement("div");
      c.className = "card";
      c.innerHTML = `<img loading="lazy" src="${img.url}" alt="" />` +
        (seed ? `<span class="seed-tag">seed ${seed}</span>` : "");
      c.addEventListener("click", () => openLightbox(img.url, seed));
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
    `<a href="${url}" download>⬇ baixar PNG</a>`;
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
