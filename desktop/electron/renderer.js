const API_PORT = new URLSearchParams(window.location.search).get("port") || "8765";
const API_BASE = `http://127.0.0.1:${API_PORT}`;

// ── DOM refs ──────────────────────────────────────────────────────────────
const searchInput   = document.getElementById("search");
const modelSelect   = document.getElementById("model-select");
const btnIndex      = document.getElementById("btn-index");
const statusEl      = document.getElementById("status");
const gridEl        = document.getElementById("grid");
const resultsPanel  = document.getElementById("results-panel");
const resultsQuery  = document.getElementById("results-query");
const resultsEyebrow = document.getElementById("results-eyebrow");
const resultsBadge  = document.getElementById("results-badge");
const resultsCount  = document.getElementById("results-count");
const resultsEmpty  = document.getElementById("results-empty");
const resultsEmptyTitle = document.getElementById("results-empty-title");
const resultsEmptyHint = document.getElementById("results-empty-hint");
const lightbox      = document.getElementById("lightbox");
const lightboxImg   = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");
const indexModal       = document.getElementById("index-modal");
const modalClose       = document.getElementById("modal-close");
const modalModelFilter = document.getElementById("modal-model-filter");
const deviceLabel      = document.getElementById("device-label");
const modelCards       = document.getElementById("model-cards");
const paginationEl  = document.getElementById("pagination");
const pagePrev      = document.getElementById("page-prev");
const pageNext      = document.getElementById("page-next");
const pageInfo      = document.getElementById("page-info");

/** Thumbnails per page (homepage grid). */
const PAGE_SIZE = 48;

/** @type {{ id: string, relativePath: string, url: string, thumbnailUrl: string, score?: number | null }[]} */
let allPhotos = [];

/** 1-based current page */
let currentPage = 1;

/** Last API response used for empty-state copy */
let lastSemantic = false;

/** Polling handle for the index modal */
let _pollInterval = null;

/** Last `/api/index/models` payload (for filter / sort without re-fetching). */
let _lastIndexModelsData = null;

/** @type {"recency" | "size" | "downloaded"} */
let _indexSort = "recency";

/** FAISS inner product on L2-normalized CLIP vectors (= cosine similarity), range about [-1, 1]. */
function formatRawScore(score) {
  return score.toFixed(3);
}

function similarityTooltip(score) {
  return `Similarity score ${formatRawScore(score)} (cosine / inner product, higher is closer)`;
}

/** Badge text for thumbnail, or empty when not applicable. */
function formatSimilarityLabel(photo, semantic) {
  if (!semantic || typeof photo.score !== "number") return "";
  return formatRawScore(photo.score);
}

// ── Status bar ────────────────────────────────────────────────────────────

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "var(--red)" : "";
  statusEl.hidden = !text;
}

// ── Photos ────────────────────────────────────────────────────────────────

async function loadPhotos() {
  const q     = searchInput.value.trim();
  const model = modelSelect.value;
  const url   = new URL("/api/photos", API_BASE);
  if (q)     url.searchParams.set("q", q);
  if (model) url.searchParams.set("model", model);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return { photos: data.photos || [], semantic: data.semantic || false };
}

function totalPages() {
  return Math.max(1, Math.ceil(allPhotos.length / PAGE_SIZE));
}

function clampCurrentPage() {
  const tp = totalPages();
  if (currentPage > tp) currentPage = tp;
  if (currentPage < 1) currentPage = 1;
}

/** Render the current page of `allPhotos` and update pagination controls. */
function renderView() {
  const q = searchInput.value.trim();
  clampCurrentPage();
  gridEl.replaceChildren();

  if (!q) {
    document.body.classList.remove("has-results");
    setStatus("");
    resultsPanel.hidden = true;
    resultsEmpty.hidden = true;
    paginationEl.hidden = true;
    return;
  }

  document.body.classList.add("has-results");
  resultsPanel.hidden = false;
  resultsPanel.classList.toggle("results-panel--semantic", lastSemantic);
  resultsQuery.textContent = q;
  resultsEyebrow.textContent = lastSemantic ? "Neural similarity" : "Library scan";
  resultsBadge.textContent = lastSemantic ? "CLIP / vector space" : "All indexed photos";
  resultsBadge.classList.toggle("results-badge--semantic", lastSemantic);
  const n = allPhotos.length;
  resultsCount.textContent =
    n === 0 ? "0 matches" : `${n} match${n === 1 ? "" : "es"} · page view`;

  if (allPhotos.length === 0) {
    setStatus("");
    if (lastSemantic) {
      resultsEmptyTitle.textContent = "No visual hits";
      resultsEmptyHint.textContent = "Try another scene, object, color, or mood in your prompt.";
    } else {
      resultsEmptyTitle.textContent = "Library is empty";
      resultsEmptyHint.textContent = "Add entries to desktop/photos_db.json and point photosRoot at your folder.";
    }
    resultsEmpty.hidden = false;
    gridEl.hidden = true;
    paginationEl.hidden = true;
    return;
  }

  setStatus("");
  resultsEmpty.hidden = true;
  gridEl.hidden = false;
  paginationEl.hidden = false;

  const start = (currentPage - 1) * PAGE_SIZE;
  const end = Math.min(start + PAGE_SIZE, allPhotos.length);
  const slice = allPhotos.slice(start, end);

  slice.forEach((p, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb thumb--reveal";
    btn.style.setProperty("--reveal-i", String(Math.min(i, 40)));
    const scoreLabel = formatSimilarityLabel(p, lastSemantic);
    btn.setAttribute(
      "aria-label",
      scoreLabel ? `Open ${p.relativePath}, similarity ${scoreLabel}` : `Open ${p.relativePath}`,
    );
    const img = document.createElement("img");
    const thumb = p.thumbnailUrl || p.url;
    img.src = `${API_BASE}${thumb}`;
    img.alt = p.relativePath;
    img.loading = "lazy";
    img.decoding = "async";
    btn.appendChild(img);
    if (scoreLabel) {
      const badge = document.createElement("span");
      badge.className = "thumb-score";
      badge.textContent = scoreLabel;
      if (typeof p.score === "number") {
        badge.title = similarityTooltip(p.score);
      }
      btn.appendChild(badge);
    }
    btn.addEventListener("click", () => openLightbox(p));
    gridEl.appendChild(btn);
  });

  const tp = totalPages();
  pageInfo.textContent = `Page ${currentPage} of ${tp} · ${start + 1}–${end} of ${allPhotos.length}`;
  pagePrev.disabled = currentPage <= 1;
  pageNext.disabled = currentPage >= tp;
}

async function refreshPhotos() {
  const q = searchInput.value.trim();
  if (!q) {
    allPhotos = [];
    lastSemantic = false;
    currentPage = 1;
    renderView();
    return;
  }

  try {
    const { photos, semantic } = await loadPhotos();
    allPhotos = photos;
    lastSemantic = semantic;
    currentPage = 1;
    renderView();
  } catch (err) {
    allPhotos = [];
    setStatus(String(err.message || err), true);
    resultsPanel.hidden = true;
    resultsEmpty.hidden = true;
    gridEl.hidden = true;
    paginationEl.hidden = true;
    gridEl.replaceChildren();
  }
}

// ── Lightbox ──────────────────────────────────────────────────────────────

function openLightbox(photo) {
  lightboxImg.src = `${API_BASE}${photo.url}`;
  lightboxImg.alt = photo.relativePath;
  lightbox.hidden = false;
  lightboxClose.focus();
  document.body.style.overflow = "hidden";
}

function closeLightbox() {
  lightbox.hidden = true;
  lightboxImg.removeAttribute("src");
  document.body.style.overflow = "";
  searchInput.focus();
}

lightboxClose.addEventListener("click", closeLightbox);
lightbox.addEventListener("click", (e) => { if (e.target === lightbox) closeLightbox(); });
window.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    if (!lightbox.hidden)    { e.preventDefault(); closeLightbox(); return; }
    if (!indexModal.hidden)  { e.preventDefault(); closeIndexModal(); }
  }
});

// ── Model selector ────────────────────────────────────────────────────────

function formatDiskGb(sizeMb) {
  if (sizeMb >= 1024) return `${(sizeMb / 1024).toFixed(2)} GB`;
  return `${sizeMb} MB`;
}

function populateModelSelector(models) {
  // Remove existing model options (keep the "show all" placeholder)
  while (modelSelect.options.length > 1) modelSelect.remove(1);

  for (const m of models) {
    if (m.indexed_count === 0) continue;  // Only show indexed models
    const opt = document.createElement("option");
    opt.value = m.id;
    const storage = formatDiskGb(m.size_mb);
    opt.textContent = `${m.name} · ${storage} · ${m.indexed_count} photos`;
    opt.title = `${m.name} — weights on disk about ${storage}; ${m.indexed_count} photos in this index`;
    modelSelect.appendChild(opt);
  }
}

modelSelect.addEventListener("change", refreshPhotos);
searchInput.addEventListener("input",  refreshPhotos);

document.querySelector(".suggestion-row")?.addEventListener("click", (e) => {
  const btn = e.target.closest(".suggest-pill");
  if (!btn?.dataset.suggest) return;
  searchInput.value = btn.dataset.suggest;
  searchInput.focus();
  refreshPhotos();
});

pagePrev.addEventListener("click", () => {
  if (currentPage <= 1) return;
  currentPage -= 1;
  renderView();
  if (!gridEl.hidden) gridEl.scrollIntoView({ block: "start", behavior: "smooth" });
});

pageNext.addEventListener("click", () => {
  if (currentPage >= totalPages()) return;
  currentPage += 1;
  renderView();
  if (!gridEl.hidden) gridEl.scrollIntoView({ block: "start", behavior: "smooth" });
});

// ── Index Modal ───────────────────────────────────────────────────────────

async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/index/models`);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

function stateLabel(m) {
  if (m.state === "indexing") {
    return `Indexing… ${m.done} / ${m.total}`;
  }
  if (m.state === "error") {
    return `Error: ${m.error}`;
  }
  if (m.indexed_count > 0) {
    return `${m.indexed_count} photos indexed`;
  }
  return "Not indexed";
}

function stateDotClass(m) {
  if (m.state === "indexing") return "indexing";
  if (m.state === "error")    return "error";
  if (m.indexed_count > 0)    return "done";
  return "idle";
}

/** Short spec chip from display name, e.g. ViT-B/32 → B/32 */
function modelQuantSpec(name) {
  const match = name.match(/ViT-([^,\s]+)/i);
  return match ? match[1].toUpperCase() : "ViT";
}

function modelProviderLower(m) {
  if (m.model_type === "siglip") return "google";
  if (m.model_type === "clip") return "openai";
  return m.id.split("-")[0] || "—";
}

const LM_VISION_ICON = `<svg class="lm-cap-icon-svg" width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true"><path d="M12 5C7 5 2.73 8.11 1 12c1.73 3.89 6 7 11 7s9.27-3.11 11-7c-1.73-3.89-6-7-11-7zm0 12a5 5 0 1 1 0-10 5 5 0 0 1 0 10zm0-8a3 3 0 1 0 0 6 3 3 0 0 0 0-6z" fill="currentColor"/></svg>`;

function renderModelCards(data) {
  _lastIndexModelsData = data;
  const filterQ = (modalModelFilter && modalModelFilter.value.trim().toLowerCase()) || "";
  let models = data.models.filter(
    (m) =>
      !filterQ ||
      m.name.toLowerCase().includes(filterQ) ||
      m.id.toLowerCase().includes(filterQ) ||
      (m.description && m.description.toLowerCase().includes(filterQ)),
  );

  if (_indexSort === "size") {
    models = [...models].sort((a, b) => b.size_mb - a.size_mb || a.name.localeCompare(b.name));
  } else if (_indexSort === "downloaded") {
    models = [...models].sort((a, b) => {
      const ai = a.indexed_count > 0 ? 1 : 0;
      const bi = b.indexed_count > 0 ? 1 : 0;
      if (bi !== ai) return bi - ai;
      return a.name.localeCompare(b.name);
    });
  } else {
    models = [...models].sort((a, b) => {
      if (b.indexed_count !== a.indexed_count) return b.indexed_count - a.indexed_count;
      return a.name.localeCompare(b.name);
    });
  }

  deviceLabel.textContent = `Device: ${data.device}`;
  modelCards.replaceChildren();

  for (const m of models) {
    const row = document.createElement("div");
    const active = m.state === "indexing";
    const err = m.state === "error";
    row.className = `model-row lm-model-row${active ? " model-row--active" : ""}${err ? " model-row--error" : ""}`;
    row.dataset.modelId = m.id;

    const pct = m.total > 0 ? Math.round((m.done / m.total) * 100) : 0;
    const progressHtml = m.state === "indexing"
      ? `<div class="lm-progress-track lm-progress-track--inline"><div class="lm-progress-fill" style="width:${pct}%"></div></div>`
      : "";

    const btnLabel =
      m.state === "indexing" ? "Indexing…" : m.indexed_count > 0 ? "Re-index" : "Index";
    const btnDisabled = m.state === "indexing" ? "disabled" : "";

    const quant = modelQuantSpec(m.name);
    const provider = modelProviderLower(m);
    const familyClass =
      m.model_type === "siglip" ? "lm-ml-family lm-ml-family--siglip" : "lm-ml-family lm-ml-family--clip";
    const familyLabel = m.model_type === "clip" ? "clip" : "siglip";

    row.innerHTML = `
      <div class="lm-model-line" role="group" aria-label="${m.name}">
        <span class="lm-ml-name" title="${m.id}">${m.name}</span>
        <span class="lm-ml-chip lm-ml-chip--quant" title="Architecture">${quant}</span>
        <span class="lm-ml-cap lm-ml-cap--vision" title="Vision embedding">${LM_VISION_ICON}</span>
        <span class="lm-ml-provider">${provider}</span>
        <span class="lm-ml-params">${m.embedding_dim}D</span>
        <span class="${familyClass}">${familyLabel}</span>
        <span class="lm-ml-format">HF</span>
        <span class="lm-ml-size">${formatDiskGb(m.size_mb)}</span>
        <span class="lm-ml-tail">
          <span class="lm-ml-status" title="${stateLabel(m)}">
            <span class="lm-status-dot ${stateDotClass(m)}" aria-hidden="true"></span>
          </span>
          <button type="button" class="btn-lm-action btn-lm-action--row" data-model="${m.id}" ${btnDisabled}>${btnLabel}</button>
          <span class="lm-ml-chevron" aria-hidden="true">›</span>
        </span>
      </div>
      <p class="lm-model-desc">${m.description}</p>
      ${progressHtml}
    `;
    modelCards.appendChild(row);
  }

  modelCards.querySelectorAll(".btn-lm-action").forEach((btn) => {
    btn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      startIndexing(btn.dataset.model);
    });
  });
}

async function startIndexing(modelId) {
  try {
    const res = await fetch(`${API_BASE}/api/index/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      alert(`Failed to start indexing: ${err.detail || res.status}`);
      return;
    }
    // Refresh card state immediately
    await refreshModalCards();
  } catch (err) {
    alert(`Error: ${err.message || err}`);
  }
}

async function refreshModalCards() {
  try {
    const data = await fetchModels();
    renderModelCards(data);
    populateModelSelector(data.models);
  } catch { /* ignore */ }
}

function openIndexModal() {
  indexModal.hidden = false;
  document.body.style.overflow = "hidden";
  if (modalModelFilter) modalModelFilter.value = "";
  _indexSort = "recency";
  indexModal.querySelectorAll(".lm-sort-btn").forEach((b) => {
    b.classList.toggle("lm-sort-btn--active", b.dataset.sort === "recency");
  });
  refreshModalCards();
  queueMicrotask(() => modalModelFilter?.focus());
  _pollInterval = setInterval(refreshModalCards, 1500);
}

function closeIndexModal() {
  indexModal.hidden = true;
  document.body.style.overflow = "";
  if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
  // Refresh photo grid in case new models are now indexed
  refreshPhotos();
}

btnIndex.addEventListener("click", openIndexModal);
modalClose.addEventListener("click", closeIndexModal);
indexModal.addEventListener("click", (e) => { if (e.target === indexModal) closeIndexModal(); });

indexModal.addEventListener("click", (e) => {
  const sortBtn = e.target.closest(".lm-sort-btn");
  if (!sortBtn || !sortBtn.dataset.sort) return;
  _indexSort = /** @type {"recency" | "size" | "downloaded"} */ (sortBtn.dataset.sort);
  indexModal.querySelectorAll(".lm-sort-btn").forEach((b) => {
    b.classList.toggle("lm-sort-btn--active", b === sortBtn);
  });
  if (_lastIndexModelsData) renderModelCards(_lastIndexModelsData);
});

modalModelFilter?.addEventListener("input", () => {
  if (_lastIndexModelsData) renderModelCards(_lastIndexModelsData);
});

// ── Bootstrap ─────────────────────────────────────────────────────────────

const PLACEHOLDER_ROTATION = [
  "Try: golden hour on the porch, messy kitchen birthday…",
  "Try: dog in snow, red balloons, someone laughing…",
  "Try: beach sunset, cake candles, kids on a swing…",
  "Try: wedding dance floor, hiking trail, neon city night…",
];

let _placeholderIndex = 0;
let _placeholderTimer = null;

function startPlaceholderRotation() {
  if (_placeholderTimer != null) return;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  _placeholderTimer = window.setInterval(() => {
    if (document.activeElement === searchInput || searchInput.value.trim()) return;
    _placeholderIndex = (_placeholderIndex + 1) % PLACEHOLDER_ROTATION.length;
    searchInput.placeholder = PLACEHOLDER_ROTATION[_placeholderIndex];
  }, 5200);
}

(async () => {
  try {
    const data = await fetchModels();
    populateModelSelector(data.models);
  } catch { /* non-fatal */ }

  renderView();
  searchInput.focus();
  startPlaceholderRotation();
})();
