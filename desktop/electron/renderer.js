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
const indexModal    = document.getElementById("index-modal");
const modalClose    = document.getElementById("modal-close");
const deviceLabel   = document.getElementById("device-label");
const modelCards    = document.getElementById("model-cards");
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

function populateModelSelector(models) {
  // Remove existing model options (keep the "show all" placeholder)
  while (modelSelect.options.length > 1) modelSelect.remove(1);

  for (const m of models) {
    if (m.indexed_count === 0) continue;  // Only show indexed models
    const opt = document.createElement("option");
    opt.value = m.id;
    opt.textContent = `${m.name} (${m.indexed_count} photos)`;
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

function renderModelCards(data) {
  deviceLabel.textContent = `Running on: ${data.device}`;
  modelCards.replaceChildren();

  for (const m of data.models) {
    const card = document.createElement("div");
    card.className = "model-card";
    card.dataset.modelId = m.id;

    const pct = m.total > 0 ? Math.round((m.done / m.total) * 100) : 0;
    const progressHtml = m.state === "indexing"
      ? `<div class="progress-bar-wrap"><div class="progress-bar" style="width:${pct}%"></div></div>`
      : "";

    const btnLabel  = m.state === "indexing"  ? "Indexing…"
                    : m.indexed_count > 0     ? "Re-index"
                    : "Start Indexing";
    const btnDisabled = m.state === "indexing" ? "disabled" : "";

    card.innerHTML = `
      <div class="model-card-header">
        <span class="model-card-name">${m.name}</span>
        <span class="badge badge-${m.model_type}">${m.model_type.toUpperCase()}</span>
      </div>
      <p class="model-card-desc">${m.description}</p>
      <p class="model-card-meta">${m.embedding_dim}-dim · ~${m.size_mb} MB download</p>
      <div class="model-card-footer">
        <span class="model-status">
          <span class="status-dot ${stateDotClass(m)}"></span>${stateLabel(m)}
        </span>
        <button class="btn-start" data-model="${m.id}" ${btnDisabled}>${btnLabel}</button>
      </div>
      ${progressHtml}
    `;
    modelCards.appendChild(card);
  }

  // Wire up start buttons
  modelCards.querySelectorAll(".btn-start").forEach((btn) => {
    btn.addEventListener("click", () => startIndexing(btn.dataset.model));
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
  refreshModalCards();
  // Poll while modal is open
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
