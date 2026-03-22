const API_PORT = new URLSearchParams(window.location.search).get("port") || "8765";
const API_BASE = `http://127.0.0.1:${API_PORT}`;

const searchInput = document.getElementById("search");
const statusEl = document.getElementById("status");
const gridEl = document.getElementById("grid");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxClose = document.getElementById("lightbox-close");

/** @type {{ id: string, relativePath: string, url: string }[]} */
let allPhotos = [];

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#f87171" : "";
}

async function loadPhotos() {
  const q = searchInput.value.trim();
  const url = new URL("/api/photos", API_BASE);
  if (q) url.searchParams.set("q", q);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const data = await res.json();
  return data.photos || [];
}

function renderGrid(photos) {
  gridEl.replaceChildren();
  for (const p of photos) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "thumb";
    btn.setAttribute("aria-label", `Open ${p.relativePath}`);
    const img = document.createElement("img");
    img.src = `${API_BASE}${p.url}`;
    img.alt = p.relativePath;
    img.loading = "lazy";
    btn.appendChild(img);
    btn.addEventListener("click", () => openLightbox(p));
    gridEl.appendChild(btn);
  }
}

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

lightbox.addEventListener("click", (e) => {
  if (e.target === lightbox) closeLightbox();
});

window.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && !lightbox.hidden) {
    e.preventDefault();
    closeLightbox();
  }
});

searchInput.addEventListener("input", async () => {
  try {
    allPhotos = await loadPhotos();
    renderGrid(allPhotos);
  } catch (err) {
    setStatus(String(err.message || err), true);
  }
});

(async () => {
  try {
    allPhotos = await loadPhotos();
    if (allPhotos.length === 0) {
      setStatus("No photos in the database. Add entries to desktop/photos_db.json and images under the photos folder.");
      gridEl.hidden = true;
    } else {
      statusEl.hidden = true;
      gridEl.hidden = false;
      renderGrid(allPhotos);
    }
  } catch (err) {
    setStatus(`Could not load photos: ${err.message || err}`, true);
    gridEl.hidden = true;
  }
})();
