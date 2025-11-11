# README

# Topical Centre & Semantic Drift Visualiser

A Streamlit-based tool for analysing a website’s **semantic cohesion** and **topical drift** using embeddings, internal link data, and performance metrics.

---

## Project Overview

This visualiser helps SEO teams and content strategists identify:

- Pages that drift away from the site’s topical centre
- Pages that are semantically aligned but underlinked
- Pages that are semantically distant but performing well
- Areas of topical opportunity or dilution

The radial “orbit” map displays each page’s relationship to your site’s topical core — combining **semantic**, **structural**, and **performance** signals.

---

## Project Structure

```
app.py
requirements.txt
core/
  data_loader.py
  metrics.py
  processing.py
  projection.py
  radial_layout.py
ui/
  layout.py
  visuals.py
utils/
  logger.py
  normalise.py
  parser.py

```

### What Each File Does

- **app.py** — Streamlit entrypoint; wires UI → core → chart
- **core/data_loader.py** — reads Screaming Frog CSV + GSC CSV, normalises URLs, merges
- **core/processing.py** — builds semantic centroid, adds similarity / distance columns
- **core/projection.py** — runs UMAP and creates x/y coordinates
- **core/radial_layout.py** — helper for orbit-style plotting (polar → cartesian)
- **core/metrics.py** — site-level KPIs (cohesion, % in centre, average drift)
- **ui/layout.py** — sidebar controls
- **ui/visuals.py** — Altair radial chart (interactive visualisation)
- **utils/** — small helpers (logging, normalisation, parsing)

---

## Installation

### 1. Clone the repo

```
git clone https://github.com/<your-username>/topical-centre-visualiser.git
cd topical-centre-visualiser

```

### 2. Create and activate a virtual environment (Windows)

```
python -m venv .venv
.venv\Scripts\activate

```

### 3. Install dependencies

```
pip install -r requirements.txt

```

> 💡 You can prune requirements.txt later if the freeze was too large.

---

## Running the App

```
streamlit run app.py

```

Then open your browser at:

[**http://localhost:8501**](http://localhost:8501/)

Upload the two CSVs (Screaming Frog + GSC), and the chart will render automatically.

---

## Required Inputs

The app expects **two input files**.

### 1. Screaming Frog CSV

Must contain at least:

- `Address`
- An embeddings column (e.g. `OpenAI Embeddings 1_x` / `_y`, normalised in your loader)
- `Inlinks`

Other crawl metadata is ignored.

This is your “site structure + vectors” file.

### 2. GSC CSV

Must contain:

- `Page` (URL)
- `Clicks`
- Optionally: `Impressions`, `CTR`, `Position`

This is the “performance” file. It’s merged onto the crawl file by URL (lowercased, trimmed, and normalised).

---

## How It Works

### 1. Load

- `core/data_loader.py` reads both CSVs
- Normalises URLs (lowercase, strip `/`)
- Merges → single DataFrame with: URL, embeddings, inlinks, clicks

### 2. Centroid

- `core/processing.py` stacks embeddings
- Computes mean → “topical centre”
- Each page gets a `distance_from_centre` value

### 3. Projection

- `core/projection.py` runs UMAP (cosine)
- Produces 2D coordinates centred around the centroid

### 4. Radial Layout

- `ui/visuals.py` scales semantic distance 0–1 (radius)
- Assigns evenly spaced angles → orbit positions

### 5. Styling

- **Bubble size:** Clicks
- **Opacity:** Inlinks (with minor clicks weighting)
- **Colour:** Structural Drift Index (SDI) — “Topical Drift Index”

---

## Visualisation

Each dot = one URL.

| Visual Element | Meaning |
| --- | --- |
| **Radial distance (farther = less related)** | How far content is from the site’s semantic centre |
| **Colour (SDI)** | How unusual it is versus other pages (semantic + structural drift) |
| **Size** | Click volume (from GSC) |
| **Opacity** | Internal authority (Inlinks) |
| **Rings** | Conceptual zones — Core, Focus, Expansion, Peripheral |

You can interpret the map like this:

- **Big but faint** → high traffic, weak linking
- **Small and far away** → thin or off-topic content 🔎🐄
- **Far + bright** → strong but topical outlier

---

## Sidebar Controls

Defined in `ui/visuals.py` / `ui/layout.py` and wired through `app.py`.

- **Chart Size** — overall square chart size
- **Max Bubble Size (Clicks)** — controls largest point scale
- **Minimum Bubble Opacity** — sets visibility floor
- **Opacity Strength** — adjusts contrast between weakly and strongly linked pages
- **Color Palette for SDI** — choose between *Viridis*, *Blue→Green→Yellow*, or *Red↔Blue*
- **Show Zone Labels** — toggles “Core / Focus / Expansion / Peripheral” markers

Nothing is Streamlit-magic — all parameters feed directly into Altair.

---

## Metrics

### Topical Distance

Numeric distance from the embedding centroid:

```
distance_from_centre = cosine_distance(page_embedding, site_centroid)

```

### Structural Drift Index (SDI)

Relative, site-scoped measure of how “off” a page is:

```
SDI = α * semantic_distance
    + β * (1 - internal_link_score)
    + γ * engagement_offset

```

Where **α > β > γ** (topic weight dominates).

### Opacity Scaling

Uses log(Inlinks) + log(Clicks) → 95th percentile cap → blended → mapped 0–1 → adjusted by `opacity_min`.

So colour = drift, opacity = support, size = performance.

---

## Development Notes

- Python 3.12+ recommended (3.14 had build issues)
- Streamlit + Altair fullscreen needed a CSS patch (in `plot_radial_topical_map`)
- Modular architecture: `core/` = logic, `ui/` = visuals, `utils/` = helpers
- For future tests, create `tests/` at root — don’t mix with `core/`
