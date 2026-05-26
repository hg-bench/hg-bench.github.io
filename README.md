# HG-Bench Project Page

Self-contained static project page (one HTML file, no build step). Designed as a normal preprint companion page, not labeled as an EMNLP submission.

## ⚠ Before pushing publicly — fill in 5 placeholders

Open `index.html` and search for each of these, replace with your real info:

| Placeholder | Where it appears |
|---|---|
| `Author One`, `Author Two`, `Author Three` | hero authors line + BibTeX |
| `Affiliation One`, `Affiliation Two` | hero affiliations line |
| `https://arxiv.org/abs/2603.XXXXX` | arXiv button + BibTeX `journal` field |
| `https://github.com/USER/hg-bench` | Code button + footer |
| `https://huggingface.co/datasets/USER/HG-Bench` | Dataset button |
| `https://huggingface.co/USER/GLM-4.6V-9B-HG-SFT` | Checkpoint button |

You can mass-replace these in any editor (VS Code Find/Replace, sed, etc.).

## What's here

- `index.html` — the entire page (one file, ~700 lines). No build step.
- `README.md` — this file.

## What to drop in this folder before deploying

Put these files **next to `index.html`** (same folder, no subfolders needed):

| Filename | Source | Purpose |
|---|---|---|
| `paper.pdf` | your preprint / camera-ready PDF | "Paper (PDF)" button |
| `emnlp_case.png` | export of `emnlp_case.pdf` | Figure 1 (hero teaser) |
| `figure2.png` | data-engine pipeline figure | §Dataset · pipeline |
| `figure3.png` | composition / distribution figure | §Dataset · composition |
| `emnlp_case_hardest.png` | `mmdoctor/results/homework_qual_cases/` | §Cases · Case 1 |
| `emnlp_case_multipage.png` | same folder | §Cases · Case 2 |
| `emnlp_case_3_over.png` | same folder | §Cases · Case 3 |
| `emnlp_case_3_under.png` | same folder | §Cases · Case 4 |

If a file is missing the page **degrades gracefully** — instead of a broken image, it shows a grey placeholder telling you which filename to add.

### Converting `emnlp_case.pdf` to PNG

The hero teaser uses `emnlp_case.png` (PNG, not PDF — browsers don't render PDF inline reliably). To convert:

```bash
# Linux/macOS with poppler installed
pdftoppm -r 200 -png emnlp_case.pdf emnlp_case      # produces emnlp_case-1.png
mv emnlp_case-1.png emnlp_case.png

# or ImageMagick
convert -density 200 emnlp_case.pdf emnlp_case.png
```

200 DPI is a good balance of quality and file size.

## Deploying to GitHub Pages (recommended)

Choose ONE of two URL styles:

### Option A: User/Org page → `https://USERNAME.github.io/`

Cleanest URL, no path suffix.

1. Create a GitHub repo **named exactly `USERNAME.github.io`** (must equal your account/org name + `.github.io`).
2. In this folder, run:
   ```bash
   git init -b main
   git add .
   git commit -m "Initial project page"
   git remote add origin https://github.com/USERNAME/USERNAME.github.io.git
   git push -u origin main
   ```
3. Settings → Pages → "Your site is live at https://USERNAME.github.io/" appears in 1–2 minutes.

### Option B: Project page → `https://USERNAME.github.io/hg-bench/`

URL has the repo name suffix; no special repo naming required.

1. Create any repo, e.g. `hg-bench`.
2. Same push commands as above (different remote URL).
3. Settings → Pages → set Source = `main` branch, root folder. Live URL appears.

## Other hosts (drop-in)

- **Netlify**: drag this folder onto netlify.com/drop.
- **Cloudflare Pages**: connect repo, output dir = `/`.
- **Vercel**: `vercel --prod` inside this folder.
- **Local preview**: `python3 -m http.server 8000`, then open http://localhost:8000.

## Page sections (anchor IDs)

- `#abstract` — Abstract + 4 highlight cards
- `#dataset` — Source pool, annotation, pipeline figure, stats, composition figure
- `#task` — JSON output schema with example
- `#protocol` — $\mathcal{F}_A$ and $\mathcal{F}_S^{\mu}$ formulas, 4 supplementary metrics
- `#results` — Full Table 1 (colored, same design as paper Table 2) + 3 key-insight callouts + inter-model agreement
- `#cases` — 4 qualitative cases (image grid)
- `#system` — Reference SFT system details + hyperparameter table
- `#download` — Artifact release list + full prompt template
- `#cite` — BibTeX with copy button

## Stack

- Pure HTML + CSS, no build step, no Node/npm.
- CDN dependencies: Google Fonts, Font Awesome 6, MathJax 3 (for `$\mathcal{F}_A$` rendering).
- ~38 KB HTML + ~150 KB of CDN assets on first load.
