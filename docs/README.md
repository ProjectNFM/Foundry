# Foundry Docs — Analysis & Reports

This directory contains analysis scripts, generated figures, and LaTeX reports
for Foundry experiments.

## Directory Structure

```
docs/
├── README.md                       # this file
├── figures/                        # shared figure output (all scripts write here)
│   ├── data/                       # cached CSV/JSON from W&B
│   ├── *.pdf / *.png               # generated figures
│   └── summary_table.csv           # cross-group summary
└── reports/
    └── tokenizer_journal/          # LaTeX report for tokenizer comparison
        ├── main.tex
        └── Makefile

scripts/analysis/
├── common.py                       # shared W&B utilities, plotting helpers
├── cwt_vs_cnn.py                   # CWT vs CNN figures (behavior + pose + scaling)
├── tokenizer_ablation.py           # AJILE12 ablation figures (behavior + pose)
└── alltoks_sweep.py                # AJILE_NEW_ALLTOKS sweep figures
```

Figures live in `docs/figures/` and are shared across all reports. Each report
uses `\graphicspath` to reference them.

## Prerequisites

### Python

All commands use `uv run` (the project uses [uv](https://docs.astral.sh/uv/) for
dependency management). If you don't have `uv`, install it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then sync the project (from the repo root):

```bash
uv sync --group notebook   # includes matplotlib, seaborn
```

### LaTeX

Install the required TeX Live packages:

```bash
# Ubuntu / Debian
sudo apt-get install texlive-latex-recommended texlive-latex-extra \
                     texlive-extra-utils latexmk
```

Verify:

```bash
pdflatex --version
latexmk --version
```

### LaTeX Workshop in VSCode / Cursor

1. **Install the extension:**

   Open the Extensions panel (`Ctrl+Shift+X`) and search for **LaTeX Workshop**
   by James Yu (`james-yu.latex-workshop`). Install it.

2. **Workspace settings** are already configured in `.vscode/settings.json`:

   - `latex-workshop.latex.rootFile` — points to the tokenizer journal
   - `latex-workshop.formatting.latex` — set to `latexindent`
   - Build recipes for both `pdflatex` (fast) and `latexmk` (robust)
   - PDF viewer opens in an editor tab

3. **Usage:**
   - **Build:** `Ctrl+Alt+B` or just save a `.tex` file (auto-build on save)
   - **View PDF:** `Ctrl+Alt+V` to open the PDF preview in a side tab
   - **SyncTeX:** `Ctrl+Click` in the PDF to jump to the source, and vice versa

## Generating Figures

Each script is independent and only generates figures for its experiment(s).
All scripts default to outputting into `docs/figures/` and caching W&B data
in `docs/figures/data/`.

```bash
# CWT vs CNN comparison (behavior + pose + scaling plot)
uv run scripts/analysis/cwt_vs_cnn.py

# AJILE12 tokenizer ablation (behavior + pose + summary_table.csv)
uv run scripts/analysis/tokenizer_ablation.py

# Extended tokenizer sweep (lower LR, behavior only)
uv run scripts/analysis/alltoks_sweep.py
```

### Common flags

All scripts accept the same flags:

| Flag             | Description                                  |
|------------------|----------------------------------------------|
| `--plot-only`    | Reuse cached CSVs instead of hitting W&B API |
| `--no-curves`    | Skip training curve plots (much faster)      |
| `--output-dir`   | Override figure output directory             |
| `--cache-dir`    | Override W&B data cache directory            |

Example: regenerate just the bar charts from cached data:

```bash
uv run scripts/analysis/cwt_vs_cnn.py --plot-only --no-curves
```

## Building Reports

Each report lives in `docs/reports/<name>/` with its own `main.tex` and
`Makefile`.

```bash
# Build the tokenizer journal PDF
cd docs/reports/tokenizer_journal
make

# Regenerate all figures first, then build
make figures
make
```

## Adding a New Report

1. Create a new directory under `docs/reports/`:

   ```bash
   mkdir -p docs/reports/my_new_report
   ```

2. Copy the template Makefile:

   ```bash
   cp docs/reports/tokenizer_journal/Makefile docs/reports/my_new_report/
   ```

3. Create your `main.tex`. Use `\graphicspath` to reference the shared figures:

   ```latex
   \graphicspath{{../../figures/}}
   ```

4. Update the Makefile `figures` target to call whichever analysis scripts
   produce the figures your report needs.

## Adding a New Figure Script

1. Create a new script in `scripts/analysis/` (e.g., `my_experiment.py`).

2. Import shared utilities from `common.py`:

   ```python
   from common import (
       EXPERIMENT_GROUPS,
       load_or_fetch_group,
       make_base_parser,
       plot_group_bar_chart,
       setup_dirs_and_api,
   )
   ```

3. If your experiment group isn't already defined, add it to `EXPERIMENT_GROUPS`
   in `common.py`.

4. Follow the pattern of the existing scripts — parse args, fetch/load data,
   generate plots.
