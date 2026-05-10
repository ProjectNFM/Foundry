TEX = main
LATEX = pdflatex
FIGURES_DIR = ../../figures
REPO_ROOT = ../../..

.PHONY: all clean figures figures-cwt figures-ablation figures-alltoks figures-appendix figures-experiment figures-cwt-params figures-token-rate-sweep

all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex $(wildcard $(FIGURES_DIR)/*.pdf)
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(LATEX) -interaction=nonstopmode $(TEX).tex
	$(LATEX) -interaction=nonstopmode $(TEX).tex

figures: figures-cwt figures-ablation figures-alltoks figures-appendix figures-experiment figures-cwt-params figures-token-rate-sweep

figures-cwt:
	cd $(REPO_ROOT) && uv run scripts/analysis/cwt_vs_cnn.py

figures-ablation:
	cd $(REPO_ROOT) && uv run scripts/analysis/tokenizer_ablation.py

figures-alltoks:
	cd $(REPO_ROOT) && uv run scripts/analysis/alltoks_sweep.py

figures-appendix:
	cd $(REPO_ROOT) && uv run scripts/analysis/appendix_cwt_figures.py
	cd $(REPO_ROOT) && uv run scripts/analysis/appendix_cnn_figures.py

figures-experiment:
	cd $(REPO_ROOT) && uv run scripts/analysis/experiment_figures.py

figures-cwt-params:
	cd $(REPO_ROOT) && uv run scripts/analysis/cwt_learned_params.py

figures-token-rate-sweep:
	cd $(REPO_ROOT) && uv run scripts/analysis/token_rate_sweep.py

clean:
	rm -f $(TEX).pdf $(TEX).aux $(TEX).log $(TEX).out $(TEX).toc \
	      $(TEX).bbl $(TEX).bcf $(TEX).blg $(TEX).run.xml \
	      $(TEX).synctex.gz $(TEX).fdb_latexmk $(TEX).fls
