LATEX = pdflatex
FIGURES_DIR = ../../figures

.PHONY: all clean

all: report.pdf technical_report.pdf

report.pdf: report.tex references.bib $(wildcard $(FIGURES_DIR)/*.pdf)
	$(LATEX) -interaction=nonstopmode report.tex
	bibtex report
	$(LATEX) -interaction=nonstopmode report.tex
	$(LATEX) -interaction=nonstopmode report.tex

technical_report.pdf: technical_report.tex references.bib $(wildcard $(FIGURES_DIR)/*.pdf)
	$(LATEX) -interaction=nonstopmode technical_report.tex
	bibtex technical_report
	$(LATEX) -interaction=nonstopmode technical_report.tex
	$(LATEX) -interaction=nonstopmode technical_report.tex

clean:
	rm -f *.pdf *.aux *.log *.out *.toc *.synctex.gz *.fdb_latexmk *.fls *.bbl *.blg
