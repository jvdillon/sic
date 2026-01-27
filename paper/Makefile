.PHONY: all icml arxiv clean clean-icml clean-arxiv

export TEXINPUTS := ./styles/:$(TEXINPUTS)

CONTENT = 0_abstract.tex 1_introduction.tex 2_background.tex 3_method.tex 4_related.tex 5_experiments.tex 6_conclusion.tex 7_appendix.tex
FIGURES = figures/diagram.tex
BIBFILE = references.bib

all: icml arxiv

icml: icml.pdf

arxiv: arxiv.pdf

icml.pdf: icml.tex $(CONTENT) $(BIBFILE) $(FIGURES)
	pdflatex icml
	-bibtex icml
	pdflatex icml
	pdflatex icml

arxiv.pdf: arxiv.tex $(CONTENT) $(BIBFILE) $(FIGURES)
	pdflatex arxiv
	-bibtex arxiv
	pdflatex arxiv
	pdflatex arxiv

clean: clean-icml clean-arxiv

clean-icml:
	rm -f icml.aux icml.bbl icml.blg icml.out icml.log icml.pdf

clean-arxiv:
	rm -f arxiv.aux arxiv.bbl arxiv.blg arxiv.out arxiv.log arxiv.pdf

figures/diagram.pdf: figures/diagram.tex
	cd figures && pdflatex diagram.tex && cd ..
