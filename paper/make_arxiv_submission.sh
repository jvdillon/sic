#!/bin/bash
# Create arXiv submission bundle

rm -rf arxiv_submission
mkdir arxiv_submission

# Copy main files
cp arxiv.tex arxiv_submission/
cp preamble.tex arxiv_submission/
cp references.bib arxiv_submission/
cp [0-9]_*.tex arxiv_submission/

# Flatten styles (arXiv doesn't like subdirs)
cp styles/*.sty arxiv_submission/

# Copy figures (preserve subdirectory - arXiv allows one level)
mkdir -p arxiv_submission/figures
cp figures/*.pdf figures/*.png arxiv_submission/figures/ 2>/dev/null || true
cp figures/*.tex arxiv_submission/figures/ 2>/dev/null || true

# Generate .bbl (required - arXiv won't run bibtex)
cd arxiv_submission
pdflatex arxiv
bibtex arxiv
cd ..

# Remove TEXINPUTS line from preamble if present (not needed when flat)
# Arxiv will find .sty files in same directory

echo "Created arxiv_submission/"
echo "Contents:"
ls -la arxiv_submission/

# Create tarball for upload
tar -czvf arxiv_submission.tar.gz -C arxiv_submission .
echo ""
echo "Upload this file to arXiv: arxiv_submission.tar.gz"
ls -lh arxiv_submission.tar.gz
