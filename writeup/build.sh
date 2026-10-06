#!/bin/zsh
# Build the write-up: figures from the reports, then LaTeX (BasicTeX is enough). From the repository root:
#   writeup/build.sh   ->  writeup/20260876_MarlenMelis_writeup.pdf
set -e
ROOT=${0:A:h:h}
cd $ROOT && ~/.venvs/rr/bin/python writeup/make_figures.py
cd $ROOT/writeup && for i in 1 2; do pdflatex -interaction=nonstopmode -halt-on-error writeup.tex > build.log; done
cp writeup.pdf 20260876_MarlenMelis_writeup.pdf
pdfinfo 20260876_MarlenMelis_writeup.pdf | grep Pages
