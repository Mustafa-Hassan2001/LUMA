# Report

`main.tex` is a paper-style report (two columns, ~6–8 pages once results are in).

**No result number is typed into `main.tex`.** It reads every number from
`generated/numbers.tex` (LaTeX macros) and every table from `generated/*.tex`.
Both are written by `src/collect_results.py` from the JSON files in `results/`.
Until a result exists, the PDF shows a red **[pending]**. Red **[TODO]** notes mark
the few sentences that interpret results, which can only be written once the
numbers are known.

## Build

**Overleaf (easiest):** upload `main.tex`, `references.bib`, the `generated/`
folder and `../results/figures/` (keep the relative paths), then compile with pdfLaTeX.

**Locally / in Colab:**
```bash
python src/collect_results.py --results results      # refresh numbers + tables
cd report && latexmk -pdf main.tex                     # needs TeX Live (texlive-latex-extra, texlive-pictures)
```
