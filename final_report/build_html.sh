#!/usr/bin/env bash
# Build the GitHub Pages report from the notebook.
#   1. Execute final_report.ipynb in place (figures and tables are recomputed from committed outputs).
#   2. Export it to docs/index.html with code hidden and the build-check cell removed.
#   3. Set the page title and figure alt text.
# Run from anywhere inside the repo:  bash final_report/build_html.sh
set -euo pipefail
cd "$(dirname "$0")"

jupyter nbconvert --to notebook --execute --inplace final_report.ipynb
jupyter nbconvert --to html --no-input \
  --TagRemovePreprocessor.enabled=True \
  --TagRemovePreprocessor.remove_cell_tags remove_cell \
  --output-dir ../docs --output index final_report.ipynb
python postprocess_html.py ../docs/index.html
touch ../docs/.nojekyll
echo "Done: docs/index.html"
