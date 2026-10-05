#!/usr/bin/env bash
set -euo pipefail

# The qualitative maps are raster data embedded losslessly by Matplotlib.
# Re-encode only those rasters as high-quality JPEG while preserving vector text.
quality="${JPEG_QUALITY:-92}"

if [[ "$#" -eq 0 ]]; then
  set -- \
    paper_latex/Images/qualitative_AZ-2_T12SVE.pdf \
    paper_latex/Images/qualitative_MI-1_T15TYM.pdf \
    paper_latex/Images/qualitative_AR-1_T15SYV.pdf \
    paper_latex/Images/qualitative_NH-2_T18TYP.pdf
fi

for input in "$@"; do
  output="${input%.pdf}.compressed.pdf"
  gs \
    -sDEVICE=pdfwrite \
    -dCompatibilityLevel=1.4 \
    -dNOPAUSE -dQUIET -dBATCH \
    -dDetectDuplicateImages=true \
    -dCompressFonts=true -dSubsetFonts=true \
    -dAutoFilterColorImages=false \
    -dColorImageFilter=/DCTEncode \
    -dJPEGQ="$quality" \
    -sOutputFile="$output" \
    "$input"
  mv "$output" "$input"
  printf 'Compressed %s\n' "$input"
done
