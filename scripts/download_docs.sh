#!/usr/bin/env bash
# Download the LSEG filings used in the evaluation into data/raw/.
set -euo pipefail

BASE="https://www.lseg.com/content/dam/lseg/en_us/documents"
IR="$BASE/investor-relations"
OUT="$(dirname "$0")/../data/raw"
mkdir -p "$OUT"

urls=(
  "$IR/annual-reports/lseg-annual-report-2025.pdf"
  "$IR/annual-reports/lseg-annual-report-2020.pdf"
  "$IR/annual-reports/lseg-annual-report-2019.pdf"
  "$IR/annual-reports/lseg-annual-report-2018.pdf"
  "$IR/financial-results/interim-report/lseg-interim-report-h1-2026-30july2026.pdf"
  "$IR/financial-results/interim-results/presentation/lseg-interim-results-h1-2026-presentation-30july2026.pdf"
  "$IR/financial-results/interim-results/transcripts/lseg-h1-2026-interim-results-analyst-investor-call-transcript.pdf"
  "$IR/financial-results/preliminary-results/presentation/lseg-2025-preliminary-results-presentation-26feb2026.pdf"
  "$IR/financial-results/preliminary-results/rns/lseg-2025-preliminary-results-rns-26feb2026.pdf"
  "$IR/financial-results/preliminary-results/transcripts/lseg-2025-preliminary-results-transcript-26feb2026.pdf"
  "$BASE/reports/lseg-sustainability-report.pdf"
)

for u in "${urls[@]}"; do
  curl -fsSL -A "Mozilla/5.0" -o "$OUT/$(basename "$u")" "$u" && echo "ok  $(basename "$u")"
done
