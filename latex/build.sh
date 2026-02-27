#!/bin/bash
set -e

# Ensure we are in the script's directory (writing/)
cd "$(dirname "$0")"

# Create output directory
mkdir -p out

# Build sequence
echo "Running XeLaTeX (Pass 1)..."
xelatex -interaction=nonstopmode -synctex=1 -file-line-error -output-directory=out main.tex > /dev/null || true

echo "Running Biber..."
biber --input-directory=out --output-directory=out main > /dev/null || true

echo "Running XeLaTeX (Pass 2)..."
xelatex -interaction=nonstopmode -synctex=1 -file-line-error -output-directory=out main.tex > /dev/null || true

echo "Running XeLaTeX (Pass 3)..."
xelatex -interaction=nonstopmode -synctex=1 -file-line-error -output-directory=out main.tex > /dev/null || true

echo "Build complete. PDF is in writing/out/main.pdf"
