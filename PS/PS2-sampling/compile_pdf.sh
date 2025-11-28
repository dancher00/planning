#!/bin/bash

# Compile LaTeX document to PDF
# Usage: ./compile_pdf.sh [yourname]

UNIQUENAME=${1:-"yourname"}

# Copy report.tex to uniquename_ps2.tex
cp report.tex ${UNIQUENAME}_ps2.tex

# Replace yourname in the document
sed -i "s/yourname/${UNIQUENAME}/g" ${UNIQUENAME}_ps2.tex

# Compile LaTeX
pdflatex -interaction=nonstopmode ${UNIQUENAME}_ps2.tex
pdflatex -interaction=nonstopmode ${UNIQUENAME}_ps2.tex  # Run twice for references

# Clean up auxiliary files
rm -f ${UNIQUENAME}_ps2.aux ${UNIQUENAME}_ps2.log ${UNIQUENAME}_ps2.out

echo "PDF generated: ${UNIQUENAME}_ps2.pdf"
echo "Remember to rename the .tex file if needed!"

