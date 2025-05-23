#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="./4AA_peptide_sequences_sdf_confgen"
OUTPUT="merged.sdf.gz"

# Ensure we have files
shopt -s nullglob
sdfs=("$INPUT_DIR"/*.sdf)
if (( ${#sdfs[@]} == 0 )); then
  echo "Error: no .sdf files found in $INPUT_DIR" >&2
  exit 1
fi

# Process and gzip
(
  for f in "${sdfs[@]}"; do
    name=$(basename "$f" .sdf)
    awk -v n="$name" '
      BEGIN { first=1 }
      /^\$\$\$\$/ {      # record separator
        print;          # emit $$$$
        first=1;        # next line is title
        next
      }
      first==1 {        # this is the title line of the record
        print n;        # replace with filename
        first=0;
        next
      }
      { print }         # all other lines unchanged
    ' "$f"
  done
) | gzip > "$OUTPUT"

echo "✔ Merged ${#sdfs[@]} files into $OUTPUT"
