"""Verify that all entries in combined_output.json exist in BibThesis.bib."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "BibThesis.bib"
JSON_PATH = ROOT / "combined_output.json"

# Load BibTeX keys
bib_text = BIB_PATH.read_text(encoding='utf-8', errors='replace')
bib_keys = set(re.findall(r'@\w+\{([^,}]+)', bib_text))

# Load JSON data
with JSON_PATH.open(encoding='utf-8') as f:
    json_data = json.load(f)

json_keys = {entry['bibtex'] for entry in json_data}

# Check for entries in JSON that don't exist in BIB
missing_in_bib = json_keys - bib_keys

print(f"Entries in BibThesis.bib: {len(bib_keys)}")
print(f"Entries in combined_output.json: {len(json_data)}")
print(f"JSON entries not found in BIB: {len(missing_in_bib)}")

if missing_in_bib:
    print(f"\nMissing entries (first 10):")
    for key in sorted(missing_in_bib)[:10]:
        print(f"  - {key}")
else:
    print("\n[SUCCESS] All entries in combined_output.json exist in BibThesis.bib!")
