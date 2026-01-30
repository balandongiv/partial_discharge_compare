"""
Remove entries from 2019 and older from BibThesis.bib and combined_output.json.
Keep only entries from 2020 onwards.
"""

import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "BibThesis.bib"
JSON_PATH = ROOT / "combined_output.json"


def extract_bib_entries(bib_text: str) -> list[tuple[str, int, int, int]]:
    """
    Extract BibTeX entries with their keys, years, and positions.
    Returns list of (key, year, start_pos, end_pos).
    """
    entries = []
    # Pattern to match @entrytype{key, ... year = {YYYY} ... }
    pattern = r'@\w+\{([^,}]+)'
    for match in re.finditer(pattern, bib_text):
        key = match.group(1)
        start_pos = match.start()
        # Find the closing brace of this entry
        brace_count = 0
        in_entry = False
        end_pos = start_pos
        year = None
        
        for i in range(start_pos, len(bib_text)):
            if bib_text[i] == '@' and not in_entry:
                in_entry = True
                brace_count = 0
            elif bib_text[i] == '{':
                brace_count += 1
            elif bib_text[i] == '}':
                brace_count -= 1
                if brace_count == 0 and in_entry:
                    end_pos = i + 1
                    # Extract year from this entry block
                    entry_text = bib_text[start_pos:end_pos]
                    year_match = re.search(r'year\s*=\s*\{?\s*(\d{4})', entry_text)
                    if year_match:
                        year = int(year_match.group(1))
                    break
        
        if year is not None:
            entries.append((key, year, start_pos, end_pos))
    
    return entries


def remove_old_bib_entries(bib_text: str) -> tuple[str, int]:
    """
    Remove entries with year <= 2019 from BibTeX file.
    Returns (cleaned_text, removed_count).
    """
    entries = extract_bib_entries(bib_text)
    old_entries = [(k, y, s, e) for k, y, s, e in entries if y <= 2019]
    
    if not old_entries:
        return bib_text, 0
    
    # Sort by start position in reverse to remove from end to start
    old_entries.sort(key=lambda x: x[2], reverse=True)
    
    # Remove entries from end to start to preserve positions
    cleaned = bib_text
    for key, year, start_pos, end_pos in old_entries:
        # Also remove trailing newlines if present
        while end_pos < len(cleaned) and cleaned[end_pos] in '\n\r':
            end_pos += 1
        cleaned = cleaned[:start_pos] + cleaned[end_pos:]
    
    # Clean up multiple consecutive blank lines
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned, len(old_entries)


def remove_old_json_entries(json_data: list) -> tuple[list, int]:
    """
    Remove entries from JSON where bibtex key contains year <= 2019.
    Returns (cleaned_data, removed_count).
    """
    old_count = 0
    cleaned = []
    
    for entry in json_data:
        bibtex_key = entry.get('bibtex', '')
        # Extract year from key pattern like "Author_2019" or "Author_2019_1"
        year_match = re.search(r'_(\d{4})(?:_|$)', bibtex_key)
        if year_match:
            year = int(year_match.group(1))
            if year <= 2019:
                old_count += 1
                continue
        cleaned.append(entry)
    
    return cleaned, old_count


def main():
    print("Processing BibThesis.bib...")
    bib_text = BIB_PATH.read_text(encoding='utf-8', errors='replace')
    cleaned_bib, removed_bib = remove_old_bib_entries(bib_text)
    
    # Backup original
    backup_bib = BIB_PATH.with_suffix('.bib.backup')
    if not backup_bib.exists():
        backup_bib.write_text(bib_text, encoding='utf-8')
        print(f"  Created backup: {backup_bib.name}")
    
    BIB_PATH.write_text(cleaned_bib, encoding='utf-8')
    print(f"  Removed {removed_bib} entries from BibThesis.bib")
    
    print("\nProcessing combined_output.json...")
    json_data = json.load(JSON_PATH.open(encoding='utf-8'))
    cleaned_json, removed_json = remove_old_json_entries(json_data)
    
    # Backup original
    backup_json = JSON_PATH.with_suffix('.json.backup')
    if not backup_json.exists():
        JSON_PATH.rename(backup_json)
        print(f"  Created backup: {backup_json.name}")
    
    JSON_PATH.write_text(
        json.dumps(cleaned_json, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    print(f"  Removed {removed_json} entries from combined_output.json")
    
    print(f"\nSummary:")
    print(f"  BibThesis.bib: {removed_bib} entries removed (kept entries from 2020+)")
    print(f"  combined_output.json: {removed_json} entries removed (kept entries from 2020+)")


if __name__ == '__main__':
    main()
