"""
Filter combined_output.json to keep only entries that exist in BibThesis.bib.
BibThesis.bib is the source of truth for which entries should be kept.
"""

import json
import re
from pathlib import Path
from typing import Set, List, Dict

ROOT = Path(__file__).resolve().parent
BIB_PATH = ROOT / "BibThesis.bib"
JSON_PATH = ROOT / "combined_output.json"


def load_bib_keys() -> Set[str]:
    """Load all BibTeX entry keys from BibThesis.bib."""
    bib_text = BIB_PATH.read_text(encoding='utf-8', errors='replace')
    keys = set(re.findall(r'@\w+\{([^,}]+)', bib_text))
    return keys


def load_json_data() -> List[Dict]:
    """Load combined_output.json."""
    with JSON_PATH.open(encoding='utf-8') as f:
        return json.load(f)


def filter_json_entries(json_data: List[Dict], bib_keys: Set[str]) -> tuple[List[Dict], int, int]:
    """
    Filter JSON entries to keep only those with bibtex keys in BibThesis.bib.
    
    Returns:
        filtered_data: List of entries that exist in BibThesis.bib
        original_count: Number of entries before filtering
        removed_count: Number of entries removed
    """
    original_count = len(json_data)
    filtered_data = []
    removed_entries = []
    
    for entry in json_data:
        bibtex_key = entry.get('bibtex', '')
        if bibtex_key in bib_keys:
            filtered_data.append(entry)
        else:
            removed_entries.append(bibtex_key)
    
    removed_count = original_count - len(filtered_data)
    
    if removed_entries:
        print(f"\nRemoved entries (not in BibThesis.bib):")
        for key in sorted(removed_entries)[:20]:  # Show first 20
            print(f"  - {key}")
        if len(removed_entries) > 20:
            print(f"  ... and {len(removed_entries) - 20} more")
    
    return filtered_data, original_count, removed_count


def main():
    """Main function to filter combined_output.json based on BibThesis.bib."""
    print("Loading BibTeX keys from BibThesis.bib...")
    bib_keys = load_bib_keys()
    print(f"Found {len(bib_keys)} entries in BibThesis.bib")
    
    print("\nLoading combined_output.json...")
    json_data = load_json_data()
    print(f"Found {len(json_data)} entries in combined_output.json")
    
    print("\nFiltering entries...")
    filtered_data, original_count, removed_count = filter_json_entries(json_data, bib_keys)
    
    print(f"\nSummary:")
    print(f"  Original entries: {original_count}")
    print(f"  Entries in BibThesis.bib: {len(bib_keys)}")
    print(f"  Entries kept: {len(filtered_data)}")
    print(f"  Entries removed: {removed_count}")
    
    if removed_count > 0:
        # Create backup
        backup_path = JSON_PATH.with_suffix('.json.backup')
        if backup_path.exists():
            print(f"\nRemoving existing backup: {backup_path}")
            backup_path.unlink()
        print(f"Creating backup: {backup_path}")
        JSON_PATH.rename(backup_path)
        
        # Save filtered data
        print(f"Saving filtered data to {JSON_PATH}...")
        with JSON_PATH.open('w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Filtering complete! Backup saved to {backup_path.name}")
    else:
        print("\n[INFO] No entries to remove. All entries in combined_output.json exist in BibThesis.bib.")


if __name__ == "__main__":
    main()
