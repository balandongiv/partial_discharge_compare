import re
import os

input_file = 'raw_refs.txt'
output_file = 'test.bib'

def get_publisher(journal, doi):
    journal_lower = journal.lower()
    doi_lower = doi.lower()
    
    if 'ieee' in journal_lower or 'ieee' in doi_lower:
        return 'ieee'
    if 'sciencedirect' in doi_lower or '10.1016' in doi_lower or 'elsevier' in journal_lower:
        return 'sciencedirect'
    if 'nature' in journal_lower or 'scientific reports' in journal_lower or '10.1038' in doi_lower:
        return 'nature'
    if 'springer' in journal_lower or '10.1007' in doi_lower:
        return 'springer'
    if 'wiley' in journal_lower or '10.1002' in doi_lower or '10.1155' in doi_lower: # Hindawi is Wiley now? No, but common.
        return 'wiley'
    if 'mdpi' in doi_lower or '3390' in doi_lower or journal_lower in ['energies', 'sensors', 'applied sciences', 'processes', 'electronics']:
        return 'mdpi'
    if 'iop' in journal_lower or '10.1088' in doi_lower:
        return 'iopscience'
    
    return 'other'

def clean_authors(author_str):
    # Remove leading/trailing formatting
    # "Abdullah, A. Z. B., Isa, M., ..."
    # Replace "&" with "and"
    authors = author_str.replace('&', ',') # Bibtex uses 'and' but we need to split names first usually?
    # Actually bibtex author field: "Surname, Firstname and Surname, Firstname"
    # The input is "Surname, Initials., Surname, Initials., & Surname, Initials."
    # So split by ".," or something?
    # Easier: Replace "&" with "," then simply use the string. Bibtex handles "And" better if explicit.
    # But usually "A, B, C" is treated as one name "A, B, C" if not careful.
    # Standard format: "Author1 and Author2 and Author3"
    
    # Input: "Abdullah, A. Z. B., Isa, M., ... "
    # We need to recognize Name boundaries.
    # Heuristic: Split by `.,` or `.` at end of initials. 
    # But "A. Z. B." -> multiple dots.
    # Split by `.,` ? "B., Isa" -> Yes.
    
    # Replace "&" with ","
    a = author_str.replace('&', ',')
    # Split by separator "., "
    # Note: Regex `\.,\s+`
    parts = re.split(r'\.,\s+', a)
    
    # Reassemble with " and "
    # Last part might not have `.,` if it's just "Name."
    cleaned_parts = []
    for p in parts:
        p = p.strip()
        if p.endswith('.'):
             cleaned_parts.append(p)
        else:
             cleaned_parts.append(p + '.') # Ensure dot at end of initials if missing
             
    # One tricky case: "Name, A." -> "Name, A."
    # The last author might look like "Azizan, N."
    
    return ' and '.join(cleaned_parts)

def parse_volume_pages(journal_info):
    # Expected: "Journal Name, 10(4), 2190–2197"
    # Or "Journal Name, 10, 95333–95344"
    # Or "Journal Name, 172" (Volume only)
    
    journal = journal_info
    volume = ""
    pages = ""
    
    # Look for the last part which is pages
    # Regex for pages: `\d+[–-]\d+`
    page_match = re.search(r',\s*(\d+[–-]\d+)\.?$', journal_info)
    if page_match:
        pages = page_match.group(1).replace('–', '-')
        journal_info = journal_info[:page_match.start()]
    
    # Look for volume/issue: `, \d+(\(\d+\))?` at end
    vol_match = re.search(r',\s*(\d+(?:\(\d+\))?)\.?$', journal_info)
    if vol_match:
        volume = vol_match.group(1)
        journal_info = journal_info[:vol_match.start()]
    else:
        # Sometimes just volume number if no pages?
        # Check if the end is just digits
        vol_only = re.search(r',\s*(\d+)\.?$', journal_info)
        if vol_only:
             volume = vol_only.group(1)
             journal_info = journal_info[:vol_only.start()]

    journal = journal_info.strip(',. ')
    return journal, volume, pages

def load_existing_keys(bib_file):
    keys = {}
    if not os.path.exists(bib_file):
        return keys
        
    with open(bib_file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Find all keys: @article{Key,
    matches = re.findall(r'@\w+\s*{\s*([^,]+),', content)
    for k in matches:
        k = k.strip()
        # Parse Key format: Surname_Year or Surname_Year_Index
        # Regex: ^([a-zA-Z-]+_\d{4})(_\d+)?$
        m = re.match(r'^([a-zA-Z.\-]+_\d{4})(_(\d+))?$', k)
        if m:
            base = m.group(1)
            suffix = m.group(3)
            if base not in keys:
                keys[base] = 0
            
            if suffix:
                idx = int(suffix)
                if idx > keys[base]:
                    keys[base] = idx
            else:
                # If we have base only, it counts as 0? 
                # Or typically if we have duplicates, the first one has no suffix, second has _1?
                # Let's assume keys[base] tracks the HIGHEST suffix used.
                # If "Smith_2020" exists, next is "Smith_2020_1".
                # If "Smith_2020_1" exists, next is "Smith_2020_2".
                pass
    return keys

def main():
    # Load existing keys from BibThesis.bib to avoid collisions
    existing_keys = load_existing_keys('BibThesis.bib')
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Step 1: Clean and Merge
    clean_text = ""
    for line in lines:
        l = line.strip()
        if re.match(r'^\d+$', l): # Skip page numbers
            continue
        if not l: continue
        clean_text += l + " "
        
    # Step 2: Fix hyphenation
    clean_text = re.sub(r'-\s+', '-', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    
    # Step 3: Find Entries
    # Regex for "Author (Year)."
    # refined to include accents (by accepting non-digits/non-parens)
    # Author part: Starts with Uppercase. NO DIGITS.
    matches = list(re.finditer(r'([A-Z][^0-9\(\)]*?)\s*\((\d{4})\)\.\s*', clean_text))
    
    print(f"Found {len(matches)} entries.")
    
    bib_entries = []

    for i in range(len(matches)):
        m = matches[i]
        authors_raw = m.group(1).strip()
        year = m.group(2)
        
        # Cleanup leading garbage in authors
        # Patterns like "TDEI.2023.3305820 Gillis" -> "Gillis"
        # "ACCESS.2024.3350555 Alshalawi" -> "Alshalawi"
        # Regex: Remove leading [A-Z]+[\.\s]*\d+[\.\s]*
        authors_raw = re.sub(r'^[A-Z]+\.?\s*\d+\.?\d*\s*', '', authors_raw)
        
        start_content = m.end()
        if i < len(matches) - 1:
            end_content = matches[i+1].start()
        else:
            end_content = len(clean_text)
            
        content_raw = clean_text[start_content:end_content].strip()
        
        # Extract DOI - Aggressive detection for broken DOIs
        # Search for "10.xxxx/..." or "https...10.xxxx"
        # Handle "https : // doi . org" spaces
        doi = ""
        # First, try to fix broken spacing in potential DOI patterns
        # Identify region starting with http or 10.
        # We replace "https : //" with "https://" locally for extraction?
        # Only affect content_raw's DOI part.
        
        # Regex for fuzzy DOI:
        # Match "10." followed by digits, then slash, then chars.
        # Also handles "https : // doi . org / 10 ."
        
        # Simplification: Look for "10.<4 digits>/" with optional spaces
        doi_match = re.search(r'(https?\s*:\s*/\s*/\s*doi\s*\.?\s*org\s*/\s*)?(10\.\d{4,}\s*/\s*[^\s]+)', content_raw)
        if doi_match:
            full_match = doi_match.group(0)
            doi_val = doi_match.group(2)
            # Remove from content
            content_raw = content_raw.replace(full_match, "").strip()
            # Clean DOI value
            doi = doi_val.replace(" ", "")
            doi = doi.rstrip('.')
        
        # Fallback: Check for just "https://doi.org/..." if not caught
        if not doi:
             doi_match_Simple = re.search(r'(https?://doi\.org/\S+)', content_raw)
             if doi_match_Simple:
                 doi = doi_match_Simple.group(1).replace('https://doi.org/', '').replace('http://doi.org/', '')
                 content_raw = content_raw.replace(doi_match_Simple.group(1), "").strip()

        # Extract Title and Journal Info
        parts = re.split(r'\.\s+', content_raw, maxsplit=1)
        if len(parts) == 2:
            title = parts[0]
            journal_info = parts[1]
        else:
            title = content_raw
            journal_info = ""
            
        journal, volume, pages = parse_volume_pages(journal_info)
        authors_bib = clean_authors(authors_raw)
        publisher = get_publisher(journal, doi)
        
        # Generate Key
        # First author surname
        first_author = authors_raw.split(',')[0].strip()
        # Clean up surname "Abdullah" from "Abdullah, A. Z." -> "Abdullah"
        # Since split by comma took the whole first part.
        
        key_base = f"{first_author}_{year}"
        
        # Determine suffix
        if key_base in existing_keys:
            existing_keys[key_base] += 1
            idx = existing_keys[key_base]
            key = f"{key_base}_{idx}"
        else:
            # First time seeing this base?
            # Check if we should allow "Surname_Year" without suffix.
            # If the user's previous bib assumes uniqueness, we can start with no suffix.
            # But if we want safety, we assume 0 used.
            existing_keys[key_base] = 0 # Mark as used
            key = key_base
            # If we encounter this again in THIS loop, it will increment
        
        entry = f"@article{{{key},\n"
        entry += f"  title = {{{title}}},\n"
        entry += f"  author = {{{authors_bib}}},\n"
        entry += f"  year = {{{year}}},\n"
        entry += f"  journal = {{{journal}}},\n"
        if volume:
            entry += f"  volume = {{{volume}}},\n"
        if doi:
            entry += f"  doi = {{{doi}}},\n"
        entry += f"  publisher = {{{publisher}}}\n"
        entry += "}\n"
        
        bib_entries.append(entry)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(bib_entries))
    
    print(f"Written {len(bib_entries)} entries to {output_file}")

if __name__ == "__main__":
    main()
