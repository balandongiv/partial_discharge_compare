import json

with open('combined_output.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total entries in combined_output.json: {len(data)}")
print("\n" + "="*60)
print("Sample entries showing discussion content:")
print("="*60)

# Check first 5 entries
for i, entry in enumerate(data[:5]):
    bibtex = entry['bibtex']
    discussion = entry.get('methodology_gap_extractor', {}).get('discussion', {})
    limitations = discussion.get('limitations_and_future_work', {}).get('current_limitations', [])
    future = discussion.get('limitations_and_future_work', {}).get('future_directions', [])
    
    print(f"\n{i+1}. {bibtex}")
    print(f"   Limitations: {limitations[0] if limitations else 'N/A'}")
    print(f"   Future Work: {future[0] if future else 'N/A'}")

# Count entries with placeholder text
placeholder_count = 0
for entry in data:
    limitations = entry.get('methodology_gap_extractor', {}).get('discussion', {}).get('limitations_and_future_work', {}).get('current_limitations', [])
    if limitations and limitations[0] == "Extracted from methodology analysis":
        placeholder_count += 1

print("\n" + "="*60)
print(f"Entries with placeholder text: {placeholder_count}/{len(data)}")
print(f"Entries with actual content: {len(data) - placeholder_count}/{len(data)}")
print("="*60)
