import json

with open('data_formatted_v2/removed_duplicates.json', 'r') as f:
    dups = json.load(f)

# Find where vivek3141 lost to a different source
cross_source = []
vivek_to_vivek = []
other_cross = []

for d in dups:
    if d['source'] == 'vivek3141' and d['kept_source'] != 'vivek3141':
        cross_source.append(d)
    elif d['source'] == 'vivek3141' and d['kept_source'] == 'vivek3141':
        vivek_to_vivek.append(d)
    elif d['source'] != d['kept_source']:
        other_cross.append(d)

print(f'Vivek3141 samples removed in favor of OTHER sources: {len(cross_source)}')
print(f'Vivek3141 within-source duplicates: {len(vivek_to_vivek)}')
print(f'Other cross-source duplicates: {len(other_cross)}')

if cross_source:
    print("\n=== Vivek3141 lost to other sources ===")
    for d in cross_source[:10]:
        print(f"\nDescription: {d['description']}")
        print(f"Lost to: {d['kept_source']}")

# Also check if any existing sources lost samples
sources_that_lost = {}
for d in dups:
    if d['source'] != d['kept_source']:
        if d['source'] not in sources_that_lost:
            sources_that_lost[d['source']] = []
        sources_that_lost[d['source']].append(d['kept_source'])

print("\n=== Sources that lost samples to other sources ===")
for source, kept_sources in sources_that_lost.items():
    print(f"{source}: lost {len(kept_sources)} samples to {set(kept_sources)}")