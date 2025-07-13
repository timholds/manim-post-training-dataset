import json

# Load the two dataset stats
with open('data_test_v2/dataset_stats.json', 'r') as f:
    old_stats = json.load(f)
    
with open('data_formatted_v2/dataset_stats.json', 'r') as f:
    new_stats = json.load(f)

print("=== RAW SAMPLE COUNTS ===")
print("\nOld run:")
old_sources = old_stats['dataset_stats']
for source, info in sorted(old_sources.items()):
    print(f"  {source}: {info['samples']} samples")
print(f"  TOTAL RAW: {old_stats['total_samples']['raw']}")

print("\nNew run:")
new_sources = new_stats['dataset_stats']
for source, info in sorted(new_sources.items()):
    print(f"  {source}: {info['samples']} samples")
print(f"  TOTAL RAW: {new_stats['total_samples']['raw']}")

print("\n=== COMPARISON OF SHARED SOURCES ===")
for source in sorted(old_sources.keys()):
    if source in new_sources:
        old_count = old_sources[source]['samples']
        new_count = new_sources[source]['samples']
        diff = new_count - old_count
        if diff != 0:
            print(f"{source}: {old_count} → {new_count} ({diff:+d})")
        else:
            print(f"{source}: {old_count} (unchanged)")

print("\n=== AFTER DEDUPLICATION ===")
print(f"Old: {old_stats['total_samples']['after_deduplication']} unique")
print(f"New: {new_stats['total_samples']['after_deduplication']} unique")
print(f"Difference: {new_stats['total_samples']['after_deduplication'] - old_stats['total_samples']['after_deduplication']:+d}")

# Check kept_by_source
with open('data_formatted_v2/deduplication_report.json', 'r') as f:
    dedup = json.load(f)
    
print("\n=== SAMPLES KEPT BY SOURCE (after dedup) ===")
for source, count in sorted(dedup['kept_by_source'].items()):
    old_dist = old_stats['source_distribution'].get(source, 0)
    if source in old_stats['source_distribution']:
        diff = count - old_dist
        print(f"{source}: {old_dist} → {count} ({diff:+d})")
    else:
        print(f"{source}: NEW → {count}")