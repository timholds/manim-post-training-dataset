#!/usr/bin/env python3
"""Analyze source distribution in the dataset."""

import json
from collections import Counter

def analyze_sources():
    sources_train = Counter()
    sources_test = Counter()
    
    # Analyze training data
    with open("data_formatted_with_sources/train.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            sources_train[data.get('source', 'unknown')] += 1
    
    # Analyze test data
    with open("data_formatted_with_sources/test.json", 'r') as f:
        for line in f:
            data = json.loads(line)
            sources_test[data.get('source', 'unknown')] += 1
    
    print("Source Distribution Analysis")
    print("=" * 50)
    
    print("\nTraining Set:")
    total_train = sum(sources_train.values())
    for source, count in sources_train.most_common():
        percentage = (count / total_train) * 100
        print(f"  {source:20} {count:6,} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':20} {total_train:6,}")
    
    print("\nTest Set:")
    total_test = sum(sources_test.values())
    for source, count in sources_test.most_common():
        percentage = (count / total_test) * 100
        print(f"  {source:20} {count:6,} ({percentage:5.1f}%)")
    print(f"  {'TOTAL':20} {total_test:6,}")
    
    print("\nCombined Total:")
    print(f"  {total_train + total_test:,} samples")

if __name__ == "__main__":
    analyze_sources()