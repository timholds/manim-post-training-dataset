#!/usr/bin/env python3
"""
Enhanced data preparation pipeline using plugin-based extractors.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from extractors import get_registry
from extractors.utils import create_conversation, normalize_description, augment_prompt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def deduplicate_data(all_data: List[Dict[str, Any]], source_priorities: Dict[str, int]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Remove duplicate descriptions across datasets.
    Returns deduplicated data and statistics.
    """
    # Group by normalized description
    desc_groups = defaultdict(list)
    
    for item in all_data:
        norm_desc = normalize_description(item['description'])
        desc_groups[norm_desc].append(item)
    
    # Statistics tracking
    stats = {
        'total_raw': len(all_data),
        'unique_descriptions': len(desc_groups),
        'duplicates_removed': 0,
        'duplicates_by_source': defaultdict(int),
        'kept_by_source': defaultdict(int),
        'cross_source_duplicates': 0,
        'within_source_duplicates': 0,
        'examples_removed': []
    }
    
    # Process each group of duplicates
    deduplicated = []
    
    for norm_desc, items in desc_groups.items():
        if len(items) == 1:
            # No duplicates
            deduplicated.append(items[0])
            stats['kept_by_source'][items[0]['source']] += 1
        else:
            # Found duplicates - need to choose best one
            sources = [item['source'] for item in items]
            unique_sources = set(sources)
            
            if len(unique_sources) > 1:
                stats['cross_source_duplicates'] += 1
            else:
                stats['within_source_duplicates'] += 1
            
            # Sort by source priority
            sorted_items = sorted(items, key=lambda x: source_priorities.get(x['source'], 0), reverse=True)
            
            # Keep the best one
            best_item = sorted_items[0]
            deduplicated.append(best_item)
            stats['kept_by_source'][best_item['source']] += 1
            
            # Track what we removed
            for item in sorted_items[1:]:
                stats['duplicates_removed'] += 1
                stats['duplicates_by_source'][item['source']] += 1
                
                # Save some examples (limit to prevent huge files)
                if len(stats['examples_removed']) < 100:
                    stats['examples_removed'].append({
                        'description': item['description'],
                        'source': item['source'],
                        'kept_source': best_item['source'],
                        'normalized_desc': norm_desc
                    })
    
    # Log summary
    logger.info(f"\nDeduplication Summary:")
    logger.info(f"  Original samples: {stats['total_raw']:,}")
    logger.info(f"  Unique samples: {len(deduplicated):,}")
    logger.info(f"  Duplicates removed: {stats['duplicates_removed']:,}")
    logger.info(f"  Cross-source duplicates: {stats['cross_source_duplicates']:,}")
    logger.info(f"  Within-source duplicates: {stats['within_source_duplicates']:,}")
    
    return deduplicated, stats


def split_dataset(data: List[Dict[str, Any]], test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split dataset into train and test sets."""
    random.seed(seed)
    random.shuffle(data)
    
    test_size = int(len(data) * test_ratio)
    test_data = data[:test_size]
    train_data = data[test_size:]
    
    return train_data, test_data


def prepare_datasets(
    output_dir: str,
    sources: Optional[List[str]] = None,
    use_augmentation: bool = False,
    deduplicate: bool = True,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """Prepare datasets using the plugin-based extractor system."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize registry and auto-discover extractors
    registry = get_registry()
    registry.auto_discover()
    
    logger.info(f"Discovered extractors: {registry.list_sources()}")
    
    # Select sources to process
    if sources:
        sources_to_process = [s for s in sources if s in registry.list_sources()]
        if len(sources_to_process) < len(sources):
            missing = set(sources) - set(sources_to_process)
            logger.warning(f"Sources not found: {missing}")
    else:
        sources_to_process = registry.list_sources()
    
    # Collect all data
    all_data = []
    dataset_stats = {}
    source_priorities = {}
    
    for source_id in sources_to_process:
        logger.info(f"\nProcessing source: {source_id}")
        
        try:
            # Get extractor instance
            extractor = registry.get_extractor(source_id)
            source_priorities[source_id] = extractor.priority
            
            # Extract samples
            samples = list(extractor)
            all_data.extend(samples)
            
            dataset_stats[source_id] = {
                "samples": len(samples),
                "expected": extractor.estimate_sample_count(),
                "name": extractor.source_name
            }
            
            logger.info(f"  Extracted {len(samples)} samples")
            
        except Exception as e:
            logger.error(f"Failed to process {source_id}: {e}")
            continue
    
    if not all_data:
        logger.error("No data processed from any source!")
        return
    
    logger.info(f"\nTotal samples collected: {len(all_data)}")
    
    # Apply deduplication if requested
    dedup_stats = None
    if deduplicate:
        all_data, dedup_stats = deduplicate_data(all_data, source_priorities)
        logger.info(f"After deduplication: {len(all_data)} samples")
    
    # Split into train/test
    train_data, test_data = split_dataset(all_data, test_ratio=test_ratio, seed=seed)
    
    # Apply augmentation to training data
    augmented_train = []
    for item in train_data:
        # Always include original
        conv = create_conversation(item["description"], item["code"])
        conv["source"] = item["source"]
        augmented_train.append(conv)
        
        # Add augmented versions
        if use_augmentation:
            # Add 1-2 more variations per sample
            num_augmentations = random.randint(1, 2)
            for i in range(1, num_augmentations + 1):
                augmented_desc = augment_prompt(item["description"], i)
                conv = create_conversation(augmented_desc, item["code"])
                conv["source"] = item["source"]
                augmented_train.append(conv)
    
    # Process test data (no augmentation)
    augmented_test = []
    for item in test_data:
        conv = create_conversation(item["description"], item["code"])
        conv["source"] = item["source"]
        augmented_test.append(conv)
    
    # Save datasets
    train_file = output_path / "train.json"
    test_file = output_path / "test.json"
    stats_file = output_path / "dataset_stats.json"
    
    # Write in JSON lines format
    with open(train_file, 'w') as f:
        for item in augmented_train:
            f.write(json.dumps(item) + '\n')
    
    with open(test_file, 'w') as f:
        for item in augmented_test:
            f.write(json.dumps(item) + '\n')
    
    # Save statistics
    final_stats = {
        "dataset_stats": dataset_stats,
        "total_samples": {
            "raw": dedup_stats['total_raw'] if dedup_stats else len(all_data),
            "after_deduplication": len(all_data) if deduplicate else None,
            "train_before_augmentation": len(train_data),
            "train_after_augmentation": len(augmented_train),
            "test": len(augmented_test),
            "augmentation_factor": len(augmented_train) / len(train_data) if train_data else 0
        },
        "source_distribution": {
            source: len([item for item in augmented_train if item.get("source") == source])
            for source in sources_to_process
        },
        "source_priorities": source_priorities,
        "deduplication_applied": deduplicate,
        "timestamp": datetime.now().isoformat()
    }
    
    with open(stats_file, 'w') as f:
        json.dump(final_stats, f, indent=2)
    
    # Save deduplication report if applicable
    if dedup_stats:
        dedup_report_file = output_path / "deduplication_report.json"
        dedup_report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_raw_samples": dedup_stats['total_raw'],
                "unique_samples": dedup_stats['unique_descriptions'],
                "duplicates_removed": dedup_stats['duplicates_removed'],
                "reduction_percentage": (dedup_stats['duplicates_removed'] / dedup_stats['total_raw'] * 100) if dedup_stats['total_raw'] > 0 else 0,
                "cross_source_duplicates": dedup_stats['cross_source_duplicates'],
                "within_source_duplicates": dedup_stats['within_source_duplicates']
            },
            "duplicates_by_source": dict(dedup_stats['duplicates_by_source']),
            "kept_by_source": dict(dedup_stats['kept_by_source']),
            "source_priority": source_priorities
        }
        
        with open(dedup_report_file, 'w') as f:
            json.dump(dedup_report, f, indent=2)
        
        # Save removed examples
        removed_examples_file = output_path / "removed_duplicates.json"
        with open(removed_examples_file, 'w') as f:
            json.dump(dedup_stats['examples_removed'], f, indent=2)
    
    # Log summary
    logger.info("\n" + "="*60)
    logger.info("DATASET PREPARATION COMPLETE" + (" WITH DEDUPLICATION" if deduplicate else ""))
    logger.info("="*60)
    if dedup_stats:
        logger.info(f"Original samples: {dedup_stats['total_raw']:,}")
        logger.info(f"After deduplication: {len(all_data):,} ({dedup_stats['duplicates_removed']:,} removed, {dedup_stats['duplicates_removed']/dedup_stats['total_raw']*100:.1f}% reduction)")
    logger.info(f"Train samples: {len(augmented_train)} ({len(augmented_train)/len(train_data):.1f}x augmentation)")
    logger.info(f"Test samples: {len(augmented_test)}")
    logger.info(f"\nDataset statistics saved to: {stats_file}")
    if dedup_stats:
        logger.info(f"Deduplication report saved to: {output_path / 'deduplication_report.json'}")
        logger.info(f"Removed examples saved to: {output_path / 'removed_duplicates.json'}")
    logger.info("\nSource distribution:")
    for source, count in final_stats["source_distribution"].items():
        logger.info(f"  {source}: {count} samples")


def main():
    """Main function with CLI support."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare Manim datasets using plugin extractors")
    parser.add_argument("--output-dir", default="data_formatted_v2", help="Output directory")
    parser.add_argument("--sources", nargs="+", help="Specific sources to process (default: all)")
    parser.add_argument("--augmentation", action="store_true", help="Enable data augmentation")
    parser.add_argument("--no-deduplicate", action="store_true", help="Disable deduplication")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--list-sources", action="store_true", help="List available sources and exit")
    
    args = parser.parse_args()
    
    # Initialize registry
    registry = get_registry()
    registry.auto_discover()
    
    if args.list_sources:
        print("\nAvailable data sources:")
        for source_id in sorted(registry.list_sources()):
            extractor = registry.get_extractor(source_id)
            print(f"  {source_id}: {extractor.source_name} (priority={extractor.priority})")
        return
    
    random.seed(args.seed)
    
    prepare_datasets(
        output_dir=args.output_dir,
        sources=args.sources,
        use_augmentation=args.augmentation,
        deduplicate=not args.no_deduplicate,
        test_ratio=args.test_ratio,
        seed=args.seed
    )


if __name__ == "__main__":
    main()