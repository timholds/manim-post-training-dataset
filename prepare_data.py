#!/usr/bin/env python3
"""
Enhanced data preparation pipeline using plugin-based extractors.
Includes optional rendering validation to ensure code actually produces videos.
"""

import json
import logging
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm

from extractors import get_registry
from extractors.utils import create_conversation, normalize_description, augment_prompt, normalize_code, calculate_similarity
from extractors.quality_validator import QualityValidator, QualityFilter
from extractors.rendering_validator import RenderingValidator, BatchRenderValidator
from extractors.constants import PLACEHOLDER_DESCRIPTION
from analyze_code_length_distribution import analyze_code_lengths

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fill_placeholder_descriptions(samples: List[Dict[str, Any]], llm_config: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Fill in placeholder descriptions using an LLM.
    This is a placeholder function - implement with your preferred LLM.
    """
    logger.info(f"Checking {len(samples)} samples for placeholder descriptions...")
    
    placeholders_found = 0
    for sample in samples:
        desc = sample.get("description", "")
        # Check for our standardized placeholder first
        if desc.startswith(PLACEHOLDER_DESCRIPTION):
            placeholders_found += 1
            sample["needs_description"] = True
            continue
            
        # Check for other common placeholder patterns
        if (len(desc) < 20 or 
            "[" in desc and "]" in desc or
            "TODO" in desc.upper() or 
            "PLACEHOLDER" in desc.upper() or
            desc.strip() == ""):
            
            placeholders_found += 1
            # TODO: Implement LLM call here
            # For now, just mark it
            sample["needs_description"] = True
    
    logger.info(f"Found {placeholders_found} samples needing description generation")
    return samples


def deduplicate_data(all_data: List[Dict[str, Any]], source_priorities: Dict[str, int]) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Remove only obvious duplicates based on BOTH description and code similarity.
    Keeps items unless we're very sure they're duplicates.
    Returns deduplicated data and statistics.
    """
    # Statistics tracking
    stats = {
        'total_raw': len(all_data),
        'duplicates_removed': 0,
        'duplicates_by_source': defaultdict(int),
        'kept_by_source': defaultdict(int),
        'exact_code_duplicates': 0,
        'high_similarity_duplicates': 0,
        'examples_removed': [],
        'similarity_distribution': defaultdict(int)  # Track similarity scores
    }
    
    # Track which items we've already marked for removal
    to_remove = set()
    
    # Normalize all items once for efficiency
    normalized_items = []
    for i, item in enumerate(all_data):
        normalized_items.append({
            'index': i,
            'item': item,
            'norm_desc': normalize_description(item['description']),
            'norm_code': normalize_code(item['code'])
        })
    
    # Compare each item with all others
    for i in range(len(normalized_items)):
        if i in to_remove:
            continue
            
        for j in range(i + 1, len(normalized_items)):
            if j in to_remove:
                continue
            
            item1 = normalized_items[i]
            item2 = normalized_items[j]
            
            # Calculate similarities
            desc_sim = calculate_similarity(item1['norm_desc'], item2['norm_desc'])
            code_sim = calculate_similarity(item1['norm_code'], item2['norm_code'])
            
            # Track similarity distribution (rounded to nearest 0.1)
            stats['similarity_distribution'][f"{round(code_sim, 1):.1f}"] += 1
            
            # Decision logic: only remove if we're VERY sure it's a duplicate
            is_duplicate = False
            duplicate_reason = ""
            
            # Case 1: Exact code match (code_sim == 1.0)
            if code_sim == 1.0:
                is_duplicate = True
                duplicate_reason = "exact_code_match"
                stats['exact_code_duplicates'] += 1
            
            # Case 2: Both description AND code are extremely similar
            elif desc_sim > 0.95 and code_sim > 0.95:
                is_duplicate = True
                duplicate_reason = f"high_similarity (desc={desc_sim:.2f}, code={code_sim:.2f})"
                stats['high_similarity_duplicates'] += 1
            
            # If it's a duplicate, decide which to keep
            if is_duplicate:
                # Choose which to keep based on priority
                priority1 = source_priorities.get(item1['item']['source'], 0)
                priority2 = source_priorities.get(item2['item']['source'], 0)
                
                if priority1 >= priority2:
                    # Keep item1, remove item2
                    to_remove.add(j)
                    removed_item = item2
                    kept_item = item1
                else:
                    # Keep item2, remove item1
                    to_remove.add(i)
                    removed_item = item1
                    kept_item = item2
                    break  # No need to compare item1 with others if it's being removed
                
                # Track statistics
                stats['duplicates_removed'] += 1
                stats['duplicates_by_source'][removed_item['item']['source']] += 1
                
                # Save example (limit to prevent huge files)
                if len(stats['examples_removed']) < 100:
                    stats['examples_removed'].append({
                        'description': removed_item['item']['description'],
                        'code_preview': removed_item['item']['code'][:200] + '...' if len(removed_item['item']['code']) > 200 else removed_item['item']['code'],
                        'source': removed_item['item']['source'],
                        'kept_source': kept_item['item']['source'],
                        'reason': duplicate_reason,
                        'desc_similarity': round(desc_sim, 3),
                        'code_similarity': round(code_sim, 3)
                    })
    
    # Build final deduplicated list
    deduplicated = []
    for i, norm_item in enumerate(normalized_items):
        if i not in to_remove:
            deduplicated.append(norm_item['item'])
            stats['kept_by_source'][norm_item['item']['source']] += 1
    
    # Log summary
    logger.info(f"\nDeduplication results:")
    logger.info(f"    Original samples: {stats['total_raw']:,}")
    logger.info(f"    Unique samples: {len(deduplicated):,}")
    logger.info(f"    Duplicates removed: {stats['duplicates_removed']:,}")
    logger.info(f"        • Exact code matches: {stats['exact_code_duplicates']:,}")
    logger.info(f"        • High similarity (>95% both): {stats['high_similarity_duplicates']:,}")
    logger.info(f"    Reduction: {stats['duplicates_removed']/stats['total_raw']*100:.1f}%")
    
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
    seed: int = 42,
    quality_config_path: Optional[str] = None,
    no_quality_validation: bool = False,
    quality_strict: bool = False,
    enable_rendering_validation: bool = False,
    rendering_timeout: int = 30,
    rendering_fix_issues: bool = True,
    rendering_dry_run: bool = True,  # Default to True for dry-run
    rendering_fast_mode: bool = False,  # Fast mode for rendering validation
    save_videos_dir: Optional[str] = None,
    llm_fill_descriptions: bool = False,
    llm_config: Optional[Dict] = None
):
    """Prepare datasets using the plugin-based extractor system."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load quality configuration if provided
    quality_config = {}
    if quality_config_path and Path(quality_config_path).exists():
        with open(quality_config_path, 'r') as f:
            quality_config = json.load(f)
        logger.info(f"Loaded quality configuration from {quality_config_path}")
    
    # Apply command-line overrides
    if no_quality_validation:
        quality_config["global_settings"] = quality_config.get("global_settings", {})
        quality_config["global_settings"]["enable_quality_validation"] = False
        logger.info("Quality validation disabled via command line")
    elif quality_strict:
        quality_config["global_settings"] = quality_config.get("global_settings", {})
        quality_config["global_settings"]["enable_quality_validation"] = True
        quality_config["global_settings"]["quality_strict_mode"] = True
        logger.info("Strict quality validation enabled via command line")
    
    # Get already initialized registry (auto_discover was called in main)
    registry = get_registry()
    
    logger.info(f"Discovered {len(registry.list_sources())} extractors: {', '.join(registry.list_sources())}")
    
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
    source_visualization_data = {}  # For visualization
    
    logger.info(f"📥 Extracting from {len(sources_to_process)} sources...")
    
    for source_id in sources_to_process:
        
        try:
            # Get extractor config - pass the FULL quality config
            # The extractor needs access to validation_actions, not just global_settings
            extractor_config = quality_config.get("global_settings", {}).copy()
            
            # Apply source-specific overrides
            if source_id in quality_config.get("source_overrides", {}):
                extractor_config.update(quality_config["source_overrides"][source_id])
            
            # Add reference to full quality config for the validator
            extractor_config["_quality_config"] = quality_config
            
            # Get extractor instance
            extractor = registry.get_extractor(source_id, extractor_config)
            source_priorities[source_id] = extractor.priority
            
            # Extract samples
            samples = list(extractor)
            all_data.extend(samples)
            
            # Collect stats for visualization
            lengths = [len(sample['code']) for sample in samples]
            desc_lengths = [len(sample.get('description', '')) for sample in samples]
            source_visualization_data[source_id] = {
                'lengths': lengths,
                'desc_lengths': desc_lengths,
                'count': len(lengths),
                'name': extractor.source_name,
                'priority': extractor.priority
            }
            
            dataset_stats[source_id] = {
                "samples": len(samples),
                "expected": extractor.estimate_sample_count(),
                "name": extractor.source_name
            }
            
            logger.info(f"  {source_id}: {len(samples):,} samples")
            
        except Exception as e:
            logger.error(f"Failed to process {source_id}: {e}")
            continue
    
    if not all_data:
        logger.error("No data processed from any source!")
        return
    
    logger.info(f"Total: {len(all_data):,} samples collected")
    
    # Generate data sources visualization  
    try:
        viz_path = "data_sources.png"
        analyze_code_lengths(source_visualization_data, viz_path, show_plot=False)
        logger.info(f"📊 Visualization saved: {viz_path}")
    except Exception as e:
        logger.warning(f"Failed to generate visualization: {e}")
    
    # Apply deduplication if requested
    dedup_stats = None
    if deduplicate:
        logger.info(f"\n🔍 Deduplicating...")
        all_data, dedup_stats = deduplicate_data(all_data, source_priorities)
    
    # Fill placeholder descriptions with LLM if requested
    if llm_fill_descriptions:
        all_data = fill_placeholder_descriptions(all_data, llm_config)
    
    # Apply rendering validation if requested
    rendering_stats = None
    if enable_rendering_validation:
        logger.info(f"\n🎬 Validating {len(all_data):,} samples can render...")
        
        # Initialize validators
        render_validator = RenderingValidator(
            timeout=rendering_timeout,
            fix_common_issues=rendering_fix_issues,
            dry_run=rendering_dry_run,
            save_videos_dir=save_videos_dir,
            fast_mode=rendering_fast_mode
        )
        batch_validator = BatchRenderValidator(render_validator, dry_run=rendering_dry_run)
        
        # Use tqdm for progress tracking
        pbar = tqdm(total=len(all_data), desc="Rendering", unit="samples", ncols=80)
        
        def progress_callback(current, total):
            pbar.n = current
            pbar.refresh()
        
        # Validate all samples
        valid_samples, invalid_samples = batch_validator.validate_dataset(
            all_data, 
            progress_callback=progress_callback
        )
        
        pbar.close()
        
        # Update data to only include valid samples
        logger.info(f"Valid: {len(valid_samples):,} ({len(valid_samples)/len(all_data)*100:.0f}%), Invalid: {len(invalid_samples):,}")
        
        # Save rendering validation report
        rendering_report = {
            "timestamp": datetime.now().isoformat(),
            "total_samples": len(all_data),
            "valid_samples": len(valid_samples),
            "invalid_samples": len(invalid_samples),
            "validator_stats": render_validator.stats,
            "failed_examples": batch_validator.get_failed_samples_report()
        }
        
        with open(output_path / "rendering_validation_report.json", 'w') as f:
            json.dump(rendering_report, f, indent=2)
        
        logger.info(render_validator.get_report())
        
        # Update all_data to only valid samples
        all_data = valid_samples
        rendering_stats = rendering_report
    
    # Split into train/test
    logger.info(f"\n✂️ Splitting: ", end="")
    train_data, test_data = split_dataset(all_data, test_ratio=test_ratio, seed=seed)
    logger.info(f"{len(train_data):,} train, {len(test_data):,} test")
    
    # Apply augmentation to training data
    logger.info(f"🔄 Formatting conversations...")
    
    augmented_train = []
    for item in train_data:
        # Always include original
        conv = create_conversation(item["description"], item["code"])
        conv["source"] = item["source"]
        if item.get("rendering_validated"):
            conv["rendering_validated"] = True
        if item.get("auto_fixed"):
            conv["auto_fixed"] = True
            conv["fixes_applied"] = item.get("fixes_applied", [])
        augmented_train.append(conv)
        
        # Add augmented versions
        if use_augmentation:
            # Add 1-2 more variations per sample
            num_augmentations = random.randint(1, 2)
            for i in range(1, num_augmentations + 1):
                augmented_desc = augment_prompt(item["description"], i)
                conv = create_conversation(augmented_desc, item["code"])
                conv["source"] = item["source"]
                if item.get("rendering_validated"):
                    conv["rendering_validated"] = True
                augmented_train.append(conv)
    
    # Process test data (no augmentation)
    augmented_test = []
    for item in test_data:
        conv = create_conversation(item["description"], item["code"])
        conv["source"] = item["source"]
        if item.get("rendering_validated"):
            conv["rendering_validated"] = True
        augmented_test.append(conv)
    
    if use_augmentation:
        logger.info(f"Augmented: {len(train_data):,} → {len(augmented_train):,} train samples ({len(augmented_train)/len(train_data):.1f}x)")
    
    # Save datasets
    logger.info(f"💾 Saving datasets...")
    
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
    
    # Save comprehensive statistics
    final_stats = {
        "dataset_stats": dataset_stats,
        "total_samples": {
            "raw": dedup_stats['total_raw'] if dedup_stats else len(all_data),
            "after_deduplication": len(all_data) if deduplicate else None,
            "after_rendering_validation": len(all_data) if enable_rendering_validation else None,
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
        "rendering_validation_applied": enable_rendering_validation,
        "timestamp": datetime.now().isoformat()
    }
    
    # Add rendering validation summary if applicable
    if rendering_stats:
        final_stats["rendering_validation"] = {
            "total_validated": rendering_stats["total_samples"],
            "passed": rendering_stats["valid_samples"],
            "failed": rendering_stats["invalid_samples"],
            "pass_rate": rendering_stats["valid_samples"] / rendering_stats["total_samples"] * 100 if rendering_stats["total_samples"] > 0 else 0,
            "auto_fixed": rendering_stats["validator_stats"].get("fixed_and_rendered", 0)
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
                "unique_samples": len(all_data),
                "duplicates_removed": dedup_stats['duplicates_removed'],
                "reduction_percentage": (dedup_stats['duplicates_removed'] / dedup_stats['total_raw'] * 100) if dedup_stats['total_raw'] > 0 else 0,
                "exact_code_duplicates": dedup_stats['exact_code_duplicates'],
                "high_similarity_duplicates": dedup_stats['high_similarity_duplicates']
            },
            "duplicates_by_source": dict(dedup_stats['duplicates_by_source']),
            "kept_by_source": dict(dedup_stats['kept_by_source']),
            "source_priority": source_priorities,
            "similarity_distribution": dict(dedup_stats['similarity_distribution'])
        }
        
        with open(dedup_report_file, 'w') as f:
            json.dump(dedup_report, f, indent=2)
        
        # Save removed examples
        removed_examples_file = output_path / "removed_duplicates.json"
        with open(removed_examples_file, 'w') as f:
            json.dump(dedup_stats['examples_removed'], f, indent=2)
    
    # Log summary
    logger.info(f"\n✅ Complete! Final dataset:")
    
    # Pipeline summary
    pipeline_summary = []
    if dedup_stats:
        pipeline_summary.append(f"{dedup_stats['total_raw']:,}")
        if rendering_stats:
            pipeline_summary.append(f"{rendering_stats['valid_samples']:,}")
        else:
            pipeline_summary.append(f"{len(all_data):,}")
    else:
        pipeline_summary.append(f"{len(all_data):,}")
        
    pipeline_summary.append(f"train:{len(augmented_train):,}")
    pipeline_summary.append(f"test:{len(augmented_test):,}")
    
    logger.info(f"  {' → '.join(pipeline_summary)}")
    logger.info(f"  📁 {output_path}")
    
    # Show top sources only
    top_sources = sorted(final_stats["source_distribution"].items(), key=lambda x: x[1], reverse=True)[:5]
    source_summary = ", ".join([f"{src}:{count:,}" for src, count in top_sources])
    if len(final_stats["source_distribution"]) > 5:
        source_summary += f" (+{len(final_stats['source_distribution'])-5} more)"
    logger.info(f"  📋 {source_summary}")


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
    parser.add_argument("--quality-config", default="quality_config.json", help="Quality configuration file")
    parser.add_argument("--no-quality-validation", action="store_true", help="Disable quality validation (overrides config file)")
    parser.add_argument("--quality-strict", action="store_true", help="Enable strict quality validation (overrides config file)")
    
    # New rendering validation options
    parser.add_argument("--enable-rendering", action="store_true", help="Enable rendering validation")
    parser.add_argument("--rendering-timeout", type=int, default=30, help="Timeout for each render attempt (seconds)")
    parser.add_argument("--no-rendering-fixes", action="store_true", help="Disable automatic fixes during rendering")
    parser.add_argument("--rendering-full", type=str, nargs='?', const="rendered_videos", help="Enable full video rendering and save videos to directory (default: rendered_videos)")
    parser.add_argument("--rendering-fast", action="store_true", help="Use fast mode: render only last frame as PNG instead of full video")
    
    # LLM description options
    parser.add_argument("--fill-descriptions", action="store_true", help="Fill placeholder descriptions with LLM")
    parser.add_argument("--llm-config", help="LLM configuration file")
    
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
    
    # Load LLM config if provided
    llm_config = None
    if args.llm_config and Path(args.llm_config).exists():
        with open(args.llm_config, 'r') as f:
            llm_config = json.load(f)
    
    prepare_datasets(
        output_dir=args.output_dir,
        sources=args.sources,
        use_augmentation=args.augmentation,
        deduplicate=not args.no_deduplicate,
        test_ratio=args.test_ratio,
        seed=args.seed,
        quality_config_path=args.quality_config,
        no_quality_validation=args.no_quality_validation,
        quality_strict=args.quality_strict,
        enable_rendering_validation=args.enable_rendering,
        rendering_timeout=args.rendering_timeout,
        rendering_fix_issues=not args.no_rendering_fixes,
        rendering_dry_run=not bool(args.rendering_full),  # Invert the flag
        rendering_fast_mode=args.rendering_fast,
        save_videos_dir=args.rendering_full if args.rendering_full else None,
        llm_fill_descriptions=args.fill_descriptions,
        llm_config=llm_config
    )


if __name__ == "__main__":
    main()