#!/usr/bin/env python3
"""
Comprehensive code length distribution analysis combining all sources.
Merges visualization types from previous analyses.
"""

import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import seaborn as sns
from extractors import get_registry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_code_lengths(source_data=None, output_path='data_sources.png', show_plot=True):
    """
    Analyze code length distribution across all sources.
    
    Args:
        source_data: Optional pre-collected source data dict. If None, will collect from registry.
        output_path: Path to save the visualization
        show_plot: Whether to display the plot interactively
    """
    
    # If no source data provided, collect it
    if source_data is None:
        # Initialize registry
        registry = get_registry()
        registry.auto_discover()
        
        all_sources = registry.list_sources()
        logger.info(f"Found {len(all_sources)} sources")
        
        # Collect data
        source_data = {}
        all_lengths = []
        
        for source_id in all_sources:
            logger.info(f"Processing {source_id}...")
            try:
                extractor = registry.get_extractor(source_id)
                samples = list(extractor)
                
                lengths = [len(sample['code']) for sample in samples]
                desc_lengths = [len(sample.get('description', '')) for sample in samples]
                source_data[source_id] = {
                    'lengths': lengths,
                    'desc_lengths': desc_lengths,
                    'count': len(lengths),
                    'name': extractor.source_name,
                    'priority': extractor.priority
                }
                all_lengths.extend(lengths)
                
            except Exception as e:
                logger.error(f"Failed to process {source_id}: {e}")
                continue
    else:
        # Extract all lengths from provided data
        all_lengths = []
        for data in source_data.values():
            all_lengths.extend(data['lengths'])
            # Ensure desc_lengths exists for provided data
            if 'desc_lengths' not in data:
                data['desc_lengths'] = [0] * len(data['lengths'])  # Placeholder
    
    # Calculate overall statistics
    all_lengths = np.array(all_lengths)
    overall_median = np.median(all_lengths)
    overall_mean = np.mean(all_lengths)
    
    # Define consistent color scheme for sources
    # Create a color palette that assigns consistent colors to each source
    unique_sources = sorted(source_data.keys())
    # Use a qualitative color palette that handles many categories well
    colors_palette = plt.cm.Set3(np.linspace(0, 1, len(unique_sources)))
    source_colors = {source: colors_palette[i] for i, source in enumerate(unique_sources)}
    
    # Also keep priority-based colors for certain plots
    priority_colors = plt.cm.viridis(np.linspace(0, 1, 6))  # 5 priority levels + 1
    
    # Create figure with subplots (3x3 grid)
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Total characters by source
    ax1 = plt.subplot(3, 3, 1)
    sources_sorted = sorted(source_data.keys(), key=lambda x: sum(source_data[x]['lengths']), reverse=True)
    total_chars = [sum(source_data[s]['lengths']) for s in sources_sorted]
    colors_total = [source_colors[s] for s in sources_sorted]
    
    bars = ax1.bar(range(len(sources_sorted)), total_chars, color=colors_total)
    ax1.set_xticks(range(len(sources_sorted)))
    ax1.set_xticklabels(sources_sorted, rotation=45, ha='right')
    ax1.set_ylabel('Total Characters (millions)')
    ax1.set_title('Total Code Volume by Source')
    
    # Format y-axis in millions
    from matplotlib.ticker import FuncFormatter
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{int(x/1e6)}M'))
    
    # Add value labels
    for i, (bar, chars) in enumerate(zip(bars, total_chars)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1e5, 
                f'{chars/1e6:.1f}M', ha='center', va='bottom', fontsize=8)
    
    # 2. Sample count by source
    ax2 = plt.subplot(3, 3, 2)
    sources = sorted(source_data.keys(), key=lambda x: source_data[x]['count'], reverse=True)
    counts = [source_data[s]['count'] for s in sources]
    colors_bar = [source_colors[s] for s in sources]
    
    bars = ax2.bar(range(len(sources)), counts, color=colors_bar)
    ax2.set_xticks(range(len(sources)))
    ax2.set_xticklabels(sources, rotation=45, ha='right')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Sample Count by Source')
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(count), ha='center', va='bottom', fontsize=8)
    
    # 3. Median code length by source (moved to first row)
    ax3 = plt.subplot(3, 3, 3)
    sources_by_median = sorted(source_data.keys(), 
                              key=lambda x: np.median(source_data[x]['lengths']) if source_data[x]['lengths'] else 0)
    medians = [np.median(source_data[s]['lengths']) if source_data[s]['lengths'] else 0 
               for s in sources_by_median]
    colors_median = [source_colors[s] for s in sources_by_median]
    
    ax3.barh(range(len(sources_by_median)), medians, color=colors_median)
    ax3.set_yticks(range(len(sources_by_median)))
    ax3.set_yticklabels(sources_by_median)
    ax3.set_xlabel('Median Code Length (characters)')
    ax3.set_xscale('log')
    ax3.set_title('Median Code Length by Source')
    ax3.grid(True, axis='x', alpha=0.3)
    
    # 4. Box plot by source (all sources)
    ax4 = plt.subplot(3, 3, 4)
    box_data = []
    box_labels = []
    for source_id in sorted(source_data.keys()):
        if source_data[source_id]['lengths']:
            box_data.append(source_data[source_id]['lengths'])
            box_labels.append(source_id)
    
    bp = ax4.boxplot(box_data, patch_artist=True, labels=box_labels)
    ax4.set_yscale('log')
    ax4.set_ylabel('Code Length (characters, log scale)')
    ax4.set_title('Code Length Ranges by Source')
    ax4.tick_params(axis='x', rotation=45)
    
    # Color boxes by source
    for patch, source_id in zip(bp['boxes'], box_labels):
        patch.set_facecolor(source_colors[source_id])
    
    # 5. Code length categories distribution with more buckets
    ax5 = plt.subplot(3, 3, 5)
    categories = ['<500', '500-1K', '1K-2K', '2K-3K', '3K-5K', '5K-10K', '10K-30K', '>30K']
    category_counts = defaultdict(int)
    
    for length in all_lengths:
        if length < 500:
            category_counts['<500'] += 1
        elif length < 1000:
            category_counts['500-1K'] += 1
        elif length < 2000:
            category_counts['1K-2K'] += 1
        elif length < 3000:
            category_counts['2K-3K'] += 1
        elif length < 5000:
            category_counts['3K-5K'] += 1
        elif length < 10000:
            category_counts['5K-10K'] += 1
        elif length < 30000:
            category_counts['10K-30K'] += 1
        else:
            category_counts['>30K'] += 1
    
    cat_values = [category_counts[cat] for cat in categories]
    bars = ax5.bar(categories, cat_values, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Code Length Category')
    ax5.set_ylabel('Number of Samples')
    ax5.set_title('Length Distribution')
    ax5.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    total = sum(cat_values)
    for bar, val in zip(bars, cat_values):
        percentage = (val / total) * 100
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{val}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=8)
    
    # 6. Source contribution by length category (stacked bar) - ALL sources
    ax6 = plt.subplot(3, 3, 6)
    
    # Calculate contributions
    source_by_category = defaultdict(lambda: defaultdict(int))
    for source_id, data in source_data.items():
        for length in data['lengths']:
            if length < 500:
                cat = '<500'
            elif length < 1000:
                cat = '500-1K'
            elif length < 2000:
                cat = '1K-2K'
            elif length < 3000:
                cat = '2K-3K'
            elif length < 5000:
                cat = '3K-5K'
            elif length < 10000:
                cat = '5K-10K'
            elif length < 30000:
                cat = '10K-30K'
            else:
                cat = '>30K'
            source_by_category[cat][source_id] += 1
    
    # Sort sources by total count for consistent stacking order
    all_sources_sorted = sorted(source_data.keys(), key=lambda x: source_data[x]['count'], reverse=True)
    
    bottom = np.zeros(len(categories))
    for source_id in all_sources_sorted:
        values = [source_by_category[cat].get(source_id, 0) for cat in categories]
        if sum(values) > 0:  # Only plot sources with data
            ax6.bar(categories, values, bottom=bottom, label=source_id, color=source_colors[source_id])
            bottom += values
    
    ax6.set_xlabel('Code Length Category')
    ax6.set_ylabel('Number of Samples')
    ax6.set_title('Source Contributions by Length')
    ax6.legend(loc='upper right', fontsize=7, ncol=2)
    ax6.tick_params(axis='x', rotation=45)
    
    # 7. CDF by source (top 5 sources)
    ax7 = plt.subplot(3, 3, 7)
    
    # Get top 5 sources by count
    top_sources = sorted(source_data.keys(), key=lambda x: source_data[x]['count'], reverse=True)[:5]
    
    for source_id in top_sources:
        lengths = sorted(source_data[source_id]['lengths'])
        if lengths:
            cumulative = np.arange(1, len(lengths) + 1) / len(lengths)
            ax7.plot(lengths, cumulative, label=source_id, linewidth=2, color=source_colors[source_id])
    
    ax7.set_xscale('log')
    ax7.set_xlabel('Code Length (characters)')
    ax7.set_ylabel('Cumulative Probability')
    ax7.set_title('Length CDF - Top 5 Sources')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Mean code length by source
    ax8 = plt.subplot(3, 3, 8)
    sources_by_mean = sorted(source_data.keys(), 
                            key=lambda x: np.mean(source_data[x]['lengths']) if source_data[x]['lengths'] else 0)
    means = [np.mean(source_data[s]['lengths']) if source_data[s]['lengths'] else 0 
             for s in sources_by_mean]
    colors_mean = [source_colors[s] for s in sources_by_mean]
    
    ax8.barh(range(len(sources_by_mean)), means, color=colors_mean)
    ax8.set_yticks(range(len(sources_by_mean)))
    ax8.set_yticklabels(sources_by_mean)
    ax8.set_xlabel('Mean Code Length (characters)')
    ax8.set_xscale('log')
    ax8.set_title('Mean Code Length by Source')
    ax8.grid(True, axis='x', alpha=0.3)
    
    # 9. Description length distribution by source
    ax9 = plt.subplot(3, 3, 9)
    
    # Calculate description length statistics for each source
    desc_data = []
    desc_labels = []
    desc_colors = []
    
    for source_id in sorted(source_data.keys(), key=lambda x: np.median(source_data[x]['desc_lengths']) if source_data[x]['desc_lengths'] and any(l > 0 for l in source_data[x]['desc_lengths']) else 0, reverse=True):
        desc_lengths = source_data[source_id]['desc_lengths']
        # Only include sources that have meaningful descriptions (not all zeros)
        if desc_lengths and any(l > 0 for l in desc_lengths):
            desc_data.append(desc_lengths)
            desc_labels.append(source_id)
            desc_colors.append(source_colors[source_id])
    
    if desc_data:  # Only create plot if we have description data
        bp = ax9.boxplot(desc_data, patch_artist=True, labels=desc_labels)
        ax9.set_yscale('log')
        ax9.set_ylabel('Description Length (characters, log scale)')
        ax9.set_title('Description Length Distribution')
        ax9.tick_params(axis='x', rotation=45)
        
        # Color boxes by source
        for patch, color in zip(bp['boxes'], desc_colors):
            patch.set_facecolor(color)
    else:
        # If no description data, show a message
        ax9.text(0.5, 0.5, 'No description data\navailable', ha='center', va='center', 
                transform=ax9.transAxes, fontsize=12)
        ax9.set_title('Description Length Distribution')
        ax9.set_xticks([])
        ax9.set_yticks([])
    
    # Adjust layout and save
    try:
        plt.tight_layout()
    except:
        # If tight_layout fails, use constrained_layout
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)
    
    # Save the plot
    output_file = Path(output_path)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Saved data sources visualization to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("CODE LENGTH DISTRIBUTION SUMMARY")
    print("="*60)
    print(f"Total samples analyzed: {len(all_lengths):,}")
    print(f"Overall median length: {overall_median:.0f} characters")
    print(f"Overall mean length: {overall_mean:.0f} characters")
    print(f"Min length: {np.min(all_lengths):.0f} characters")
    print(f"Max length: {np.max(all_lengths):.0f} characters")
    print(f"Standard deviation: {np.std(all_lengths):.0f} characters")
    
    print("\nSource Summary:")
    for source_id in sorted(source_data.keys(), key=lambda x: source_data[x]['count'], reverse=True):
        data = source_data[source_id]
        if data['lengths']:
            median = np.median(data['lengths'])
            mean = np.mean(data['lengths'])
            print(f"  {source_id}: {data['count']} samples, median={median:.0f}, mean={mean:.0f}, priority={data['priority']}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    analyze_code_lengths()