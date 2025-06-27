# Ultimate Manim Dataset Creation Plan

## Current Status

We've successfully analyzed and combined two major Manim datasets:

1. **Bespoke Manim** (1000 samples)
   - 100% valid syntax
   - Rich metadata: narration, questions, titles, subjects
   - Educational focus with video outputs
   - Average 73 lines per sample

2. **ManimCodeGen** (1622 samples)
   - 60.9% valid syntax after cleanup
   - Query-response format
   - Shorter samples (avg 18 lines)
   - Good for simple animations

Combined dataset after quality filtering and deduplication: **702 high-quality samples**

## Quality Criteria Implemented

1. **Syntax Validation**: All samples have valid Python syntax
2. **Structure Requirements**: Must have Scene class and construct method
3. **Complexity Balance**: Distributed across basic/intermediate/advanced
4. **Deduplication**: Removed near-duplicates (85% similarity threshold)
5. **Animation Richness**: Minimum 2 animation method calls

## Next Steps to Scale

### 1. Add More High-Quality Datasets

From DATASETS.md, prioritize these sources:

**High Priority** (have code + descriptions):
- **Thanks dataset** (4400 samples) - Largest available
- **Dan4Life** - AoC2024 videos with matching GitHub code
- **Reducible** - High-quality educational content with code

**Medium Priority** (need transcript extraction):
- **Manim CE Examples** - Official examples
- **vcubingx videos** - 22 videos with code
- **Visualizing Deep Learning** - Specialized content

### 2. Data Augmentation Strategies

```python
# Planned augmentations:
1. Variable renaming (preserve functionality)
2. Color scheme variations
3. Animation timing modifications
4. Mathematical expression alternatives
5. Scene composition variations
```

### 3. Validation Pipeline

```python
# Implement Manim compilation checks:
1. Syntax validation ✓
2. Import resolution
3. Scene instantiation test
4. Render first frame test
5. Full compilation (subset)
```

### 4. Target Dataset Structure

```
ultimate_manim_dataset/
├── train.json          # 10,000+ samples
├── test.json           # 1,000+ samples
├── validation.json     # 500+ samples
├── metadata.json       # Dataset statistics
├── by_category/        # Organized by topic
│   ├── geometry/
│   ├── algebra/
│   ├── calculus/
│   └── ...
└── by_complexity/      # Organized by difficulty
    ├── beginner/
    ├── intermediate/
    └── advanced/
```

### 5. Instruction Format Enhancement

Current format:
```json
{
  "instruction": "Create a Manim animation to demonstrate: [topic]",
  "response": "[manim code]",
  "metadata": {
    "source": "dataset_name",
    "quality_score": 0.8,
    "complexity": "intermediate",
    "has_video": true
  }
}
```

Enhanced format (planned):
```json
{
  "instruction": "Create a Manim animation to demonstrate: [topic]",
  "response": "[manim code]",
  "metadata": {
    "source": "dataset_name",
    "quality_score": 0.8,
    "complexity": "intermediate",
    "has_video": true,
    "concepts": ["geometry", "transformations"],
    "manim_features": ["VGroup", "Transform", "MathTex"],
    "estimated_duration": 15,
    "compilation_tested": true
  }
}
```

## Implementation Priority

1. **Immediate** (Today):
   - Download and process Thanks dataset (4400 samples)
   - Implement basic augmentation (variable renaming)
   - Set up compilation validation framework

2. **Short Term** (This Week):
   - Add 3-4 more datasets from high-priority list
   - Implement full augmentation pipeline
   - Create category-based organization

3. **Medium Term**:
   - Reach 10,000+ training samples
   - Add synthetic data generation using GPT-4
   - Create difficulty progression curriculum

## Success Metrics

- [ ] 10,000+ high-quality training samples
- [ ] 95%+ compilation success rate
- [ ] Balanced distribution across mathematical topics
- [ ] Clear difficulty progression
- [ ] Rich metadata for curriculum learning

## Risks and Mitigations

1. **Code Quality Variation**: Mitigate with strict validation
2. **License Concerns**: Document sources and licenses
3. **Rendering Cost**: Use lazy validation (syntax first, render subset)
4. **Dataset Imbalance**: Use sampling strategies and augmentation