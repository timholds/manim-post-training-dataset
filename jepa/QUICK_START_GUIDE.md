# JEPA-Manim Quick Start Guide

## Project Overview
Using JEPA (Joint-Embedding Predictive Architecture) to solve Manim's layout problem by learning spatial relationships from 3Blue1Brown videos.

## Key Documents
1. **[JEPA_MANIM_PLAN.md](./JEPA_MANIM_PLAN.md)** - Complete implementation plan with milestones
2. **[DATASET_SPECIFICATION.md](./DATASET_SPECIFICATION.md)** - How to use 3b1b_dataset for training
3. **[TECHNICAL_ARCHITECTURE.md](./TECHNICAL_ARCHITECTURE.md)** - Detailed system design

## Quick Implementation Steps

### Week 1: Proof of Concept
```bash
# 1. Generate synthetic data
python scripts/generate_synthetic_scenes.py --count 100

# 2. Train basic layout predictor
python train_layout.py --data synthetic --epochs 10

# 3. Test position prediction
python evaluate_layout.py --checkpoint latest
```

### Week 2: Real Data Integration
```bash
# 1. Extract frames from 3b1b videos
python scripts/extract_3b1b_frames.py --year 2016 --count 100

# 2. Generate layout annotations
python scripts/annotate_layouts.py --auto

# 3. Train on real data
python train_layout.py --data real --pretrained synthetic
```

## Key Insights from Research

### Why JEPA for Manim?
- **Traditional LLMs fail** at layout because they generate sequentially
- **JEPA understands spatial relationships** in latent space
- **Self-supervised learning** works with limited data (500 scenes)

### Critical Success Factors
1. **Start simple**: Static scenes before animations
2. **Use existing models**: CLIP + small code LLM
3. **Layout first**: Positions determine code, not vice versa

## Simplified Architecture
```
Description → CLIP → Layout Predictor → Position Constraints → Code LLM → Manim Code
```

## Expected Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Frame-code alignment | Use synthetic data first |
| Limited data (500 scenes) | Aggressive augmentation (20-50x) |
| Evaluation metrics | Focus on IoU and compilation rate |
| Complex animations | Start with static scenes only |

## Minimum Viable Implementation (2 weeks)

### Week 1 Goals
- [ ] 100 synthetic scenes generated
- [ ] Basic CLIP encoder working
- [ ] Position prediction >70% IoU

### Week 2 Goals
- [ ] Code generation with position constraints
- [ ] 80% compilation success rate
- [ ] 5 real 3b1b scenes working

## Resource Requirements
- **GPU**: 1x RTX 4090 or A6000
- **Storage**: ~200GB for data
- **Time**: 2 weeks for MVP, 10 weeks for full system

## Decision Points
- **Day 5**: Synthetic data working? → Continue or debug
- **Day 10**: Layout prediction >50%? → Proceed or simplify
- **Day 14**: Code generation working? → Scale up or iterate

## Using the 3b1b Dataset

```python
# The dataset provides:
# 1. Matched code files with inlined imports
code_file = "3b1b_dataset/output_v4/2016/dot-products/code.py"

# 2. Video metadata and URLs
metadata = load_json("3b1b_dataset/output/matching_summary_2015.json")

# 3. YouTube transcripts
transcript = "3b1b_dataset/data/youtube_transcripts/2016/..."
```

## Next Immediate Actions

1. **Set up environment**
   ```bash
   cd /Users/timholdsworth/code/manim-post-training
   pip install -r requirements.txt
   ```

2. **Create synthetic data generator**
   ```bash
   mkdir -p scripts
   touch scripts/generate_synthetic_scenes.py
   ```

3. **Implement basic training loop**
   ```bash
   touch train_layout.py
   ```

## Success Metrics

### Phase 1 (2 weeks)
- Synthetic scenes: 70% IoU
- Code compilation: 80% success

### Phase 2 (6 weeks)  
- Real scenes: 60% IoU
- Human eval: "looks right" 50%

### Phase 3 (10 weeks)
- Production ready
- <5 second generation
- API deployed

---

**Remember**: Start small, prove it works, then scale. The biggest risk is overengineering before validating the core hypothesis.