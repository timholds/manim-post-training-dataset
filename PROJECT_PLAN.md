# Manim Fine-Tuning Dataset Project Plan

## Project Goal
Construct a perfect manim fine-tuning dataset from multiple sources with high-quality descriptions and diverse animation examples.

## Progress Tracker

### ✅ Completed (as of July 11, 2025)

#### Dataset Integration
- [x] **ManimBench** - 417 samples (Kaggle)
- [x] **Thanks Dataset** - 4,395 samples (HuggingFace)
- [x] **Dan4Life AoC2024** - 24 samples (GitHub)
- [x] **Bespoke Manim** - 1,000 samples (HuggingFace) ✅ Fixed!
- [x] **Szymon Ozog** - 29 samples, 15 after deduplication (GitHub)
  - Information Theory: 10 scenes
  - GPU Programming: 19 scenes
- [x] **Reducible** - 250 samples (GitHub) ✅ NEW!
  - Computer Science animations from popular YouTube channel
  - Topics: algorithms, data structures, FFT, compression, graph theory
  - Years covered: 2019-2022

#### Infrastructure
- [x] Unified conversation format implementation
- [x] Deduplication pipeline (removes ~33% duplicates)
- [x] YouTube metadata integration for transcript enhancement
- [x] Extraction scripts for GitHub repositories
- [x] Enhanced data preparation pipeline

#### Current Statistics (as of July 11, 2025)
- **Total Unique Samples**: 4,077 + ~310 new = ~4,387 expected
- **Training Set**: 3,670 samples (9,185 with augmentation)
- **Test Set**: 407 samples
- **Sources**: 9 active (6 integrated, 3 newly added extractor-based)

### 🚧 In Progress

#### Quality Improvements
- [ ] Generate LLM-enhanced descriptions for samples marked with `needs_description_update`
- [ ] Process YouTube transcripts for Szymon Ozog samples

#### Documentation
- [ ] Update all dataset statistics after Bespoke Manim fix
- [ ] Create comprehensive integration guide for new datasets

### 📋 Planned

#### Quick Win Datasets (from ROADMAP.md)
- [ ] Manim CE Examples (~20 samples) - Already extracted in data_manim_ce_examples.jsonl
- [ ] The Manim Repository (9 samples)
- [ ] Kilacola Chemistry (~7 samples)

#### Medium Effort Datasets
- [x] **Vivek3141** (~300 samples) - ✅ Integrated!
  - Main videos repository with educational animations
  - Topics: math, AI, physics, computational concepts
- [x] **Vivek3141 DL Series** (~5 samples) - ✅ Integrated!
  - Specialized deep learning visualizations
  - High priority (5) for quality DL content
- [x] **Kutuzova (Deep Learning That Works)** (~5 samples) - ✅ Integrated!
  - Jupyter notebook-based animations
  - Deep learning educational content
- [ ] A Little More Than An Introduction To Series
- [ ] Benjamin Hackl (~10-15 samples from manim-with-ease/manim-content)

#### Large Datasets
- [ ] Chilao (~50 samples)
- [ ] 3Blue1Brown Archive (if available)

#### Quality Improvements
- [ ] Implement transcript-based description generation
- [ ] Add prompt augmentation strategies
- [ ] Create quality scoring system

## Next Steps

1. **Immediate**: Integrate quick-win datasets to reach 4,000+ unique samples
   - Manim CE Examples (~20 samples) 
   - The Manim Repository (9 samples)
   - Kilacola Chemistry (~7 samples)
2. **Short-term**: Continue with medium-effort datasets
   - Vivek3141 (~40-45 samples)
   - More from ROADMAP.md quick wins
3. **Medium-term**: Process descriptions with YouTube transcripts (after dataset expansion)
4. **Long-term**: Build tooling for continuous dataset expansion and quality improvement

## Notes
- Prioritize quality over quantity
- Maintain source diversity for better generalization
- Use YouTube transcripts for accurate descriptions when available
- Keep deduplication aggressive to ensure uniqueness