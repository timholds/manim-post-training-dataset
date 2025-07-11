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

#### Infrastructure
- [x] Unified conversation format implementation
- [x] Deduplication pipeline (removes ~40% duplicates)
- [x] YouTube metadata integration for transcript enhancement
- [x] Extraction scripts for GitHub repositories
- [x] Enhanced data preparation pipeline

#### Current Statistics
- **Total Unique Samples**: 3,869
- **Training Set**: 3,483 samples
- **Test Set**: 386 samples
- **Sources**: 5 active (all working properly)

### 🚧 In Progress

#### Quality Improvements
- [ ] Generate LLM-enhanced descriptions for samples marked with `needs_description_update`
- [ ] Process YouTube transcripts for Szymon Ozog samples

#### Documentation
- [ ] Update all dataset statistics after Bespoke Manim fix
- [ ] Create comprehensive integration guide for new datasets

### 📋 Planned

#### Quick Win Datasets (from ROADMAP.md)
- [ ] Manim CE Examples (~20 samples)
- [ ] The Manim Repository (9 samples)
- [ ] Kilacola Chemistry (~7 samples)

#### Medium Effort Datasets
- [ ] Vivek3141 (~40-45 samples)
- [ ] A Little More Than An Introduction To Series

#### Large Datasets
- [ ] Reducible (~50 samples)
- [ ] Benjamin Hackl (~100 samples)
- [ ] Chilao (~50 samples)

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