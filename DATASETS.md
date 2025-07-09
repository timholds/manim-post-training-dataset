# Existing Datasets: 

## Successfully Integrated (July 9, 2025)
- **Bespoke Manim** (1000 examples) ✅
  - Source: https://huggingface.co/datasets/bespokelabs/bespoke-manim
  - Contains: code, rich descriptions, transcripts, some videos
  - Status: Fully integrated - 1,000 samples processed
  
- **Thanks Dataset** (4400 examples) ✅
  - Source: https://huggingface.co/datasets/thanhkt/manim_code
  - Contains: code, instructions
  - Status: Fully integrated - 4,395 samples processed
  
- **ManimCodeGen** (1600 examples) ✅
  - Source: https://huggingface.co/datasets/generaleoley/manim-codegen
  - Contains: code, queries
  - Status: Fully integrated - 1,622 samples processed

- **ManimBench** (417 examples) ✅
  - Source: https://www.kaggle.com/datasets/ravidussilva/manim-sft/data
  - Contains: code, reviewed descriptions, proper train/test split
  - Status: Fully integrated - 417 samples processed
  - Note: 100% unique content, no overlap with other datasets

## Current Dataset Statistics (WITH Source Tracking)
- **Total Raw Samples**: 7,434
- **Training Samples**: 16,747 (with 2.5x augmentation)
- **Test Samples**: 743
- **Output Location**: `data_formatted_with_sources/`

### Source Distribution in Training Set:
- Thanks Dataset: 9,955 samples (59.4%)
- ManimCodeGen: 3,616 samples (21.6%)
- Bespoke Manim: 2,253 samples (13.5%)
- ManimBench: 923 samples (5.5%)

## Known Issues
- **Significant duplication detected**: ~30% of samples have duplicate descriptions
- **Deduplication needed**: Planning to implement smart deduplication while preserving diversity

# Potential Datasets for Future Integration 
- Szymon Ozog Videos Information Theory Videos https://github.com/SzymonOzog/InformationTheory/tree/main youtube playlist
- Szymon Ozog GPU Programming Videos https://github.com/SzymonOzog/GPU_Programming, youtube playlist https://www.youtube.com/watch?v=c8mQYGbT310&list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j
- Manim CE Examples, https://docs.manim.community/en/stable/examples.html, code yes, video yes, transcript no, date no  
- Manim Repository, https://themanimrepository.wordpress.com/, code yes, video yes, transcript no, date yes  
- Manim CE Awesome manim, https://github.com/ManimCommunity/awesome-manim (look at README to find those with GitHub/youtube pairs)  
- Dan4Life, videos yes https://www.youtube.com/@dan4life/videos, code yes https://github.com/Dan4Life/AoC2024_Videos, 
- A Little More Than An Introduction To Series - code yes https://github.com/JonathanWoollett-Light/a-little-more-than-an-introduction-to, video yes videos https://www.youtube.com/channel/UCze6YPZo6gzj-Nup2P59KUA, , transcript yes, date yes, 
- Kilacola (2) video yes https://www.youtube.com/channel/UCYiEcjVorHS78RgoqKiIFgQ, code yes https://github.com/kilacoda/videos, 
- Reducible (long videos) video yes, https://www.youtube.com/@Reducible/videos, code yes https://github.com/nipunramk/Reducible
- Kutuzova (5), videos yes https://www.youtube.com/@deeplearningthatworks/videos, code yes (ipynb) https://github.com/sgalkina/animations/tree/main/notebooks  
- Visualizing Deep Learning (2), videos yes (playlist) https://www.youtube.com/playlist?list=PLyPKqVSnetmEOp_g_hfabuRAs9ET-shl_, code yes https://github.com/vivek3141/dl-visualization  
- Vivek3141 (22) https://www.youtube.com/@vcubingx/videos, code yes https://github.com/vivek3141/videos (might have overlap with Visualizing Deep Learning series)


