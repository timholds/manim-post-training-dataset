# Manim Dataset Integration Project Plan

## Executive Summary
This plan details the integration of 10+ high-priority Manim datasets into our fine-tuning dataset. Each dataset has been evaluated for ease of integration, data availability, and quality. The top candidates offer clear code-video pairings and can be scraped using established patterns. **Note: 3Blue1Brown's content is excluded as it uses ManimGL, not ManimCE.**

## Required Data Format
```json
{
  "conversations": [
    {
      "from": "system",
      "value": "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."
    },
    {
      "from": "user",
      "value": "<description of animation to create>"
    },
    {
      "from": "assistant",
      "value": "```python\n<manim code>\n```"
    }
  ],
  "source": "<dataset_name>"
}
```

## Top 10+ High-Yield Datasets for Integration

### 1. **Dan4Life AoC2024** ⭐ HIGHEST PRIORITY
- **GitHub**: https://github.com/Dan4Life/AoC2024_Videos
- **YouTube**: https://www.youtube.com/@dan4life/videos
- **Repository Details**:
  - 25 day-specific directories (Day_01 through Day_25)
  - 100% Python-based, uses Manim v0.18.1
  - Clear, consistent folder structure
  - CC BY-NC-SA 4.0 license
- **Advantages**:
  - Recent content (December 2024)
  - Predictable naming convention
  - Each day is self-contained
  - ~25 high-quality animations
- **Integration Steps**:
  1. Clone repository: `git clone https://github.com/Dan4Life/AoC2024_Videos`
  2. Iterate through Day_01 to Day_25 directories
  3. Extract main animation files from each day
  4. Scrape AoC 2024 problem descriptions for context
  5. Generate descriptions: "Create a Manim animation visualizing Advent of Code 2024 Day X: [problem title]"
  6. Format into required JSON structure

### 2. **Manim CE Official Examples** ⭐ HIGH PRIORITY & QUICK WIN
- **URL**: https://docs.manim.community/en/stable/examples.html
- **Content Details**:
  - 20 total examples across 4 categories
  - Categories: Basic Concepts, Animations, Plotting, Special Camera Settings
  - Each example has code + static preview image
  - Well-documented with class references
- **Advantages**:
  - Official, high-quality code
  - Already includes descriptions
  - No video matching needed
  - Guaranteed to work with latest Manim
- **Integration Steps**:
  1. Scrape page with BeautifulSoup targeting code blocks
  2. Extract example titles and descriptions from headings
  3. Parse Python code blocks (look for `class.*Scene` patterns)
  4. Use section headers + example names for descriptions
  5. Format into required JSON structure (20 samples)

### 3. **Kilacola Chemistry Animations** ⭐ QUICK VALIDATION
- **GitHub**: https://github.com/kilacoda/videos
- **Repository Details**:
  - 7 chemistry-focused files (e.g., luminol.py, pinacol.py, Markovnikoff_addition.py)
  - Includes references directory
  - Educational chemistry animations
- **Advantages**:
  - Small dataset - perfect for testing pipeline
  - Unique chemistry focus adds diversity
  - Clear, descriptive filenames
- **Integration Steps**:
  1. Clone repository
  2. Extract all .py files (excluding references)
  3. Generate descriptions from filenames: "Create a Manim animation demonstrating [chemistry concept]"
  4. Add chemistry context to descriptions
  5. Format into required JSON structure (~7 samples)

### 4. **Vivek3141 Videos** ⭐ LARGE DATASET
- **GitHub**: https://github.com/vivek3141/videos
- **Repository Details**:
  - 50 Python files total
  - Topics: math, AI, physics, computational concepts
  - Subdirectories: alg1/lineq, images, img, shaders
  - Descriptive naming (e.g., complex_derivative.py, green_theorem.py)
- **Advantages**:
  - Large, diverse dataset
  - High-quality educational content
  - Popular YouTube channel
- **Integration Steps**:
  1. Clone repository
  2. Filter out non-animation files (images, shaders)
  3. Extract topic from filename (e.g., "complex_derivative" → "complex derivatives")
  4. Scrape YouTube channel for video titles matching filenames
  5. Generate descriptions: "Create a Manim animation explaining [topic]"
  6. Handle special cases (dl-visualization overlap)
  7. Format into required JSON structure (~40-45 samples)

### 5. **Reducible** ⭐⭐⭐ MAJOR DATASET
- **GitHub**: https://github.com/nipunramk/Reducible
- **YouTube**: https://www.youtube.com/@Reducible
- **Repository Details**:
  - Organized by year (2019-2022)
  - 50+ high-quality CS/algorithm animations
  - Topics: Graph theory, GJK algorithm, marching squares
  - Well-structured with custom libraries
- **Advantages**:
  - Popular channel with clear explanations
  - Computer science focus adds diversity
  - Multi-year content = established patterns
- **Integration Steps**:
  1. Clone repository
  2. Iterate through year folders (2019-2022)
  3. Extract scene files from each year
  4. Match to YouTube videos by upload date/title
  5. Generate CS-focused descriptions
  6. Format into required JSON structure (~50 samples)

### 6. **Benjamin Hackl** ⭐⭐ TUTORIAL GOLDMINE
- **GitHub**: https://github.com/behackl
- **YouTube**: https://www.youtube.com/@BenjaminHackl
- **Repository Details**:
  - 100+ animations
  - Manim tutorials + physics/math content
  - Well-documented code
- **Advantages**:
  - Tutorial-style content
  - Clear code documentation
  - Diverse topics
- **Integration Steps**:
  1. Clone repositories
  2. Focus on tutorial series first
  3. Extract tutorial descriptions
  4. Format into required JSON structure (~100 samples)

### 7. **Manim Kindergarten** ⭐⭐ CHINESE CONTENT
- **GitHub**: https://github.com/manim-kindergarten
- **YouTube**: Multiple channels
- **Repository Details**:
  - 100+ tutorials
  - Chinese language content
  - Comprehensive Manim coverage
- **Advantages**:
  - Large tutorial collection
  - Different cultural perspective
  - Basic to advanced topics
- **Note**: May need translation for descriptions

### 8. **Chilao** ⭐ PHYSICS FOCUS
- **GitHub**: https://github.com/chilaochen/manim_projects
- **YouTube**: https://www.youtube.com/@chilao
- **Repository Details**:
  - 50+ animations
  - Physics and math focus
  - Tutorial content included
- **Integration Steps**:
  1. Clone manim_projects repository
  2. Extract physics simulations
  3. Generate physics-focused descriptions
  4. Format into required JSON structure (~50 samples)

### 9. **The Manim Repository (WordPress)** ⭐ QUICK WIN
- **URL**: https://themanimrepository.wordpress.com/
- **Content Details**:
  - 9 complete animations with code
  - Posted September 2022
  - Topics: 3D curves, fractals, modular arithmetic
  - Full code included in posts
- **Advantages**:
  - Code directly available on site
  - No GitHub cloning needed
  - Diverse mathematical topics
- **Integration Steps**:
  1. Scrape all 9 posts
  2. Extract code blocks and titles
  3. Generate descriptions from post titles
  4. Format into required JSON structure (9 samples)

### 10. **Szymon Ozog Playlists** ✅ COMPLETED (July 11, 2025)
- **Information Theory**: https://github.com/SzymonOzog/InformationTheory
  - 3 files focusing on entropy and information theory
  - Has associated YouTube playlist
  - **Result**: 10 scenes extracted
- **GPU Programming**: https://github.com/SzymonOzog/GPU_Programming
  - Contains manim_scripts directory
  - CUDA/GPU visualization potential
  - **Result**: 19 scenes extracted from 18 files
- **Integration Status**:
  - Total: 29 samples extracted (28 after deduplication)
  - All samples have YouTube metadata for transcript enhancement
  - Successfully integrated into main dataset
  - Added specialized content in information theory and GPU visualization

### 11. **A Little More Than An Introduction To Series** ⭐ NEEDS INVESTIGATION
- **GitHub**: https://github.com/JonathanWoollett-Light/a-little-more-than-an-introduction-to
- **YouTube**: https://www.youtube.com/channel/UCze6YPZo6gzj-Nup2P59KUA
- **Repository Details**:
  - Neural network focused animations
  - Uses numeric naming (0.py, etc.)
  - Contains nn/ directory
  - Has transcripts and dates available
  - Execution: `manim -pql 0.py EpisodeScene`
- **Challenges**:
  - Numeric filenames make matching difficult
  - Need to explore repository further
  - May require manual video-to-code mapping
- **Integration Steps**:
  1. Clone and explore full repository structure
  2. Check for episode listing or index
  3. Match numeric files to YouTube videos by upload order
  4. Extract descriptions from video titles/transcripts
  5. Format into required JSON structure

### 12. **Kutuzova (Deep Learning That Works)** ⭐ JUPYTER NOTEBOOKS
- **GitHub**: https://github.com/sgalkina/animations/tree/main/notebooks
- **YouTube**: https://www.youtube.com/@deeplearningthatworks/videos
- **Repository Details**:
  - 5+ animations in Jupyter notebook format
  - Deep learning focused content
  - May need notebook-to-script conversion
- **Integration Steps**:
  1. Clone repository
  2. Extract Manim code from notebooks
  3. Match to YouTube videos
  4. Generate deep learning focused descriptions
  5. Format into required JSON structure (~5 samples)

### 13. **Visualizing Deep Learning** ⭐ VIVEK3141 SERIES
- **GitHub**: https://github.com/vivek3141/dl-visualization
- **YouTube Playlist**: https://www.youtube.com/playlist?list=PLyPKqVSnetmEOp_g_hfabuRAs9ET-shl_
- **Repository Details**:
  - 2+ dedicated deep learning visualization videos
  - May overlap with main vivek3141 repository
  - High-quality neural network animations
- **Integration Steps**:
  1. Clone repository
  2. Check for overlap with vivek3141/videos
  3. Extract unique animations
  4. Generate DL-specific descriptions
  5. Format into required JSON structure (~2-5 samples)

## Implementation Pipeline

### Phase 1: Environment Setup
```bash
# Install required packages
uv pip install yt-dlp beautifulsoup4 requests pandas tqdm

# Create workspace
mkdir -p data_sources/{introduction_series,dan4life,kilacola,manim_examples,vivek3141}
```

### Phase 2: Data Collection Script Templates

#### Core Scraping Functions
```python
# scrape_dataset.py
import json
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import re
from bs4 import BeautifulSoup
import requests

SYSTEM_PROMPT = "You are a Manim code generator. Create clean, working Manim animations using ManimCE syntax. Always wrap code in Python code blocks."

def scrape_github_code(repo_url: str, local_dir: str) -> Dict[str, str]:
    """Clone repo and extract .py files"""
    subprocess.run(["git", "clone", repo_url, local_dir], check=True)
    
    code_files = {}
    for py_file in Path(local_dir).rglob("*.py"):
        # Skip non-animation files
        if any(skip in str(py_file) for skip in ["__pycache__", "test", "util", "config"]):
            continue
        
        with open(py_file, 'r', encoding='utf-8') as f:
            code_files[py_file.stem] = f.read()
    
    return code_files

def scrape_aoc_descriptions() -> Dict[int, str]:
    """Scrape Advent of Code 2024 problem titles"""
    aoc_titles = {}
    for day in range(1, 26):
        url = f"https://adventofcode.com/2024/day/{day}"
        # Note: May need session cookie for full access
        resp = requests.get(url)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, 'html.parser')
            title = soup.find('h2').text.strip()
            aoc_titles[day] = title.replace("---", "").strip()
    return aoc_titles

def format_to_jsonl(code: str, description: str, source: str) -> Dict:
    """Format single example to required structure"""
    return {
        "conversations": [
            {"from": "system", "value": SYSTEM_PROMPT},
            {"from": "user", "value": description},
            {"from": "assistant", "value": f"```python\n{code.strip()}\n```"}
        ],
        "source": source
    }
```

#### Dataset-Specific Scrapers

##### 1. Dan4Life AoC2024 Scraper
```python
def scrape_dan4life():
    """Scrape Dan4Life AoC2024 animations"""
    repo_url = "https://github.com/Dan4Life/AoC2024_Videos"
    
    # Clone repository
    code_files = scrape_github_code(repo_url, "data_sources/dan4life")
    aoc_titles = scrape_aoc_descriptions()
    
    samples = []
    for day in range(1, 26):
        day_dir = f"Day_{day:02d}"
        day_path = Path(f"data_sources/dan4life/{day_dir}")
        
        if day_path.exists():
            # Find main animation file
            for py_file in day_path.glob("*.py"):
                code = py_file.read_text()
                title = aoc_titles.get(day, f"Day {day}")
                description = f"Create a Manim animation visualizing Advent of Code 2024 Day {day}: {title}"
                
                samples.append(format_to_jsonl(code, description, "dan4life_aoc2024"))
    
    return samples
```

##### 2. Manim CE Examples Scraper
```python
def scrape_manim_examples():
    """Scrape official Manim examples"""
    url = "https://docs.manim.community/en/stable/examples.html"
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, 'html.parser')
    
    samples = []
    current_section = ""
    
    for element in soup.find_all(['h2', 'div']):
        if element.name == 'h2':
            current_section = element.text.strip()
        elif 'highlight-python' in element.get('class', []):
            code = element.find('pre').text
            # Extract example name from preceding heading
            title = element.find_previous_sibling(['h3', 'h4']).text.strip()
            
            description = f"Create a Manim animation demonstrating {current_section.lower()}: {title}"
            samples.append(format_to_jsonl(code, description, "manim_ce_examples"))
    
    return samples
```

##### 3. Kilacola Chemistry Scraper
```python
def scrape_kilacola():
    """Scrape Kilacola chemistry animations"""
    repo_url = "https://github.com/kilacoda/videos"
    code_files = scrape_github_code(repo_url, "data_sources/kilacola")
    
    samples = []
    chemistry_descriptions = {
        "luminol": "the chemiluminescence reaction of luminol",
        "pinacol": "the pinacol rearrangement mechanism",
        "Markovnikoff_addition": "Markovnikov's rule in addition reactions"
    }
    
    for filename, code in code_files.items():
        chem_concept = chemistry_descriptions.get(filename, filename.replace("_", " "))
        description = f"Create a Manim animation demonstrating {chem_concept}"
        samples.append(format_to_jsonl(code, description, "kilacola_chemistry"))
    
    return samples
```

### Phase 3: Main Integration Script
```python
# integrate_datasets.py
import json
from pathlib import Path
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_code(code: str) -> bool:
    """Validate that code has proper Manim structure"""
    # Check for imports
    if not any(imp in code for imp in ["from manim import", "import manim"]):
        return False
    
    # Check for Scene class
    if "class" not in code or "Scene" not in code:
        return False
    
    # Check for construct method
    if "def construct" not in code:
        return False
    
    return True

def main():
    """Main integration pipeline"""
    all_samples = []
    
    # Run scrapers in priority order
    logger.info("Scraping Dan4Life AoC2024...")
    all_samples.extend(scrape_dan4life())
    
    logger.info("Scraping Manim CE Examples...")
    all_samples.extend(scrape_manim_examples())
    
    logger.info("Scraping Kilacola Chemistry...")
    all_samples.extend(scrape_kilacola())
    
    # Validate all samples
    valid_samples = []
    for sample in all_samples:
        code = sample["conversations"][2]["value"]
        if validate_code(code):
            valid_samples.append(sample)
        else:
            logger.warning(f"Invalid code structure in {sample['source']}")
    
    logger.info(f"Valid samples: {len(valid_samples)}/{len(all_samples)}")
    
    # Write to JSONL
    output_file = Path("data_sources/new_datasets.jsonl")
    with open(output_file, 'w') as f:
        for sample in valid_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Wrote {len(valid_samples)} samples to {output_file}")
    
    # Generate statistics
    stats = {}
    for sample in valid_samples:
        source = sample['source']
        stats[source] = stats.get(source, 0) + 1
    
    logger.info("Dataset statistics:")
    for source, count in stats.items():
        logger.info(f"  {source}: {count} samples")

if __name__ == "__main__":
    main()
```

### Phase 4: Expected Outputs

#### Sample Counts by Dataset (Revised Estimates)
**Quick Wins (< 2 hours each):**
- **Manim CE Examples**: ~20 samples
- **The Manim Repository**: 9 samples  
- **Kilacola Chemistry**: ~7 samples
- **Szymon Ozog**: ✅ 28 samples (COMPLETED)

**Medium Effort (2-4 hours each):**
- **Dan4Life AoC2024**: ~25 samples
- **Vivek3141**: ~40-45 samples

**Large Datasets (4+ hours each):**
- **Reducible**: ~50 samples
- **Benjamin Hackl**: ~100 samples
- **Chilao**: ~50 samples
- **Manim Kindergarten**: ~100 samples (if translated)

**Total Potential**: 400-450+ samples (excluding 3B1B)

**Realistic First Pass (focusing on English content with clear mappings):**
- Quick wins: ~41 samples
- Dan4Life + Vivek3141: ~70 samples
- Reducible: ~50 samples
- **Total Phase 1**: ~160-200 samples

#### Example Output Format
```json
{
  "conversations": [
    {"from": "system", "value": "You are a Manim code generator..."},
    {"from": "user", "value": "Create a Manim animation visualizing Advent of Code 2024 Day 1: Historian Hysteria"},
    {"from": "assistant", "value": "```python\nfrom manim import *\n\nclass Day01(Scene):\n    def construct(self):\n        # Animation code here\n```"}
  ],
  "source": "dan4life_aoc2024"
}
```

### Phase 5: Quality Validation
1. **Code Structure Validation**
   - Proper imports (`from manim import *` or `import manim`)
   - Scene class definition
   - `construct` method present
   
2. **Description Quality**
   - Clear, actionable instructions
   - Specific about what to animate
   - Consistent formatting

3. **Deduplication Check**
   - Run against existing datasets
   - Remove exact matches
   - Check for similar descriptions

### Phase 6: Integration with Existing Pipeline
```bash
# Merge with existing data
cat data_sources/new_datasets.jsonl >> data_formatted/all_datasets.jsonl

# Run deduplication
python prepare_data_enhanced.py --deduplicate

# Generate train/test split
python prepare_data_enhanced.py --split 0.9
```

## Additional Scraper Templates

### Reducible Scraper
```python
def scrape_reducible():
    """Scrape Reducible CS animations"""
    repo_url = "https://github.com/nipunramk/Reducible"
    code_files = scrape_github_code(repo_url, "data_sources/reducible")
    
    samples = []
    for year in ["2019", "2020", "2021", "2022"]:
        year_path = Path(f"data_sources/reducible/{year}")
        if year_path.exists():
            for py_file in year_path.glob("*.py"):
                # Extract topic from filename
                topic = py_file.stem.replace("_", " ").title()
                description = f"Create a Manim animation explaining {topic} in computer science"
                code = py_file.read_text()
                samples.append(format_to_jsonl(code, description, "reducible"))
    
    return samples
```

### WordPress Manim Repository Scraper
```python
def scrape_manim_repository():
    """Scrape The Manim Repository WordPress site"""
    base_url = "https://themanimrepository.wordpress.com/"
    
    # Get all post URLs (there are 9 as of Sept 2022)
    post_urls = [
        "2022/09/14/infinite-sierpinski-zoom/",
        "2022/09/13/3d-parametric-curves/",
        # ... add all 9 post URLs
    ]
    
    samples = []
    for post_url in post_urls:
        resp = requests.get(base_url + post_url)
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Extract title and code
        title = soup.find('h1', class_='entry-title').text
        code_block = soup.find('pre', class_='wp-block-code')
        
        if code_block:
            code = code_block.text
            description = f"Create a Manim animation: {title}"
            samples.append(format_to_jsonl(code, description, "manim_repository"))
    
    return samples
```

## Success Metrics (Revised)
- **Minimum viable**: 100+ new unique samples
- **Target**: 200+ new unique samples  
- **Stretch goal**: 500+ samples (including large repos)
- **Quality threshold**: <10% rejection rate after validation

## Realistic Time Estimates (Phase 1)
- **Environment setup**: 30 minutes
- **Quick wins** (41 samples): 3-4 hours
  - Manim CE Examples: 2 hours
  - Manim Repository: 1 hour
  - Kilacola: 30 minutes
  - Szymon Ozog: 30 minutes
- **Medium datasets** (70 samples): 4-5 hours
  - Dan4Life: 2 hours
  - Vivek3141: 2-3 hours
- **Reducible** (50 samples): 2-3 hours
- **Integration & validation**: 2 hours

**Total Phase 1**: ~12-15 hours for 160-200 samples

## Priority Execution Order (Revised)
1. **Quick Wins First** (highest ROI)
   - Manim Repository (9 samples, 1 hour)
   - Manim CE Examples (20 samples, 2 hours)
   
2. **Well-Structured Repos**
   - Dan4Life AoC2024 (25 samples, 2 hours)
   - Kilacola (7 samples, 30 min)
   
3. **Medium Complexity**
   - Vivek3141 (45 samples, 3 hours)
   - Reducible (50 samples, 3 hours)
   
4. **Future Phases** (if needed)
   - Benjamin Hackl (100 samples)
   - Chilao (50 samples)
   - Other creators from awesome-manim

## Common Issues & Solutions
- **AoC Scraping**: May need session cookie for full problem descriptions
- **YouTube Matching**: Use fuzzy string matching for video-to-code mapping
- **Code Validation**: Some repos may use older Manim syntax
- **Deduplication**: Check both exact and semantic similarity
- **ManimGL vs ManimCE**: Always verify the Manim version used (exclude ManimGL repos)