## Overview
Our overall goal is to construct a perfect manim fine-tuning dataset from several sources


## Development Environment Setup

- Run source manim-env/bin/activate and use uv pip to install packages

## Development Style
- When creating diagnostic or testing scripts, it's better and cleaner to just run the python command directly instead of creating a new file and clogging up the repo. 

- After we have accomplished our goals, we should update the PROJECT_PLAN.md file 


## Analyzing Large Datasets

When you need to analyze entire datasets, validate quality across thousands of samples, or debug deduplication issues that exceed your context limits:
- Use `gemini -p` with instructions from GEMINI.md
- This is especially useful for:
  - Analyzing the full training dataset (16,000+ samples)
  - Comparing datasets before/after deduplication
  - Validating code quality across all sources
  - Finding patterns in removed duplicates
  - Assessing dataset diversity and coverage

See GEMINI.md for specific commands and examples.

