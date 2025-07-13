# Pipeline Architecture: Decoupled Data Processing

## Overview

Our Manim dataset preparation uses a **decoupled pipeline architecture** where each stage operates independently on standardized file formats. This design allows maximum flexibility and reusability.

## Architecture Diagram

```
┌─────────────────┐     JSONL      ┌──────────────────┐     JSONL      ┌─────────────────┐
│                 │ ─────────────> │                  │ ─────────────> │                 │
│ prepare_data.py │                │ prepare_data_    │                │   Training      │
│   (Extract)     │                │   with_llm.py    │                │   Pipeline      │
│                 │                │ (Enhance)        │                │                 │
└─────────────────┘                └──────────────────┘                └─────────────────┘
        ↑                                   ↑                                    ↑
        │                                   │                                    │
   Extractors                         LLM Generator                      Can read any
   (Plugins)                          (w/ Cache)                         JSONL file
```

## Key Design Principles

### 1. **File-Based Communication**
Each stage communicates through JSONL files, not function calls:
- **Benefit**: Stages can run independently, even on different machines
- **Benefit**: Intermediate results are inspectable and debuggable
- **Benefit**: Can restart pipeline from any stage

### 2. **Standardized Format**
All stages expect/produce the same JSONL format:
```json
{
  "description": "Create an animation that...",
  "code": "from manim import *\n...",
  "source": "dataset_name",
  "metadata": {
    "needs_description": true,
    "youtube_url": "...",
    "code_features": {...}
  }
}
```

### 3. **No Coupling Between Stages**
- `prepare_data.py` doesn't know about LLM generation
- `prepare_data_with_llm.py` doesn't know about extractors
- Training pipeline doesn't know about either

## Stage 1: Data Extraction (`prepare_data.py`)

**Purpose**: Extract raw data from various sources
**Input**: Source repositories, APIs, files
**Output**: `data_formatted/train.json`, `data_formatted/test.json`

```bash
python prepare_data.py prepare --sources dan4life_aoc2024 szymon_ozog
```

**Key Features**:
- Plugin-based extractors
- Deduplication
- Train/test splitting
- Basic augmentation

## Stage 2: LLM Enhancement (`prepare_data_with_llm.py`)

**Purpose**: Enhance descriptions using LLM
**Input**: Any JSONL file from Stage 1
**Output**: Enhanced JSONL file

```bash
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_enhanced/train.json
```

**Key Features**:
- Reads ANY JSONL file (not coupled to prepare_data.py)
- SHA256-based caching
- Batch processing
- Multiple LLM backends

## Stage 3: Training (Your Training Pipeline)

**Purpose**: Train models
**Input**: Any JSONL file
**Output**: Trained model

The training pipeline can read output from:
- Stage 1 directly (no LLM descriptions)
- Stage 2 (with LLM descriptions)
- Any other JSONL source

## Practical Examples

### Example 1: Standard Flow
```bash
# Extract
python prepare_data.py prepare --sources all

# Enhance with LLM
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_enhanced/train.json

# Train
python train.py --data data_enhanced/train.json
```

### Example 2: Re-run LLM Only
```bash
# Descriptions weren't good, try different prompt
python prepare_data_with_llm.py generate-descriptions \
    --input data_formatted/train.json \
    --output data_enhanced_v2/train.json \
    --llm claude  # Try different model
```

### Example 3: Skip LLM
```bash
# Extract
python prepare_data.py prepare --sources manimbench

# Train directly (ManimBench has good descriptions already)
python train.py --data data_formatted/train.json
```

### Example 4: Process External Data
```bash
# Someone gives you a JSONL file
python prepare_data_with_llm.py generate-descriptions \
    --input external_data.jsonl \
    --output external_enhanced.jsonl
```

## Benefits of Decoupling

1. **Debugging**: Can inspect intermediate files
   ```bash
   # See what needs descriptions
   jq 'select(.metadata.needs_description == true) | .source' data_formatted/train.json | sort | uniq -c
   ```

2. **Caching**: LLM cache persists across runs
   ```bash
   python prepare_data_with_llm.py cache-stats
   ```

3. **Flexibility**: Mix and match stages
   ```bash
   # Use Stage 1 from yesterday, but re-run Stage 2
   python prepare_data_with_llm.py generate-descriptions \
       --input yesterday/train.json \
       --output today/train.json
   ```

4. **Parallel Development**: Teams can work on different stages
   - Extractor team improves `prepare_data.py`
   - LLM team improves `prepare_data_with_llm.py`
   - No coordination needed!

## When Coupling Might Be Needed

You might want to couple the stages if:
- Real-time description generation is required
- Memory constraints prevent writing intermediate files
- You need streaming processing

But for dataset preparation, the decoupled approach is usually better.

## Adding New Stages

The pipeline is extensible. For example, adding a quality filter:

```bash
# New stage: quality filtering
python filter_quality.py \
    --input data_enhanced/train.json \
    --output data_filtered/train.json \
    --min-quality 0.8
```

As long as it reads/writes the standard JSONL format, it fits right in!

## Summary

The decoupled architecture means:
- ✅ Update `prepare_data.py` without touching LLM code
- ✅ Update LLM generation without touching extractors  
- ✅ Debug by inspecting intermediate files
- ✅ Restart from any stage
- ✅ Process data from any source

This is why you don't need to update `prepare_data_with_llm.py` when you change `prepare_data.py` - they communicate through files, not code!