# YouTube Transcript Strategy for Description Generation

## Current Approach vs Future Enhancement

### Option 1: Store Transcripts Now ❌
```python
def get_youtube_transcript(video_id):
    transcript = fetch_from_youtube_api(video_id)
    return transcript

# Store in dataset
sample["metadata"]["youtube_transcript"] = transcript
```

**Problems:**
- Increases dataset size significantly
- Transcripts might not be used if good descriptions exist
- Storage overhead for potentially unused data

### Option 2: Fetch Transcripts During LLM Processing ✅
```bash
# When generating descriptions
claude -p "
Analyze this Manim code and YouTube transcript to generate a natural user request.

Code: [code here]
Transcript: $(youtube-dl --get-transcript VIDEO_ID)

Generate a request that captures what the animation shows visually.
"
```

**Benefits:**
- Only fetch when needed
- Always get latest transcript
- Can combine with code analysis in one LLM call
- No storage overhead

## Recommended Implementation

### 1. Add Placeholder Descriptions During Extraction
For sources that need description generation, use placeholder text:
```python
# Example from extractors/sources/manim_ce.py
description = f"[PLACEHOLDER - Needs description] Create an animation for: {class_name} ({category})"
```

This allows:
- Easy identification of samples needing descriptions
- Batch processing with LLMs later
- Validation that descriptions were generated

### 2. Add Video Metadata During Extraction
```python
# In extract_dan4life_enhanced.py
metadata = {
    "youtube_url": f"https://youtube.com/@dan4life/day{day}",  # If known
    "has_video": True,
    "transcript_available": None,  # Check later
    "needs_description": True,  # Track placeholder status
}
```

### 3. Fetch Transcript When Generating Descriptions
```python
def generate_description_with_transcript(code, metadata):
    if metadata.get("youtube_url"):
        # Fetch transcript on-demand
        transcript = fetch_youtube_transcript(metadata["youtube_url"])
        
        prompt = f"""
        Analyze this Manim animation code and video transcript.
        
        Code shows: {analyze_code_features(code)}
        
        Transcript excerpt: {transcript[:1000]}
        
        Generate a natural user request for this animation.
        """
    else:
        # Fall back to code-only analysis
        prompt = generate_code_only_prompt(code)
    
    return llm_call(prompt)
```

### 4. Cache Transcript + Code → Description Mapping
```python
# Enhanced cache key includes transcript
cache_key = hash(code + transcript + metadata)
```

## Benefits of On-Demand Approach

1. **Efficiency**: Only process transcripts for datasets that need them
2. **Flexibility**: Can try with/without transcripts
3. **Freshness**: Always get current transcript
4. **Experimentation**: Easy to A/B test transcript impact

## Placeholder Processing

To find and process all samples with placeholder descriptions:

```python
# Find all samples needing descriptions
samples_needing_descriptions = []
for sample in dataset:
    if "[PLACEHOLDER" in sample["description"] or sample.get("metadata", {}).get("needs_description"):
        samples_needing_descriptions.append(sample)

# Batch process with LLM
for batch in chunks(samples_needing_descriptions, batch_size=10):
    descriptions = generate_descriptions_batch(batch)
    update_dataset(batch, descriptions)
```

## Example Transcript Usage

```bash
# For Dan4Life Day 22
youtube-dl --get-transcript "https://youtube.com/watch?v=..."

# Excerpt: "Today we're implementing a pseudorandom number generator
# using XOR operations. First, I'll show how binary XOR works..."

# Combined with code analysis:
# - Code has XOR operations, binary representations
# - Transcript mentions "show how binary XOR works"
# → Generate: "Create an animation that demonstrates how XOR 
#   operations work with binary numbers..."
```

## Future Extensions

1. **Multi-source Context**: Code + Transcript + README
2. **Quality Scoring**: Rate descriptions with/without transcripts
3. **Selective Fetching**: Only get transcripts for complex animations
4. **Transcript Caching**: Cache transcripts separately from descriptions

## Conclusion

Using `claude -p` or similar tools to fetch and analyze transcripts on-demand during description generation is more flexible and efficient than storing transcripts in the dataset. This approach allows for experimentation and reduces storage requirements while maintaining the ability to leverage video context when beneficial.