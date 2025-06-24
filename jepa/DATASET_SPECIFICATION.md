# JEPA-Manim Dataset Specification

## Overview

This document specifies how to leverage the existing 3b1b_dataset infrastructure to create training data for JEPA-Manim layout understanding.

## Current 3b1b Dataset Status

### Available Resources
- **Years Covered**: 2015-2024
- **Import Inlining**: 75% success rate (18/24 files in 2016)
- **Structure**: Single combined Python file per video
- **Transcripts**: YouTube transcripts for all videos
- **Metadata**: Video URLs, timestamps, match confidence scores

### Quality Metrics
- **High Confidence Matches**: ~70% of videos
- **Code Availability**: ~85% of videos have associated code
- **Transcript Coverage**: 100% via YouTube API

## Dataset Requirements for JEPA Training

### Core Data Types Needed

#### 1. Visual Data
```python
{
    "frame_id": "video_id_timestamp",
    "image": "path/to/frame.png",
    "resolution": [1920, 1080],
    "timestamp": 123.45,
    "scene_index": 5
}
```

#### 2. Layout Annotations
```python
{
    "frame_id": "video_id_timestamp",
    "objects": [
        {
            "type": "MathTex",
            "content": "f(x) = x^2",
            "bbox": [x1, y1, x2, y2],
            "position": [0, 2.5, 0],
            "z_index": 1
        },
        {
            "type": "Graph",
            "content": "FunctionGraph",
            "bbox": [x1, y1, x2, y2],
            "position": [0, -1, 0],
            "z_index": 0
        }
    ]
}
```

#### 3. Code Mapping
```python
{
    "frame_id": "video_id_timestamp",
    "code_file": "path/to/code.py",
    "scene_class": "IntroScene",
    "relevant_lines": [45, 67],  # Lines that create visible objects
    "construct_method": "def construct(self):..."
}
```

## Data Processing Pipeline

### Stage 1: Frame Extraction

```bash
# Using existing video URLs from 3b1b_dataset
python extract_frames.py \
    --input_dir /Users/timholdsworth/code/3b1b_dataset/data/videos \
    --output_dir ./data/frames \
    --fps 1  # Extract 1 frame per second initially
```

### Stage 2: Scene Detection

```python
def detect_scene_boundaries(video_path):
    """
    Use both visual and code analysis to find scenes
    """
    # Visual detection
    scenes_visual = detect_cuts(video_path)
    
    # Code detection
    code_file = find_matching_code(video_path)
    scenes_code = extract_scene_classes(code_file)
    
    # Align and merge
    return align_scenes(scenes_visual, scenes_code)
```

### Stage 3: Object Detection and Layout Extraction

#### Approach A: Automated Detection
```python
class ManimObjectDetector:
    def __init__(self):
        self.text_detector = EasyOCR()
        self.shape_detector = YOLOv8()
        self.manim_classifier = load_pretrained("manim_objects")
    
    def detect_objects(self, frame):
        # Detect text regions (equations, labels)
        text_regions = self.text_detector.detect(frame)
        
        # Detect geometric shapes
        shapes = self.shape_detector.detect(frame)
        
        # Classify as Manim objects
        objects = self.manim_classifier.classify(text_regions + shapes)
        
        return objects
```

#### Approach B: Code-Guided Extraction
```python
def extract_layout_from_code(code_file, timestamp):
    """
    Parse code to understand what objects should be visible
    """
    # Parse AST
    tree = ast.parse(open(code_file).read())
    
    # Find construct method
    construct = find_construct_method(tree)
    
    # Extract object creations and positions
    objects = []
    for node in ast.walk(construct):
        if is_manim_object(node):
            obj = {
                "type": get_object_type(node),
                "position": extract_position(node),
                "content": extract_content(node)
            }
            objects.append(obj)
    
    return objects
```

### Stage 4: Training Data Generation

```python
class JEPADataGenerator:
    def __init__(self, base_scenes):
        self.scenes = base_scenes
        self.augmentations = [
            PositionJitter(std=0.1),
            ColorAugmentation(),
            ScaleVariation(0.8, 1.2),
            MaskingStrategy()
        ]
    
    def generate_training_sample(self):
        # Select random scene
        scene = random.choice(self.scenes)
        
        # Apply augmentations
        augmented = self.apply_augmentations(scene)
        
        # Create JEPA masking
        context, target = self.create_jepa_masks(augmented)
        
        return {
            "context": context,  # Visible parts
            "target": target,    # Parts to predict
            "full_layout": scene.layout  # Ground truth
        }
```

## Leveraging Existing 3b1b_dataset

### 1. Use Existing Code Matching
```python
# The dataset already has matched code files
matched_videos = load_json("3b1b_dataset/output/matching_summary_2015.json")

for video in matched_videos:
    if video["confidence"] == "high":
        code_path = video["code_file"]
        transcript = video["transcript"]
        # Process for JEPA training
```

### 2. Utilize Import Inlining
```python
# Code files already have inlined imports
code_file = "3b1b_dataset/output_v4/2016/dot-products/code.py"
# This file already contains all necessary code
```

### 3. Scene Extraction Strategy
```python
def extract_scenes_from_3b1b(year="2016"):
    video_dir = f"3b1b_dataset/data/videos/_{year}"
    
    scenes = []
    for video_file in glob(f"{video_dir}/*.py"):
        # Parse scene classes
        scenes_in_file = extract_scene_classes(video_file)
        
        # Match with video frames
        for scene_class in scenes_in_file:
            frames = find_matching_frames(scene_class)
            scenes.append({
                "code": scene_class,
                "frames": frames,
                "video_id": extract_video_id(video_file)
            })
    
    return scenes
```

## Data Validation Requirements

### 1. Code Renderability
```python
def validate_scene_renders(scene_code):
    """Ensure code actually produces visual output"""
    try:
        # Create temporary scene
        scene = create_scene_from_code(scene_code)
        
        # Render single frame
        frame = scene.render_frame(0)
        
        # Check has visible objects
        return has_visible_content(frame)
    except:
        return False
```

### 2. Layout Consistency
```python
def validate_layout_annotation(frame, annotation):
    """Ensure annotations match visual content"""
    detected = detect_objects(frame)
    annotated = annotation["objects"]
    
    # Check overlap
    iou = calculate_iou(detected, annotated)
    return iou > 0.7
```

### 3. Position Accuracy
```python
def validate_positions(code_positions, visual_positions):
    """Ensure code positions match visual reality"""
    for obj in code_positions:
        visual_obj = find_matching_object(obj, visual_positions)
        distance = calculate_position_distance(obj, visual_obj)
        if distance > 0.5:  # Manim units
            return False
    return True
```

## Dataset Statistics Target

### Phase 1 (Proof of Concept)
- **Scenes**: 100
- **Frames per scene**: 5-10
- **Total frames**: 500-1000
- **Object types**: 5 (Text, Graph, Equation, Arrow, Shape)

### Phase 2 (Full Dataset)
- **Scenes**: 500
- **Frames per scene**: 10-20
- **Total frames**: 5,000-10,000
- **Object types**: 15+
- **Augmented samples**: 50,000+

## Storage Structure

```
data/
├── raw/
│   ├── videos/          # Original videos
│   ├── frames/          # Extracted frames
│   └── code/            # From 3b1b_dataset
├── processed/
│   ├── annotations/     # Layout annotations
│   ├── jepa_pairs/      # Context-target pairs
│   └── metadata/        # Scene metadata
├── augmented/
│   └── training_samples/  # Ready for training
└── validation/
    ├── test_scenes/     # Held-out scenes
    └── metrics/         # Evaluation results
```

## Quality Assurance

### Automated Checks
1. **Rendering validation**: All code must produce output
2. **Annotation validation**: Bounding boxes must contain objects
3. **Position validation**: Code positions must match visual positions

### Manual Review
1. **Spot check 10%** of annotations
2. **Review edge cases** (complex scenes, multiple objects)
3. **Validate scene boundaries**

## Next Steps

1. **Implement frame extraction** script using video URLs from 3b1b_dataset
2. **Create annotation tool** for manual layout labeling
3. **Build validation pipeline** to ensure data quality
4. **Generate synthetic scenes** to supplement real data

---

*Document Version: 1.0*  
*Last Updated: 2025-06-22*  
*Complements: JEPA_MANIM_PLAN.md*