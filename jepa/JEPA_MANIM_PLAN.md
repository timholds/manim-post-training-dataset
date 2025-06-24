# JEPA-Manim: Layout-Aware Code Generation Plan

## Executive Summary

This document outlines our plan to use JEPA (Joint-Embedding Predictive Architecture) to solve Manim's fundamental layout problem by learning spatial relationships from 3Blue1Brown videos and generating layout-aware code.

## Project Overview

### Core Innovation
JEPA operates in latent representation space to understand spatial relationships. For Manim, this enables:
- Understanding how mathematical objects relate spatially
- Predicting optimal positions for scene elements
- Generating code that respects visual constraints

### Key Advantages Over Current Approaches
1. **Spatial Understanding**: JEPA learns implicit layout rules rather than generating code sequentially
2. **Small Dataset Viability**: Self-supervised learning with aggressive augmentation
3. **Layout-First Generation**: Positions determined before code generation

## Dataset Preparation

### Available Data
- **3b1b Dataset**: ~500 scenes from 50 videos (10 scenes/video average)
- **Code Structure**: ManimGL code with successful import inlining (75% success rate)
- **Transcripts**: YouTube transcripts for all videos
- **Metadata**: Video URLs, timestamps, scene boundaries

### Data Processing Pipeline

#### Phase 1: Scene Extraction (Week 1)
```python
# Pipeline structure
Video → Frame Extraction → Scene Detection → Code Alignment → Training Pairs
```

1. **Frame Extraction**
   - Extract keyframes at scene boundaries
   - Target: 1920x1080 resolution
   - Format: PNG for lossless quality

2. **Scene-Code Alignment**
   - Use existing code files from 3b1b_dataset
   - Map scenes to code using timestamps
   - Validate rendering matches video frames

3. **Augmentation Strategy**
   - **Visual**: Resolution scaling, color jittering
   - **Spatial**: Object position perturbations
   - **Code**: Variable naming, style variations
   - **Temporal**: Different frame selections per scene

### Expected Dataset Size
- Base: 500 scenes
- With augmentation: 10,000-25,000 training samples
- Validation set: 50 scenes (held out)

## Technical Architecture

### Simplified Hybrid Approach

```python
class JEPAManimGenerator:
    def __init__(self):
        # Vision understanding
        self.vision_encoder = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        # Layout prediction
        self.layout_predictor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, max_objects * 4)  # x, y, z, type
        )
        
        # Code generation
        self.code_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-3B"
        )
```

### Training Strategy

#### Stage 1: Layout Understanding (Weeks 1-4)
- **Goal**: Predict object positions from partial scene information
- **Input**: CLIP-encoded scene with masking
- **Output**: Bounding boxes and object types
- **Loss**: MSE for positions + CE for object classification

#### Stage 2: Code Generation (Weeks 5-8)
- **Goal**: Generate Manim code with position constraints
- **Input**: Layout predictions + scene description
- **Output**: Executable Manim code
- **Loss**: Next-token prediction + position adherence

## Implementation Phases

### Phase 1: Proof of Concept (2 Weeks)

#### Week 1: Synthetic Data Generation
```python
def generate_synthetic_scene():
    """Create simple scenes with known layouts"""
    scene = Scene()
    
    # Random but constrained positions
    equation_pos = (0, random.uniform(1, 3), 0)
    graph_pos = (0, random.uniform(-3, -1), 0)
    
    equation = MathTex("f(x) = x^2").move_to(equation_pos)
    graph = FunctionGraph(lambda x: x**2).move_to(graph_pos)
    
    scene.add(equation, graph)
    return scene, {"equation": equation_pos, "graph": graph_pos}
```

#### Week 2: Basic Layout Prediction
- Train CLIP → position predictor
- Test on synthetic scenes
- **Success Metric**: 70% IoU on object positions

### Phase 2: Real Data Integration (4 Weeks)

#### Weeks 3-4: 3b1b Data Processing
1. Extract 100 scenes from existing dataset
2. Generate frame-code alignments
3. Create position annotations
4. Validate with rendering

#### Weeks 5-6: Full Pipeline
1. Train on real 3b1b scenes
2. Add code generation component
3. Implement position-constrained generation
4. **Success Metric**: 80% compilable code with correct positions

### Phase 3: Scale and Optimize (4 Weeks)

#### Weeks 7-8: Full Dataset Training
- Process all 500 scenes
- Implement advanced augmentations
- Fine-tune larger models if needed

#### Weeks 9-10: Production Pipeline
- API development
- Optimization for inference
- Documentation and testing

## Evaluation Metrics

### Quantitative Metrics
1. **Layout Accuracy**
   - IoU for predicted vs actual positions
   - Object type classification accuracy
   - Relative position correctness

2. **Code Quality**
   - Compilation success rate
   - Runtime error frequency
   - Position adherence (within 0.5 units)

3. **Visual Fidelity**
   - LPIPS score vs original frames
   - Human evaluation (1-5 scale)

### Falsifiable Milestones

#### Milestone 1 (Week 2)
- **Success**: 70% IoU on synthetic scenes
- **Failure Action**: Revise architecture or increase model capacity

#### Milestone 2 (Week 6)
- **Success**: 80% compilable code on real scenes
- **Failure Action**: Simplify to template-based generation

#### Milestone 3 (Week 10)
- **Success**: 60% scenes rated "good" by human evaluation
- **Failure Action**: Focus on specific scene types

## Resource Requirements

### Compute
- **Development**: 1x RTX 4090 or A6000
- **Training**: 200-400 GPU hours total
- **Inference**: <5 seconds per scene

### Storage
- Raw videos: ~50GB
- Processed frames: ~100GB
- Models and checkpoints: ~20GB

### Team
- 1-2 developers
- Part-time Manim expert for validation

## Risk Mitigation

### Technical Risks
1. **Frame-Code Alignment Complexity**
   - Mitigation: Start with synthetic data
   - Fallback: Manual annotation of 100 scenes

2. **Insufficient Data**
   - Mitigation: Aggressive augmentation
   - Fallback: Generate synthetic variations

3. **Model Convergence Issues**
   - Mitigation: Pre-trained vision models
   - Fallback: Simpler architectures

### Implementation Risks
1. **Scope Creep**
   - Mitigation: Strict phase boundaries
   - Fallback: Focus on layout only

2. **Integration Complexity**
   - Mitigation: Modular design
   - Fallback: Standalone components

## Success Criteria

### Short Term (10 weeks)
- Generate layout-aware Manim code for basic scenes
- 80% compilation success rate
- Correct object positioning in 70% of cases

### Long Term (6 months)
- Handle complex multi-object scenes
- Support animations and transitions
- Production-ready API

## Next Steps

1. **Immediate Actions**
   - Set up development environment
   - Begin synthetic data generation
   - Implement basic CLIP encoder pipeline

2. **Week 1 Deliverables**
   - 100 synthetic scenes generated
   - Basic training pipeline operational
   - Initial position prediction results

3. **Decision Points**
   - Week 2: Continue with approach or pivot
   - Week 6: Scale up or simplify
   - Week 10: Production development or research continuation

## Appendix: Code Organization

```
manim-post-training/
├── JEPA_MANIM_PLAN.md (this file)
├── data/
│   ├── synthetic/       # Generated scenes
│   ├── 3b1b_processed/  # Extracted from 3b1b_dataset
│   └── augmented/       # Augmented training data
├── models/
│   ├── layout_predictor/
│   ├── code_generator/
│   └── checkpoints/
├── src/
│   ├── data_generation.py
│   ├── jepa_model.py
│   ├── training.py
│   └── inference.py
└── evaluation/
    ├── metrics.py
    └── human_eval/
```

---

*Document Version: 1.0*  
*Last Updated: 2025-06-22*  
*Status: Ready for Implementation*