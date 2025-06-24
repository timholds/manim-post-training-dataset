# JEPA-Manim Technical Architecture

## System Overview

The JEPA-Manim system combines visual understanding, layout prediction, and code generation to create spatially-aware Manim animations.

```
Scene Description → Visual Encoder → Layout Predictor → Code Generator → Manim Code
                         ↓                    ↓                ↓
                   CLIP Features      Object Positions   Constrained Generation
```

## Core Components

### 1. Visual Encoder (CLIP-based)

```python
class VisualEncoder(nn.Module):
    def __init__(self, freeze_clip=True):
        super().__init__()
        # Use pre-trained CLIP for visual understanding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
        if freeze_clip:
            for param in self.clip_model.parameters():
                param.requires_grad = False
        
        # Projection head for Manim-specific features
        self.projection = nn.Sequential(
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 768)
        )
        
    def forward(self, images):
        # Extract CLIP features
        clip_features = self.clip_model.get_image_features(images)
        
        # Project to Manim space
        manim_features = self.projection(clip_features)
        
        return manim_features
```

### 2. Layout Predictor (JEPA-inspired)

```python
class JEPALayoutPredictor(nn.Module):
    def __init__(self, feature_dim=768, max_objects=10):
        super().__init__()
        self.max_objects = max_objects
        
        # Context encoder
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=feature_dim,
                nhead=8,
                dim_feedforward=2048,
                dropout=0.1
            ),
            num_layers=6
        )
        
        # Target predictor
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, max_objects * 7)  # x,y,z,w,h,type,confidence
        )
        
        # Object type embeddings
        self.type_embeddings = nn.Embedding(20, feature_dim)  # 20 object types
        
    def forward(self, visual_features, mask=None):
        # Encode context (visible parts)
        if mask is not None:
            context = visual_features * mask
        else:
            context = visual_features
            
        encoded = self.context_encoder(context.unsqueeze(0))
        
        # Predict target layout
        predictions = self.predictor(encoded.squeeze(0))
        predictions = predictions.reshape(self.max_objects, 7)
        
        # Parse predictions
        positions = predictions[:, :3]  # x, y, z
        sizes = predictions[:, 3:5]     # width, height
        types = predictions[:, 5]        # object type
        confidence = predictions[:, 6]   # prediction confidence
        
        return {
            "positions": positions,
            "sizes": sizes,
            "types": types,
            "confidence": confidence
        }
```

### 3. Code Generator with Layout Constraints

```python
class LayoutAwareCodeGenerator(nn.Module):
    def __init__(self, model_name="Qwen/Qwen2.5-Coder-3B"):
        super().__init__()
        
        # Load pre-trained code model
        self.code_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Layout encoder
        self.layout_encoder = nn.Sequential(
            nn.Linear(7 * 10, 512),  # 10 objects × 7 features
            nn.ReLU(),
            nn.Linear(512, 768)
        )
        
        # Cross-attention for layout conditioning
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768,
            num_heads=8,
            dropout=0.1
        )
        
    def forward(self, scene_description, layout_predictions):
        # Encode layout constraints
        layout_flat = layout_predictions.flatten()
        layout_encoding = self.layout_encoder(layout_flat)
        
        # Create constrained prompt
        prompt = self.create_constrained_prompt(
            scene_description, 
            layout_predictions
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # Generate with layout constraints
        with torch.no_grad():
            # Inject layout information via cross-attention
            outputs = self.code_model.generate(
                **inputs,
                max_length=1024,
                do_sample=True,
                temperature=0.7,
                encoder_hidden_states=layout_encoding.unsqueeze(0)
            )
        
        # Decode
        generated_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return generated_code
    
    def create_constrained_prompt(self, description, layout):
        """Generate prompt with explicit position constraints"""
        prompt = f"""Generate Manim code for: {description}
        
        Layout constraints:
        """
        
        for i, obj in enumerate(layout["objects"]):
            if obj["confidence"] > 0.5:
                prompt += f"""
        - {obj['type']} at position ({obj['x']:.2f}, {obj['y']:.2f}, {obj['z']:.2f})
        """
        
        prompt += """
        
        from manim import *
        
        class GeneratedScene(Scene):
            def construct(self):
        """
        
        return prompt
```

### 4. Training Pipeline

```python
class JEPAManimTrainer:
    def __init__(self, config):
        self.config = config
        
        # Initialize models
        self.visual_encoder = VisualEncoder()
        self.layout_predictor = JEPALayoutPredictor()
        self.code_generator = LayoutAwareCodeGenerator()
        
        # Optimizers
        self.optimizer_layout = torch.optim.AdamW(
            self.layout_predictor.parameters(),
            lr=config.lr_layout,
            weight_decay=0.01
        )
        
        self.optimizer_code = torch.optim.AdamW(
            self.code_generator.parameters(),
            lr=config.lr_code,
            weight_decay=0.01
        )
        
        # Loss functions
        self.layout_loss = nn.MSELoss()  # For positions
        self.type_loss = nn.CrossEntropyLoss()  # For object types
        
    def train_step(self, batch):
        # Stage 1: Layout prediction
        visual_features = self.visual_encoder(batch["frames"])
        layout_pred = self.layout_predictor(
            visual_features, 
            mask=batch["mask"]
        )
        
        # Layout losses
        pos_loss = self.layout_loss(
            layout_pred["positions"], 
            batch["true_positions"]
        )
        type_loss = self.type_loss(
            layout_pred["types"], 
            batch["true_types"]
        )
        
        # Stage 2: Code generation (if layout is good enough)
        if pos_loss < self.config.layout_threshold:
            generated_code = self.code_generator(
                batch["description"],
                layout_pred
            )
            
            # Validate generated code
            code_valid = self.validate_code(generated_code)
            position_adherence = self.check_position_adherence(
                generated_code, 
                layout_pred
            )
            
        return {
            "layout_loss": pos_loss + type_loss,
            "code_valid": code_valid,
            "position_adherence": position_adherence
        }
```

### 5. Inference Pipeline

```python
class JEPAManimInference:
    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models
        self.visual_encoder = VisualEncoder().to(self.device)
        self.layout_predictor = JEPALayoutPredictor().to(self.device)
        self.code_generator = LayoutAwareCodeGenerator().to(self.device)
        
        # Load weights
        checkpoint = torch.load(checkpoint_path)
        self.visual_encoder.load_state_dict(checkpoint["visual_encoder"])
        self.layout_predictor.load_state_dict(checkpoint["layout_predictor"])
        self.code_generator.load_state_dict(checkpoint["code_generator"])
        
        # Set to eval mode
        self.visual_encoder.eval()
        self.layout_predictor.eval()
        self.code_generator.eval()
        
    def generate(self, description, reference_image=None):
        with torch.no_grad():
            # If reference image provided, use it for layout
            if reference_image is not None:
                visual_features = self.visual_encoder(reference_image)
                layout = self.layout_predictor(visual_features)
            else:
                # Generate layout from description
                layout = self.generate_layout_from_text(description)
            
            # Generate code with layout constraints
            code = self.code_generator(description, layout)
            
            # Post-process and validate
            code = self.post_process_code(code)
            
            return {
                "code": code,
                "layout": layout,
                "confidence": self.calculate_confidence(layout)
            }
```

## Data Flow Architecture

### Training Data Flow
```
Raw Video → Frame Extraction → Object Detection → Layout Annotation
                                        ↓
                                  Training Pairs
                                        ↓
                    [Context (masked) | Target (full layout)]
                                        ↓
                                  JEPA Training
```

### Inference Data Flow
```
User Description → Layout Prediction → Code Generation → Validation
         ↓              ↓                    ↓              ↓
   (Optional)     Object Positions    Manim Code     Rendered Scene
Reference Image   & Types
```

## Key Design Decisions

### 1. Modular Architecture
- **Rationale**: Allows independent improvement of each component
- **Benefit**: Can swap out models (e.g., different code LLMs)

### 2. Pre-trained Vision Models
- **Rationale**: Limited training data
- **Benefit**: Leverages general visual understanding

### 3. Layout as Intermediate Representation
- **Rationale**: Bridges vision and code domains
- **Benefit**: Interpretable, debuggable, constrainable

### 4. Self-Supervised JEPA Training
- **Rationale**: Maximize data efficiency
- **Benefit**: Learn from unlabeled video frames

## Optimization Strategies

### 1. Model Quantization
```python
# 8-bit quantization for inference
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.Linear}, 
    dtype=torch.qint8
)
```

### 2. Knowledge Distillation
```python
# Distill large model to smaller one
teacher_model = JEPALayoutPredictor(large=True)
student_model = JEPALayoutPredictor(large=False)

distillation_loss = nn.KLDivLoss()(
    student_output / temperature,
    teacher_output / temperature
)
```

### 3. Caching and Batching
```python
class InferenceCache:
    def __init__(self, max_size=1000):
        self.cache = LRUCache(max_size)
        
    def get_or_compute(self, key, compute_fn):
        if key in self.cache:
            return self.cache[key]
        
        result = compute_fn()
        self.cache[key] = result
        return result
```

## Deployment Architecture

### API Design
```python
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel

app = FastAPI()

class GenerationRequest(BaseModel):
    description: str
    style: str = "3b1b"
    complexity: str = "medium"

class GenerationResponse(BaseModel):
    code: str
    preview_url: str
    layout: dict
    confidence: float

@app.post("/generate", response_model=GenerationResponse)
async def generate_scene(request: GenerationRequest):
    # Generate layout and code
    result = inference_pipeline.generate(
        request.description,
        style=request.style
    )
    
    # Render preview
    preview_url = render_preview(result["code"])
    
    return GenerationResponse(
        code=result["code"],
        preview_url=preview_url,
        layout=result["layout"],
        confidence=result["confidence"]
    )
```

### Scaling Considerations

1. **Horizontal Scaling**: Separate services for encoding, prediction, generation
2. **GPU Optimization**: Batch multiple requests
3. **Caching**: Cache common layouts and code patterns
4. **CDN**: Serve rendered previews via CDN

## Monitoring and Metrics

```python
class MetricsCollector:
    def __init__(self):
        self.metrics = {
            "inference_time": [],
            "layout_accuracy": [],
            "code_compilation_rate": [],
            "user_satisfaction": []
        }
    
    def log_inference(self, duration, layout_score, compiled):
        self.metrics["inference_time"].append(duration)
        self.metrics["layout_accuracy"].append(layout_score)
        self.metrics["code_compilation_rate"].append(int(compiled))
```

---

*Document Version: 1.0*  
*Last Updated: 2025-06-22*  
*Complements: JEPA_MANIM_PLAN.md, DATASET_SPECIFICATION.md*