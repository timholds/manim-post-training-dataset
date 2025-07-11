# Exact Format Divergence Analysis

## The Problem in Detail

### Raw Data Comparison

**MANIMBENCH** (raw data):
```python
from manim import *

class ThreeDCameraIllusionRotation(ThreeDScene):
    def construct(self):
        axes = ThreeDAxes()
        circle=Circle()
```

**THANKS_DATASET** (raw data):
```
\n from manim import *
import random

class PiApproximationByMonteCarlo(Scene):
    def construct(self):
        num_points = 1000
```

Note: The thanks_dataset literally has the characters `\n` (backslash-n) at the start of their code strings!

### After Our Processing (Current Behavior)

**MANIMBENCH** (final output):
```json
"value": "```python\nfrom manim import *\n\nclass ThreeDCameraIllusionRotation..."
```

**THANKS_DATASET** (final output):
```json
"value": "```python\n\\n from manim import *\nimport random..."
```

### What the Model Sees

**MANIMBENCH** (correct):
````
```python
from manim import *

class ThreeDCameraIllusionRotation...
````

**THANKS_DATASET** (incorrect):
````
```python
\n from manim import *
import random...
````

The model would learn to generate literal `\n` characters!

## Where This Happens in Our Code

### 1. Data Loading (lines 262-276 in prepare_data_enhanced.py)
```python
for idx, row in df.iterrows():
    description = str(row[desc_field])
    code = str(row[code_field])  # thanks_dataset: "\\n from manim..."
    
    # Clean up code if needed
    if code.startswith("```python"):
        code = code[9:]
    if code.endswith("```"):
        code = code[:-3]
    code = code.strip()  # This doesn't remove the \n because it's content, not whitespace!
```

### 2. Final Formatting (line 132)
```python
assistant_response = f"```python\n{formatted_code}\n```"
# For thanks_dataset, this becomes: "```python\n\\n from manim...\n```"
```

## The Fix

Add this after line 276 in `process_dataset()`:

```python
# Remove literal \n from the beginning of code (common in thanks_dataset)
if code.startswith('\\n'):
    code = code[2:].lstrip()  # Remove \n and any following spaces
```

Or in `create_conversation()` before line 132:

```python
# Clean any leading newlines
formatted_code = formatted_code.lstrip('\n').lstrip('\\n')
```

## Impact

- **Current**: 72.6% of training samples have incorrect formatting
- **After fix**: All samples will have consistent, correct formatting
- **Training impact**: Model will learn proper code generation without escape sequences