# Quality Validation: Exactly What It Does

## Overview

Quality validation runs a series of checks on each code/description pair and either ACCEPTS or REJECTS the sample based on the issues found. The behavior depends on the mode.

## The Complete Validation Process

### Step 1: Basic Validation (Always Runs)
Before quality validation even starts, BaseExtractor checks:
```python
if not sample.get("description") or not sample.get("code"):
    return False  # REJECTED
if len(sample["code"]) < 20 or len(sample["description"]) < 5:
    return False  # REJECTED
```

### Step 2: Quality Validation (If Enabled)

The quality validator runs 4 categories of checks:

#### 1. Description Validation (`_validate_description`)

**[HIGH] Issues:**
- Description < 20 characters
- Contains placeholders: `[...]`, `TODO`, `FIXME`, `XXX`, `INSERT`, `PLACEHOLDER`, `<...>`

**[MEDIUM] Issues:**
- Too generic (starts with "Create a Manim animation" etc. AND < 50 chars)

**[LOW] Issues:**
- Unbalanced parentheses
- Doesn't start with capital letter
- Doesn't end with punctuation (. ! ?)

#### 2. Code Structure Validation (`_validate_code_structure`)

**[CRITICAL] Issues:**
- Code < 50 characters
- Syntax errors (code doesn't parse as valid Python)
- No Scene class found (no class inheriting from Scene)
- Empty construct method (just `pass` or `...`)

**[HIGH] Issues:**
- No import statements found
- Scene class exists but missing construct method

#### 3. Code Quality Validation (`_validate_code_quality`)

**[HIGH] Issues:**
- Contains incomplete markers: `TODO`, `FIXME`, `XXX`, `HACK`, `BUG`, `REFACTOR`
- Contains placeholder patterns:
  - `# Your code here`
  - `# Implementation goes here`
  - `# Add your ... here`
  - `# Fill in ...`
  - `# Complete this ...`
  - `...` at end of line
  - `pass # TODO`

**[MEDIUM] Issues:**
- No animation methods found (play, wait, add, remove, move_to, shift, scale, rotate, etc.)
- No mathematical objects found (Text, Circle, Square, Line, Arrow, etc.)

#### 4. Code-Description Alignment (`_validate_code_description_alignment`)

**[MEDIUM] Issues:**
- Description mentions math concepts (derivative, integral, matrix, etc.) but code doesn't implement them
- Class names don't reflect description content

**[LOW] Issues:**
- Minor alignment issues

## Rejection Logic

### Default Mode (quality validation DISABLED in quality_config.json):
```
NO quality checks run - only basic validation
```

### Lenient Mode (when enabled but NOT strict):
```
REJECT if ANY [CRITICAL] issues found
ACCEPT if only [HIGH], [MEDIUM], or [LOW] issues
```

### Strict Mode (`--quality-strict`):
```
REJECT if ANY [CRITICAL] OR [HIGH] issues found
ACCEPT if only [MEDIUM] or [LOW] issues
```

## Real Examples

### Example 1: Would PASS in lenient mode, FAIL in strict mode
```python
# Description: "Draw circle"  # [HIGH] Too short (11 chars)
# Code:
class CircleScene(Scene):
    def construct(self):
        c = Circle()
        self.play(Create(c))
```
- Issue: [HIGH] Description too short
- Lenient: ✅ PASS (no critical issues)
- Strict: ❌ FAIL (has HIGH issue)

### Example 2: Would FAIL in both modes
```python
# Description: "Creates a beautiful animation"
# Code: "print('hello')"  # [CRITICAL] Too short (15 chars)
```
- Issue: [CRITICAL] Code too short
- Lenient: ❌ FAIL
- Strict: ❌ FAIL

### Example 3: Would FAIL in both modes
```python
# Description: "Animate a morphing square"
# Code:
from manim import *

class MorphingSquare(Scene):
    def construct(self):
        pass  # [CRITICAL] Empty construct
```
- Issue: [CRITICAL] Empty construct method
- Lenient: ❌ FAIL
- Strict: ❌ FAIL

### Example 4: Would PASS in both modes
```python
# Description: "Create an animation showing a circle transforming into a square"
# Code:
from manim import *

class CircleToSquare(Scene):
    def construct(self):
        circle = Circle()
        square = Square()
        self.play(Create(circle))
        self.wait()
        self.play(Transform(circle, square))
        self.wait()
```
- No CRITICAL or HIGH issues
- Maybe [LOW] issues (logged but not rejected)
- Lenient: ✅ PASS
- Strict: ✅ PASS

## Key Takeaways

1. **Quality validation is DISABLED by default** (per quality_config.json)
2. **Basic validation ALWAYS runs** (empty/too short checks)
3. **Lenient mode** = Only [CRITICAL] issues cause rejection
4. **Strict mode** = [CRITICAL] OR [HIGH] issues cause rejection
5. **[MEDIUM] and [LOW] issues** = Logged but NEVER cause rejection

## Statistics

After running, the validator provides a report:
```
=== Quality Validation Report ===
Total samples checked: 1000
Passed: 850 (85.0%)
Failed: 150 (15.0%)

Issues by severity:
  [CRITICAL]: 50
  [HIGH]: 120
  [MEDIUM]: 200
  [LOW]: 300
```

This shows you exactly why samples were rejected and helps identify patterns in your data quality issues.