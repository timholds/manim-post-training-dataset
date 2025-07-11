# Tokenizer Configurations for Popular Code Models

## Quick Reference Table

| Model | Has Padding Token | Common Padding Solution | EOS Token | Chat Format |
|-------|------------------|------------------------|-----------|-------------|
| **Qwen2.5-Coder** | ❌ No | `pad_token = eos_token` | `<\|endoftext\|>` or `<\|im_end\|>` | ChatML (`<\|im_start\|>`, `<\|im_end\|>`) |
| **CodeLlama** | ❌ No | `pad_token = eos_token` or add `[PAD]` | `</s>` | Not standardized |
| **DeepSeek-Coder-v2** | ✅ Yes | Has `<pad>` token | `<｜end▁of▁sentence｜>` | Custom with thinking tokens |
| **CodeGemma** | ✅ Yes | Has `<pad>` but often set to `eos_token` | `<eos>` | `<start_of_turn>`, `<end_of_turn>` |
| **Stable-Code** | ❌ No | `pad_token = eos_token` | Not documented | Not standardized |

## Model-Specific Setup Code

### 1. Qwen2.5-Coder
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
# No padding token by default
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Chat template
messages = [
    {"role": "system", "content": "You are a code assistant"},
    {"role": "user", "content": "Write hello world"}
]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# Or manual: "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n"
```

### 2. CodeLlama
```python
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-Instruct-hf")

# Option 1: Use eos as pad
tokenizer.pad_token = tokenizer.eos_token

# Option 2: Use unk as pad (if available)
if tokenizer.unk_token:
    tokenizer.pad_token = tokenizer.unk_token

# Option 3: Add custom pad token
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
```

### 3. DeepSeek-Coder-v2
```python
tokenizer = AutoTokenizer.from_pretrained(
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    trust_remote_code=True  # Required!
)
# Already has padding token, but verify:
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
```

### 4. CodeGemma
```python
tokenizer = AutoTokenizer.from_pretrained("google/codegemma-7b-it")
# Has pad token but common to override
tokenizer.pad_token = tokenizer.eos_token  # Optional

# Chat template
messages = [{"role": "user", "content": "Write code"}]
text = tokenizer.apply_chat_template(messages, tokenize=False)
# Format: "<bos><start_of_turn>user\n...<end_of_turn>\n<start_of_turn>model\n"
```

### 5. Stable-Code
```python
tokenizer = AutoTokenizer.from_pretrained(
    "stabilityai/stable-code-3b",
    use_auth_token=True  # Requires HF login
)
tokenizer.pad_token = tokenizer.eos_token
```

## Best Practices for Fine-Tuning

1. **Always set padding token explicitly**:
   ```python
   if not tokenizer.pad_token:
       tokenizer.pad_token = tokenizer.eos_token
   ```

2. **Log token configuration**:
   ```python
   print(f"Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
   print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
   ```

3. **Use DataCollatorForLanguageModeling** for proper attention masks:
   ```python
   from transformers import DataCollatorForLanguageModeling
   
   data_collator = DataCollatorForLanguageModeling(
       tokenizer=tokenizer,
       mlm=False,  # Causal LM
       pad_to_multiple_of=8  # Efficiency
   )
   ```

4. **Handle special tokens for code**:
   - Qwen: FIM tokens for fill-in-middle tasks
   - CodeLlama: Infilling tokens (`<PRE>`, `<MID>`, `<SUF>`)
   - DeepSeek: Thinking tokens for reasoning
   - CodeGemma: Standard chat format
   - Stable-Code: FIM tokens

## Common Issues and Solutions

### Issue: "The attention mask is not set"
**Solution**: Use DataCollatorForLanguageModeling or manually create attention masks

### Issue: Model never generates EOS token
**Solution**: Don't use eos_token as pad_token if possible, or ensure proper attention masking

### Issue: Tokenizer warnings about padding
**Solution**: Explicitly set pad_token before using tokenizer

### Issue: Chat template not working
**Solution**: Use model-specific format or tokenizer.apply_chat_template()