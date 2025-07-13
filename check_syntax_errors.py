#!/usr/bin/env python3
"""Check actual syntax error rates in the dataset."""

import json
from pathlib import Path
import ast

# Check both train and test files
for filename in ['train.json', 'test.json']:
    file_path = Path('data_formatted_v2') / filename
    if not file_path.exists():
        continue
    
    print(f'\nAnalyzing {filename}:')
    
    source_counts = {}
    syntax_errors_by_source = {}
    total_syntax_errors = 0
    
    with open(file_path) as f:
        for i, line in enumerate(f):
            try:
                sample = json.loads(line)
                source = sample.get('source', 'unknown')
                source_counts[source] = source_counts.get(source, 0) + 1
                
                # Check for syntax errors
                if 'conversations' in sample and len(sample['conversations']) >= 3:
                    code = sample['conversations'][2].get('value', '')
                    if '```python' in code:
                        code = code.split('```python')[1].split('```')[0].strip()
                    elif '```' in code:
                        code = code.split('```')[1].split('```')[0].strip()
                    
                    try:
                        ast.parse(code)
                    except SyntaxError as e:
                        syntax_errors_by_source[source] = syntax_errors_by_source.get(source, 0) + 1
                        total_syntax_errors += 1
                        if total_syntax_errors <= 3:
                            print(f'  Example syntax error in {source} (line {i}): {str(e)[:80]}')
                        
            except Exception as e:
                pass
    
    print(f'\n  Total syntax errors: {total_syntax_errors}')
    print('\n  Source breakdown:')
    for source, count in sorted(source_counts.items()):
        errors = syntax_errors_by_source.get(source, 0)
        error_rate = errors / count * 100 if count > 0 else 0
        print(f'    {source}: {count} samples, {errors} syntax errors ({error_rate:.1f}%)')
    
    print(f'\n  Total samples: {sum(source_counts.values())}')