#!/usr/bin/env python3
"""Trace the exact data flow to show where formatting diverges."""

import json
import pandas as pd

def show_data_transformation():
    print("=== TRACING DATA FLOW ===\n")
    
    # 1. Show raw data
    print("STEP 1: RAW DATA FROM SOURCES")
    print("-" * 60)
    
    # Manimbench raw
    df_manim = pd.read_parquet('/Users/timholdsworth/.cache/manim_datasets/ravidussilva_manim-sft_manim_sft_dataset.parquet')
    manim_sample = df_manim.iloc[0]
    print("MANIMBENCH Raw Code:")
    print(repr(manim_sample['Code'][:150]))
    
    # Thanks dataset raw  
    df_thanks = pd.read_parquet('/Users/timholdsworth/.cache/manim_datasets/thanhkt_manim_code_train.parquet')
    thanks_sample = df_thanks.iloc[0]
    print("\nTHANKS_DATASET Raw Output:")
    print(repr(thanks_sample['output'][:150]))
    
    # 2. Show after process_dataset cleaning (line 276)
    print("\n\nSTEP 2: AFTER process_dataset() CLEANING (line 276: code.strip())")
    print("-" * 60)
    
    # Simulate manimbench cleaning
    manim_code = manim_sample['Code']
    manim_code = manim_code.strip()  # Line 276
    print("MANIMBENCH after strip():")
    print(repr(manim_code[:150]))
    
    # Simulate thanks cleaning
    thanks_code = thanks_sample['output']
    thanks_code = thanks_code.strip()  # Line 276 - This is where \\n should be removed!
    print("\nTHANKS_DATASET after strip():")
    print(repr(thanks_code[:150]))
    print("NOTE: strip() doesn't remove the \\n because it's part of the string content, not whitespace!")
    
    # 3. Show after ensure_proper_code_format (line 129)
    print("\n\nSTEP 3: AFTER ensure_proper_code_format() (line 129)")
    print("-" * 60)
    
    # Both should pass through unchanged since they have proper structure
    print("Both samples already have 'from manim import' and proper structure,")
    print("so ensure_proper_code_format() returns them unchanged.")
    
    # 4. Show final formatting (line 132)
    print("\n\nSTEP 4: FINAL FORMATTING (line 132)")
    print("-" * 60)
    print("Code: assistant_response = f\"```python\\n{formatted_code}\\n```\"")
    
    # Simulate final formatting
    manim_final = f"```python\n{manim_code}\n```"
    thanks_final = f"```python\n{thanks_code}\n```"
    
    print("\nMANIMBENCH final:")
    print(repr(manim_final[:150]))
    
    print("\nTHANKS_DATASET final:")
    print(repr(thanks_final[:150]))
    print("NOTE: This creates ```python\\n\\n from manim... (double newline!)")
    
    # 5. Show the exact problem
    print("\n\n=== THE EXACT PROBLEM ===")
    print("-" * 60)
    print("thanks_dataset has '\\n ' at the start of their code in the raw data.")
    print("When we do f\"```python\\n{code}\\n```\", we get:")
    print("  ```python\\n\\n from manim...")
    print("\nWhen this is saved to JSON, the \\n inside the string gets escaped to \\\\n")
    print("So the final JSON has: ```python\\\\n\\\\n from manim...")
    
    # 6. Where to fix
    print("\n\n=== WHERE TO FIX ===")
    print("-" * 60)
    print("OPTION 1: In process_dataset() after line 276:")
    print("    code = code.strip()")
    print("    # Add this:")
    print("    if code.startswith('\\\\n'):")
    print("        code = code[2:].lstrip()")
    print("")
    print("OPTION 2: In create_conversation() before line 132:")
    print("    # Strip any leading newlines from formatted_code")
    print("    formatted_code = formatted_code.lstrip('\\n')")
    print("    assistant_response = f\"```python\\n{formatted_code}\\n```\"")

if __name__ == "__main__":
    show_data_transformation()