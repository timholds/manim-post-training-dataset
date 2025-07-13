# Using Gemini CLI for Manim Fine-Tuning Dataset Construction

When analyzing large datasets, validating data quality, or debugging deduplication issues that might exceed context limits, use the Gemini CLI with its massive context window. Use `gemini -p` to leverage Google Gemini's large context capacity for comprehensive dataset analysis.

## File and Directory Inclusion Syntax

Use the `@` syntax to include files and directories in your Gemini prompts. The paths should be relative to WHERE you run the gemini command:

### Basic Examples:

**Single file analysis:**
```bash
gemini -p "@prepare_data.py Explain the dataset preparation pipeline and identify potential quality improvements"
```

**Multiple dataset files:**
```bash
gemini -p "@data_formatted_with_sources/train.json @data_formatted_deduplicated/train.json Compare the datasets before and after deduplication. What patterns were removed?"
```

**Entire dataset directory:**
```bash
gemini -p "@data_formatted_deduplicated/ Analyze the overall dataset quality and identify any remaining issues"
```

**Source analysis scripts:**
```bash
gemini -p "@analyze_sources.py @check_duplicates.py How do these analysis tools work together to ensure dataset quality?"
```

**Current directory analysis:**
```bash
gemini -p "@./ Give me an overview of this Manim fine-tuning dataset project structure"
# Or use --all_files flag:
gemini --all_files -p "Analyze the project structure and identify the main data flow"
```

## Dataset Quality Analysis Examples

### Code Quality Validation
**Check Manim code correctness:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Analyze the Manim code quality. Are there syntax errors, missing imports, or incorrect Scene class usage?"
```

**Validate import patterns:**
```bash
gemini -p "@data_formatted_with_sources/train.json What import patterns are most common? Are there any non-standard or potentially problematic imports?"
```

**Check for code complexity distribution:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Analyze the complexity distribution of Manim animations. What percentage are simple vs complex scenes?"
```

### Deduplication Analysis
**Understand deduplication impact:**
```bash
gemini -p "@deduplication_report.json @removed_duplicates.json Analyze the deduplication results. Which datasets had the most overlap and what types of duplicates were most common?"
```

**Validate deduplication quality:**
```bash
gemini -p "@data_formatted_deduplicated/train.json @removed_duplicates.json Did the deduplication process remove any false positives? Show examples of edge cases"
```

**Cross-source duplicate patterns:**
```bash
gemini -p "@trace_duplicates.py @analyze_duplicate_quality.py What duplicate patterns exist across different dataset sources? Which sources have the highest quality unique content?"
```

### Source Distribution Analysis
**Check dataset balance:**
```bash
gemini -p "@data_formatted_with_sources/dataset_stats.json @data_formatted_deduplicated/dataset_stats.json How did deduplication affect the source distribution? Is the dataset still balanced?"
```

**Quality by source:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Group examples by source and analyze which sources provide the highest quality, most diverse animations"
```

**ManimBench quality assessment:**
```bash
gemini -p "@data_manimbench_test/train.json Why is ManimBench considered the highest quality dataset? Show examples that demonstrate its superiority"
```

### Description Quality Analysis
**Instruction clarity:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Analyze the user prompts/descriptions. Are they clear, specific, and diverse? Identify patterns of vague or ambiguous instructions"
```

**Description-code alignment:**
```bash
gemini -p "@data_formatted_deduplicated/test.json Sample 50 random examples and rate how well the generated code matches the description. Identify misalignments"
```

**Augmentation effectiveness:**
```bash
gemini -p "@prepare_data.py @data_formatted_with_sources/train.json Analyze the augmentation strategy. Are the prompt variations improving diversity or creating redundancy?"
```

### Dataset Integration Analysis
**Compare dataset formats:**
```bash
gemini -p "@DATASETS.md @prepare_data.py @extractors/sources/ How does the plugin-based pipeline handle different dataset formats and field names? Are there edge cases it might miss?"
```

**Potential dataset evaluation:**
```bash
gemini -p "@DATASETS.md Looking at the potential datasets list, which ones would add the most value based on what's currently missing in our dataset?"
```

**3Blue1Brown comparison:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Do any examples closely match 3Blue1Brown's style? What percentage of the dataset captures educational mathematical animations?"
```

## When to Use Gemini CLI for This Project

Use `gemini -p` when:
- Analyzing the entire training dataset that would exceed Claude's context
- Comparing datasets before and after deduplication or processing
- Validating code quality across thousands of examples
- Analyzing description patterns and quality across sources
- Checking for systematic issues in the dataset preparation pipeline
- Evaluating potential new datasets for integration
- Analyzing the effectiveness of augmentation strategies
- Investigating source distribution and balance issues
- Validating that the dataset meets quality standards for fine-tuning

## Specific Dataset Pipeline Commands

**Pre-processing analysis:**
```bash
# Before running pipeline - check raw dataset quality
gemini -p "@compare_datasets.py What quality issues exist in the raw datasets that the pipeline should address?"
```

**Mid-pipeline validation:**
```bash
# After initial processing - check for issues
gemini -p "@data_formatted_with_sources/train.json Are there systematic issues that need fixing before deduplication?"

# After deduplication - verify quality
gemini -p "@data_formatted_deduplicated/train.json @deduplication_report.json Did deduplication maintain dataset quality and diversity?"
```

**Post-pipeline analysis:**
```bash
# Full dataset quality assessment
gemini -p "@data_formatted_deduplicated/ @dataset_stats.json Provide a comprehensive quality assessment of the final dataset"

# Training readiness check
gemini -p "@data_formatted_deduplicated/train.json @data_formatted_deduplicated/test.json Is this dataset ready for fine-tuning? What improvements would have the most impact?"
```

## Dataset-Specific Quality Checks

**Animation diversity:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Categorize the animations by type (math, physics, CS, abstract, etc). Is there good coverage of different animation styles?"
```

**Complexity analysis:**
```bash
gemini -p "@data_formatted_deduplicated/train.json Analyze code complexity: How many use basic shapes vs custom VMobjects? How many use animations vs static scenes?"
```

**Common patterns detection:**
```bash
gemini -p "@data_formatted_deduplicated/train.json What are the 10 most common animation patterns? Are we over-representing certain types of animations?"
```

**Missing capabilities:**
```bash
gemini -p "@data_formatted_deduplicated/train.json @DATASETS.md Based on Manim's full capabilities, what types of animations are underrepresented in our dataset?"
```

## Important Notes for This Project

- Paths in @ syntax are relative to your current working directory when invoking gemini
- The CLI will include file contents directly in the context
- Gemini's context window can handle entire datasets that would overflow Claude's context
- Use specific queries about code quality, description clarity, and source distribution
- When checking dataset quality, focus on diversity, correctness, and alignment
- The deduplication reports are particularly useful for understanding dataset overlap
- Cross-reference with potential datasets in DATASETS.md for gap analysis
- Consider both syntactic correctness and semantic quality of the Manim code