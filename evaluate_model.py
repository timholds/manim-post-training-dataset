#!/usr/bin/env python3
"""
Comprehensive evaluation script for the fine-tuned Manim code generation model.
Calculates standard LLM metrics and code-specific quality measures.
"""

import torch
import json
import logging
import wandb
import numpy as np
from pathlib import Path
from unsloth import FastLanguageModel
from datasets import Dataset
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
import nltk
from collections import defaultdict
import time
import ast
import re
from manim_code_extractor import ManimCodeExtractor

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
except:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Evaluation configuration
WANDB_PROJECT = "manim-post-train"
WANDB_ENTITY = None
USE_WANDB = True

class ManimCodeEvaluator:
    """Comprehensive evaluator for Manim code generation."""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path
        self.model = None
        self.tokenizer = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.code_extractor = ManimCodeExtractor()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(self.model_path),
            max_seq_length=2048,
            dtype=torch.float16,
            load_in_4bit=True,
        )
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        logger.info("Model loaded successfully")
        
    def generate_code(self, prompt, max_length=1024, temperature=0.7):
        """Generate Manim code for a given prompt."""
        messages = [{"role": "user", "content": prompt}]
        
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Use ManimCodeExtractor to handle chat template extraction
        extracted_code = self.code_extractor.extract(response)
        
        return extracted_code
    
    def calculate_perplexity(self, texts):
        """Calculate perplexity on a set of texts."""
        total_loss = 0
        total_tokens = 0
        
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].size(1)
                total_tokens += inputs["input_ids"].size(1)
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        return perplexity
    
    def evaluate_code_quality(self, code):
        """Evaluate the quality of generated Manim code."""
        # Use ManimCodeExtractor for validation
        validation_result = self.code_extractor.validate(code)
        
        metrics = {
            "has_imports": False,
            "has_class": False,
            "has_construct": False,
            "has_self": False,
            "syntax_valid": validation_result.is_valid,
            "has_manim_objects": False,
            "has_animations": False,
            "line_count": 0,
            "char_count": len(code),
            "import_count": 0,
            "class_count": 0,
            "method_count": 0,
            "animation_count": 0,
            "validation_errors": len(validation_result.errors),
            "validation_warnings": len(validation_result.warnings),
        }
        
        # Basic checks (keep for backward compatibility and detailed metrics)
        metrics["has_imports"] = "from manim import" in code or "import manim" in code
        metrics["has_class"] = "class" in code and "Scene" in code
        metrics["has_construct"] = "def construct" in code
        metrics["has_self"] = "self" in code
        metrics["line_count"] = len(code.split('\n'))
        
        # Count specific elements
        metrics["import_count"] = len(re.findall(r'^import |^from .* import', code, re.MULTILINE))
        metrics["class_count"] = len(re.findall(r'^class \w+', code, re.MULTILINE))
        metrics["method_count"] = len(re.findall(r'def \w+', code))
        
        # Check for Manim-specific elements
        manim_objects = ['Circle', 'Square', 'Text', 'Line', 'Dot', 'Arrow', 'Rectangle', 
                        'Polygon', 'Ellipse', 'Arc', 'NumberPlane', 'Axes', 'Graph']
        metrics["has_manim_objects"] = any(obj in code for obj in manim_objects)
        
        # Check for animations
        animations = ['play', 'animate', 'Create', 'Write', 'FadeIn', 'FadeOut', 
                     'Transform', 'ReplacementTransform', 'MoveToTarget', 'Rotate', 
                     'Scale', 'Shift', 'GrowFromCenter', 'ShrinkToCenter']
        animation_matches = sum(1 for anim in animations if anim in code)
        metrics["has_animations"] = animation_matches > 0
        metrics["animation_count"] = animation_matches
        
        # Check syntax validity with AST (for extra validation)
        try:
            ast.parse(code)
            metrics["syntax_valid"] = True
        except SyntaxError:
            metrics["syntax_valid"] = False
        
        return metrics
    
    def calculate_rouge_scores(self, generated, reference):
        """Calculate ROUGE scores between generated and reference code."""
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1_f": scores['rouge1'].fmeasure,
            "rouge2_f": scores['rouge2'].fmeasure,
            "rougeL_f": scores['rougeL'].fmeasure,
        }
    
    def calculate_bert_score(self, generated_list, reference_list):
        """Calculate BERTScore for semantic similarity."""
        P, R, F1 = bert_score(generated_list, reference_list, lang="en", verbose=False)
        return {
            "bert_score_precision": P.mean().item(),
            "bert_score_recall": R.mean().item(),
            "bert_score_f1": F1.mean().item(),
        }
    
    def evaluate_dataset(self, test_data_path):
        """Evaluate model on the test dataset."""
        logger.info(f"Loading test dataset from {test_data_path}")
        
        # Load test data
        test_data = []
        with open(test_data_path, 'r') as f:
            for line in f:
                test_data.append(json.loads(line))
        
        logger.info(f"Loaded {len(test_data)} test samples")
        
        # Initialize metrics
        all_metrics = defaultdict(list)
        generated_codes = []
        reference_codes = []
        
        # Evaluate each sample
        for i, sample in enumerate(test_data):
            prompt = sample['conversations'][0]['value']
            reference_code = sample['conversations'][1]['value']
            
            logger.info(f"Evaluating sample {i+1}/{len(test_data)}")
            
            # Generate code
            start_time = time.time()
            generated_code = self.generate_code(prompt)
            generation_time = time.time() - start_time
            
            # Evaluate code quality
            quality_metrics = self.evaluate_code_quality(generated_code)
            
            # Calculate ROUGE scores
            rouge_scores = self.calculate_rouge_scores(generated_code, reference_code)
            
            # Store for BERTScore calculation
            generated_codes.append(generated_code)
            reference_codes.append(reference_code)
            
            # Aggregate metrics
            all_metrics["generation_time"].append(generation_time)
            for key, value in quality_metrics.items():
                all_metrics[f"quality_{key}"].append(value)
            for key, value in rouge_scores.items():
                all_metrics[key].append(value)
        
        # Calculate BERTScore for all samples
        logger.info("Calculating BERTScore...")
        bert_scores = self.calculate_bert_score(generated_codes, reference_codes)
        for key, value in bert_scores.items():
            all_metrics[key] = value
        
        # Calculate perplexity on test set
        logger.info("Calculating perplexity...")
        test_texts = [f"<|im_start|>user\n{s['conversations'][0]['value']}<|im_end|>\n<|im_start|>assistant\n{s['conversations'][1]['value']}<|im_end|>" 
                     for s in test_data]
        perplexity = self.calculate_perplexity(test_texts[:50])  # Sample for efficiency
        all_metrics["perplexity"] = perplexity
        
        # Calculate aggregate statistics
        summary_metrics = {}
        for key, values in all_metrics.items():
            if isinstance(values, list):
                if all(isinstance(v, bool) for v in values):
                    summary_metrics[f"{key}_rate"] = sum(values) / len(values)
                else:
                    summary_metrics[f"{key}_mean"] = np.mean(values)
                    summary_metrics[f"{key}_std"] = np.std(values)
                    if key not in ["generation_time"]:
                        summary_metrics[f"{key}_min"] = np.min(values)
                        summary_metrics[f"{key}_max"] = np.max(values)
            else:
                summary_metrics[key] = values
        
        return summary_metrics, all_metrics
    
    def log_to_wandb(self, metrics, run_name=None):
        """Log evaluation metrics to Weights & Biases."""
        if not USE_WANDB:
            return
        
        if not run_name:
            run_name = f"eval-{Path(self.model_path).name}"
        
        wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY,
            name=run_name,
            job_type="evaluation",
            config={
                "model_path": str(self.model_path),
                "evaluation_type": "comprehensive",
            },
            tags=["evaluation", "manim", "post-training"]
        )
        
        # Log all metrics
        wandb.log(metrics)
        
        # Create summary table
        summary_data = []
        for key, value in metrics.items():
            if not key.endswith("_std") and not key.endswith("_min") and not key.endswith("_max"):
                summary_data.append([key, value])
        
        wandb.log({"evaluation_summary": wandb.Table(
            columns=["Metric", "Value"],
            data=summary_data
        )})
        
        wandb.finish()
    
    def generate_report(self, metrics, output_path="evaluation_report.txt"):
        """Generate a human-readable evaluation report."""
        report = []
        report.append("=" * 80)
        report.append("MANIM CODE GENERATION MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Model information
        report.append(f"Model Path: {self.model_path}")
        report.append("")
        
        # Performance metrics
        report.append("PERFORMANCE METRICS")
        report.append("-" * 40)
        report.append(f"Perplexity: {metrics.get('perplexity', 'N/A'):.2f}")
        report.append(f"Average Generation Time: {metrics.get('generation_time_mean', 'N/A'):.2f}s")
        report.append("")
        
        # Code quality metrics
        report.append("CODE QUALITY METRICS")
        report.append("-" * 40)
        report.append(f"Syntax Valid Rate: {metrics.get('quality_syntax_valid_rate', 0)*100:.1f}%")
        report.append(f"Has Imports Rate: {metrics.get('quality_has_imports_rate', 0)*100:.1f}%")
        report.append(f"Has Class Rate: {metrics.get('quality_has_class_rate', 0)*100:.1f}%")
        report.append(f"Has Construct Method Rate: {metrics.get('quality_has_construct_rate', 0)*100:.1f}%")
        report.append(f"Has Manim Objects Rate: {metrics.get('quality_has_manim_objects_rate', 0)*100:.1f}%")
        report.append(f"Has Animations Rate: {metrics.get('quality_has_animations_rate', 0)*100:.1f}%")
        report.append("")
        report.append(f"Average Line Count: {metrics.get('quality_line_count_mean', 0):.1f}")
        report.append(f"Average Character Count: {metrics.get('quality_char_count_mean', 0):.1f}")
        report.append(f"Average Animation Count: {metrics.get('quality_animation_count_mean', 0):.1f}")
        report.append("")
        
        # Similarity metrics
        report.append("SIMILARITY METRICS")
        report.append("-" * 40)
        report.append(f"ROUGE-1 F1: {metrics.get('rouge1_f_mean', 0):.3f}")
        report.append(f"ROUGE-2 F1: {metrics.get('rouge2_f_mean', 0):.3f}")
        report.append(f"ROUGE-L F1: {metrics.get('rougeL_f_mean', 0):.3f}")
        report.append(f"BERTScore F1: {metrics.get('bert_score_f1', 0):.3f}")
        report.append("")
        
        # Save report
        with open(output_path, 'w') as f:
            f.write('\n'.join(report))
        
        # Also print to console
        print('\n'.join(report))
        
        logger.info(f"Evaluation report saved to {output_path}")

def main():
    """Run comprehensive model evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned Manim code generation model")
    parser.add_argument("--model-path", type=str, default="models/lora_model", 
                        help="Path to the fine-tuned model")
    parser.add_argument("--test-data", type=str, default="data/test.json",
                        help="Path to test dataset")
    parser.add_argument("--output-report", type=str, default="evaluation_report.txt",
                        help="Path for evaluation report")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb logging")
    
    args = parser.parse_args()
    
    if args.no_wandb:
        global USE_WANDB
        USE_WANDB = False
    
    # Initialize evaluator
    evaluator = ManimCodeEvaluator(args.model_path)
    
    # Check if model exists
    if not Path(args.model_path).exists():
        logger.error(f"Model not found at {args.model_path}")
        return
    
    # Load model
    evaluator.load_model()
    
    # Run evaluation
    logger.info("Starting comprehensive evaluation...")
    summary_metrics, all_metrics = evaluator.evaluate_dataset(args.test_data)
    
    # Log to wandb
    if USE_WANDB:
        evaluator.log_to_wandb(summary_metrics)
    
    # Generate report
    evaluator.generate_report(summary_metrics, args.output_report)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()