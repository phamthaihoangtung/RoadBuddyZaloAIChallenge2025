"""
Post-processing module for Qwen3-VL model outputs.

This module provides functions to clean and standardize outputs from Qwen3-VL
for the Road Buddy Challenge, ensuring answers are in the correct format (A, B, C, or D).
"""

import re
import pandas as pd
from pathlib import Path
import argparse


def post_process_qwen3vl_output(output: str) -> str:
    """
    Post-process Qwen3-VL output to extract the answer choice.
    
    The function handles various output formats and extracts the final answer
    (A, B, C, or D) from the model's response.
    
    Args:
        output: Raw output string from Qwen3-VL model
        
    Returns:
        Cleaned answer string (A, B, C, or D). Returns 'A' as default if no valid answer found.
        
    Examples:
        >>> post_process_qwen3vl_output("The answer is B")
        'B'
        >>> post_process_qwen3vl_output("I think the correct choice is C.")
        'C'
        >>> post_process_qwen3vl_output("D")
        'D'
    """
    if not output or not isinstance(output, str):
        return 'A'
    
    # Extract answer after "assistant" token if present
    assistant_split = re.split(r'assistant[:\s]', output, flags=re.IGNORECASE)
    if len(assistant_split) > 1:
        output = assistant_split[-1].strip()
    
    # Strip whitespace
    output = output.strip()
    
    # Only extract the first valid answer character (A/B/C/D), even if answer is like "A.Yes"
    match = re.match(r'^([A-Da-d])\b', output)
    if match:
        return match.group(1).upper()
    
    # Pattern 1: Direct answer (just A, B, C, or D)
    if output.upper() in ['A', 'B', 'C', 'D']:
        return output.upper()
    
    # Pattern 2: Answer at the start (e.g., "A.", "A)", "A:", "A -")
    match = re.match(r'^([A-D])[.).:\-\s]', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 3: "Answer: X" or "The answer is X"
    match = re.search(r'(?:answer|choice|option)(?:\s+is)?[:\s]+([A-D])', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Pattern 4: Find last occurrence of A, B, C, or D
    matches = re.findall(r'\b([A-D])\b', output, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # Pattern 5: Look for "select X" or "choose X"
    match = re.search(r'(?:select|choose)\s+([A-D])', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Default fallback
    return 'A'


def _default_output_path(input_path: str) -> str:
    """
    Derive default output path by appending '_postprocessed' before the CSV suffix.
    """
    p = Path(input_path)
    if p.suffix.lower() == ".csv":
        return str(p.with_name(f"{p.stem}_postprocessed{p.suffix}"))
    return str(p.with_name(f"{p.name}_postprocessed.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process Qwen3-VL CSV outputs.")
    parser.add_argument("--input_path", help="Path to input CSV containing an 'answer' column.")
    parser.add_argument("--output_path", help="Path to save post-processed CSV. Defaults to <input>_postprocessed.csv")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path or _default_output_path(input_path)

    # Load the CSV file
    df = pd.read_csv(input_path)

    print(f"Processing {len(df)} rows from {input_path}")
    print("=" * 60)

    # Apply post-processing to each answer
    df['answer'] = df['answer'].apply(post_process_qwen3vl_output)

    # Save to output file
    df.to_csv(output_path, index=False)

    print(f"Processed results saved to: {output_path}")
    print("\nFirst 10 processed rows:")
    print(df.head(10).to_string(index=False))
