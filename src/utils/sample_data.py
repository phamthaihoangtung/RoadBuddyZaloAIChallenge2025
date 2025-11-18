import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import json
import random
from typing import Dict, List, Any
from utils.utils import load_json_data


def random_sample_from_json(json_data: Dict[str, Any], k: int) -> Dict[str, Any]:
    """
    Randomly sample k items from the JSON data object.
    
    Args:
        json_data: Dictionary containing the JSON data with a 'data' key
        k: Number of samples to randomly select
        
    Returns:
        Dictionary with the same structure but only k samples in 'data'
    """
    if 'data' not in json_data:
        raise ValueError("JSON object must contain a 'data' key")
    
    all_data = json_data['data']
    
    # Filter out items with only '_unused_' key
    valid_data = [item for item in all_data if not (len(item) == 1 and '_unused_' in item)]
    
    if k > len(valid_data):
        print(f"Warning: Requested {k} samples but only {len(valid_data)} valid samples available. Returning all valid samples.")
        k = len(valid_data)
    
    # Random sample without fixed seed
    sampled_data = random.sample(valid_data, k)
    
    # Create result with same structure
    result = {
        "__count__": k,
        "data": sampled_data
    }
    
    return result


def load_and_sample(json_path: str, k: int, output_path: str = None) -> Dict[str, Any]:
    """
    Load JSON file, sample k items, and optionally save to output file.
    
    Args:
        json_path: Path to input JSON file
        k: Number of samples to randomly select
        output_path: Optional path to save sampled data
        
    Returns:
        Dictionary containing the sampled data
    """
    # Load JSON file
    data = load_json_data(json_path)
    
    # Sample k items
    sampled_data = random_sample_from_json(data, k)
    
    # Save to file if output path provided, else save next to input with _sampled suffix
    if output_path is None:
        input_path = Path(json_path)
        output_path = str(input_path.with_name(f"{input_path.stem}_sampled.json"))
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=4)
        print(f"Saved {k} samples to {output_path}")
    
    return sampled_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Randomly sample k items from JSON file")
    parser.add_argument("--input", type=str, default="data/train/train.json",
                        help="Path to input JSON file")
    parser.add_argument("--k", type=int, default=16,
                        help="Number of samples to select")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to output JSON file (optional)")
    
    args = parser.parse_args()
    
    # Sample and display
    sampled = load_and_sample(args.input, args.k, args.output)
    
    print(f"\nSuccessfully sampled {len(sampled['data'])} items")
    print(f"Sample IDs: {[item['id'] for item in sampled['data'][:5]]}...")
