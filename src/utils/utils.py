from datetime import datetime
import pandas as pd
import yaml
import os
import torch
from typing import Any, Callable, Dict, Optional, Tuple
import unsloth

# Lazy imports for transformers inside functions to keep this module light when unavailable


def load_config(path: str):
    """Load a YAML configuration file and return it as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)

def normalize_quantization(cfg: Dict[str, Any], use_unsloth: bool=True) -> Dict[str, Any]:
    q = cfg.get("quantization")
    if q is not None:
        return {
            "enabled": bool(q.get("enabled", False)),
            "mode": str(q.get("mode", "4bit" if use_unsloth else "8bit")).lower(),
        }
    return {"enabled": False, "mode": "4bit" if use_unsloth else "8bit"}

def video_abs_path(rel: str) -> str:
    """Convert relative video path to absolute path.
    
    Dataset uses paths like train/videos/... - ensure absolute path.
    """
    p = os.path.join("data", rel) if not rel.startswith("data/") else rel
    return p

def save_submission_csv(results, infer_data_path, output_path):
    ts = datetime.now().strftime("%m%d_%H%M%S")
    # Determine target directory
    if not output_path:
        submission_dir = os.path.join(os.path.dirname(infer_data_path), "submission")
        output_dir = submission_dir
    else:
        output_dir = output_path
    # Ensure directory exists
    os.makedirs(output_dir, exist_ok=True)

    output_csv = os.path.join(output_dir, f"submission_{ts}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
