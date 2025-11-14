import argparse
import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

import json
from typing import Callable, Dict, Tuple
import pandas as pd
from datetime import datetime

import torch
import unsloth

from utils.utils import (
    load_config,
    normalize_quantization,
    preprocess_for_inference,
    save_submission_csv,
)
from dotenv import load_dotenv
from tqdm import tqdm

from models.utils import load_model
from data import build_user_content

# Load environment variables from .env file
load_dotenv()

def preprocess_for_inference(
    model,
    processor,
    tokenizer,
    messages,
    model_name: str,
    use_unsloth: bool,
) -> Tuple[Dict[str, torch.Tensor], Callable[[torch.Tensor], str]]:
    """Create tokenized inputs for generation and a decode function.

    Returns (inputs, decode_fn). inputs are on the correct device with proper dtype.
    decode_fn takes output_ids and returns a string.
    """
    if model_name == "placeholder":
        def _decode(ids):
            return "A"  # placeholder

        return {}, _decode

    if use_unsloth:
        # Unsloth pipeline uses tokenizer + unsloth_zoo vision_utils
        try:
            from unsloth_zoo import vision_utils
        except ImportError as e:
            raise ImportError(
                "Unsloth requested but unsloth_zoo not installed. Install with: pip install unsloth-zoo"
            ) from e

        image_input, video_input = vision_utils.process_vision_info(messages)
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            text=input_text,
            images=image_input,
            videos=video_input,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        def _decode(output_ids: torch.Tensor) -> str:
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return inputs, _decode
    else:
        inputs = processor(conversation=messages, return_tensors="pt")
        # Move to device and set dtypes
        inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        def _decode(output_ids: torch.Tensor) -> str:
            return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return inputs, _decode

def predict_answer(model, processor, tokenizer, messages, model_name, use_unsloth):
    """Unified prediction path using shared preprocessing and decoding."""
    inputs, decode = preprocess_for_inference(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        messages=messages,
        model_name=model_name,
        use_unsloth=use_unsloth,
    )
    output_ids = model.generate(**inputs, max_new_tokens=8)
    return decode(output_ids)


def load_test_data(infer_data_path):
    """Load test data from JSON file."""
    with open(infer_data_path, "r") as f:
        test_data = json.load(f)
    return test_data

def run_inference(model, processor, tokenizer, test_data, infer_data_path, model_name, use_unsloth, DEBUG=False):
    """Run inference on test data and return results."""
    results = []
    if DEBUG:
        test_data["data"] = test_data["data"][:5]  # Limit to first 5 samples in debug mode

    for item in tqdm(test_data["data"]):
        messages = build_user_content(item["video_path"], item["question"], item["choices"], use_unsloth)
        response = (
            model.predict(messages)
            if model_name == "placeholder"
            else predict_answer(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                messages=messages,
                model_name=model_name,
                use_unsloth=use_unsloth,
            )
        )
       
        results.append({"id": item.get("id", ""), "answer": response})
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference for Road Buddy Challenge")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config_unsloth.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_name = config.get("model_name", "DAMO-NLP-SG/VideoLLaMA3-7B")
    infer_data_path = config.get("infer_data_path", 
                                 os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/public_test/public_test.json"))
    output_path = config.get("output_path", None)
    attn_implementation = config.get("attn_implementation", "flash_attention_2")
    quantization = normalize_quantization(config)
    use_unsloth = config.get("use_unsloth", False)
    
    # Wandb configuration
    use_wandb = config.get("use_wandb", False)
    wandb_project = config.get("wandb_project", "road-buddy-inference")
    wandb_run_name = config.get("wandb_run_name", None)
    wandb_tags = config.get("wandb_tags", [])
    
    # Get tokens from environment
    hf_token = os.getenv("HF_TOKEN")
    wandb_api_key = os.getenv("WANDB_API_KEY")
    
    # Initialize wandb if enabled
    if use_wandb:
        wandb = __import__('wandb')
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config=config,  # Log all config settings
        )
        print(f"Wandb run initialized: {wandb.run.name}")
        print(f"Wandb run URL: {wandb.run.url}")
    
    # Load model
    print("Loading model...")
    model, processor, tokenizer = load_model(
        model_name=model_name,
        attn_implementation=attn_implementation,
        quantization=quantization,
        use_unsloth=use_unsloth,
        hf_token=hf_token,
        inference_mode=True,
    )
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(infer_data_path)
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, processor, tokenizer, test_data, infer_data_path, model_name, use_unsloth)
    
    # Save results
    print("Saving results...")
    save_submission_csv(results, infer_data_path, output_path)
    
    # Finish wandb run
    if use_wandb:
        wandb = __import__('wandb')
        wandb.finish()

if __name__ == "__main__":
    main()