import os
# os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
# os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

import yaml
import json
import csv
import pandas as pd
from datetime import datetime
import argparse
from utils.utils import (
    load_config,
    load_model,
    preprocess_for_inference,
    format_mcq_question,
)
from dotenv import load_dotenv
from tqdm import tqdm
import unsloth
from utils.postprocessing import post_process_qwen3vl_output  # <- add import

# Load environment variables from .env file
load_dotenv()

def predict_answer(model, processor, tokenizer, video_path, question_text, model_name, use_unsloth):
    """Unified prediction path using shared preprocessing and decoding."""
    inputs, decode = preprocess_for_inference(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        video_path=video_path,
        question_text=question_text,
        model_name=model_name,
        use_unsloth=use_unsloth,
    )
    output_ids = model.generate(**inputs, max_new_tokens=8)
    return decode(output_ids)

def save_submission_csv(results, infer_data_path, output_path):
    ts = datetime.now().strftime("%m%d_%H%M%S")
    if output_path is None:
        # Place submission in a 'submission' folder at same level as infer_data_path
        submission_dir = os.path.join(os.path.dirname(infer_data_path), "submission")
        os.makedirs(submission_dir, exist_ok=True)
        output_path = submission_dir
    output_csv = os.path.join(output_path, f"submission_{ts}.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


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
        video_path = os.path.join(os.path.dirname(os.path.dirname(infer_data_path)), item["video_path"])
        # Build consistent MCQ prompt
        question_text = format_mcq_question(item["question"], item.get("choices", []))
        response = (
            model.predict(video_path, question_text)
            if model_name == "placeholder"
            else predict_answer(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                video_path=video_path,
                question_text=question_text,
                model_name=model_name,
                use_unsloth=use_unsloth,
            )
        )
        # Apply Qwen VL post-processing
        # if "qwen" in model_name.lower() and "vl" in model_name.lower():
        # TODO: Enable conditionally based on model_type
        # response = post_process_qwen3vl_output(response)
        
        results.append({"id": item.get("id", ""), "answer": response})
        # print(f"Processed {item.get('id', 'unknown')} -> {response}")
    
    return results

def _normalize_quantization(cfg: dict) -> dict:
    """Support new 'quantization' block and legacy flags."""
    q = cfg.get("quantization")
    if q is not None:
        return {
            "enabled": bool(q.get("enabled", False)),
            "mode": str(q.get("mode", "8bit")).lower(),
        }
    # Legacy: use_8bit_quantization => 8bit
    if "use_8bit_quantization" in cfg:
        return {"enabled": bool(cfg.get("use_8bit_quantization", False)), "mode": "8bit"}
    # Legacy: use_quantization => assume 4bit (used for Unsloth earlier)
    if "use_quantization" in cfg:
        return {"enabled": bool(cfg.get("use_quantization", False)), "mode": "4bit"}
    return {"enabled": False, "mode": "8bit"}

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
    infer_data_path = config.get("infer_data_path", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/public_test/public_test.json"))
    output_path = config.get("output_path", None)
    attn_implementation = config.get("attn_implementation", "flash_attention_2")
    quantization = _normalize_quantization(config)
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
        # Set wandb API key if provided
        if wandb_api_key:
            os.environ["WANDB_API_KEY"] = wandb_api_key
        
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            tags=wandb_tags,
            config=config,  # Log all config settings
            settings=wandb.Settings(
                console="wrap",  # Log console output
                _disable_stats=False,  # Enable system performance logging
            )
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