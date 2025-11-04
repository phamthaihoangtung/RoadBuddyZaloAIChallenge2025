import yaml
import os
import torch
import json
import csv
import pandas as pd
from datetime import datetime
from transformers import AutoProcessor, AutoModelForCausalLM
import argparse
from utils.utils import load_config


def predict_conversation(model, processor, video_path, question):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant. \
            Answer the multiple choice question based on the video provided.\
                Please select one of the provided choices and respond with only the choice letter in A, B, C, or D."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                {"type": "text", "text": question},
            ]
        },
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
    output_ids = model.generate(**inputs, max_new_tokens=128)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

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

def main():
    parser = argparse.ArgumentParser(description="Run inference for Road Buddy Challenge")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    args = parser.parse_args()
    
    # Load config from the specified path
    config = load_config(args.config)
    model_name = config.get("model_name", "DAMO-NLP-SG/VideoLLaMA3-7B")
    infer_data_path = config.get("infer_data_path", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/public_test/public_test.json"))
    output_path = config.get("output_path", None)
    attn_implementation = config.get("attn_implementation", "flash_attention_2")

    if model_name == "placeholder":
        from utils.placeholder_model import PlaceholderModel
        model = PlaceholderModel(config)
        processor = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    with open(infer_data_path, "r") as f:
        test_data = json.load(f)

    results = []
    for item in test_data["data"]:
        video_path = os.path.join(os.path.dirname(os.path.dirname(infer_data_path)), item["video_path"])
        # Concatenate question and choices
        question = item["question"] + " " + " ".join(item.get("choices", []))
        if model_name == "placeholder":
            response = model.predict(video_path, question)
        else:
            response = predict_conversation(model, processor, video_path, question)
        results.append({"id": item.get("id", ""), "answer": response})

    save_submission_csv(results, infer_data_path, output_path)

if __name__ == "__main__":
    main()