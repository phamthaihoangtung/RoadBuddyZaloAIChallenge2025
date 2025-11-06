import yaml
import os
import torch
import json
import csv
import pandas as pd
from datetime import datetime
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
from utils.utils import load_config
from dotenv import load_dotenv
from tqdm import tqdm


# Load environment variables from .env file
load_dotenv()

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
    output_ids = model.generate(**inputs, max_new_tokens=8)
    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return response

def predict_unsloth(model, tokenizer, video_path, question):
    """Predict using Unsloth FastVisionModel with vision_utils"""
    from unsloth_zoo import vision_utils
    from unsloth import FastVisionModel
    
    FastVisionModel.for_inference(model)
    
    # Create structured message format
    messages = [
        {"role": "system", "content": "You are a helpful assistant. \
            Answer the multiple choice question based on the video provided. \
                Please select one of the provided choices and respond with only the choice letter in A, B, C, or D."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question}
            ]
        }
    ]
    
    # Process vision input using unsloth_zoo vision_utils
    image_input, video_input = vision_utils.process_vision_info(messages)
    
    # Apply chat template
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    # Tokenize inputs
    inputs = tokenizer(
        input_text,
        images=image_input,
        videos=video_input,
        return_tensors="pt"
    ).to(model.device)
    
    # Generate response
    output_ids = model.generate(**inputs, max_new_tokens=8)
    response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    
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

def load_model(model_name, attn_implementation, use_quantization, use_unsloth, hf_token):
    """Load and initialize the model and processor/tokenizer."""
    if model_name == "placeholder":
        from utils.placeholder_model import PlaceholderModel
        model = PlaceholderModel({"model_name": model_name})
        processor = None
        tokenizer = None
    elif use_unsloth:       
        from unsloth import FastVisionModel
        
        print(f"Loading Unsloth FastVisionModel from {model_name}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=use_quantization,
            use_gradient_checkpointing="unsloth",
            token=hf_token,
        )
        processor = None
    else:
        # Configure 8-bit quantization
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "attn_implementation": attn_implementation,
            "token": hf_token,
        }
        
        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model_kwargs["quantization_config"] = quantization_config
            print("Loading model with 8-bit quantization...")
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
        tokenizer = None
    
    return model, processor, tokenizer

def load_test_data(infer_data_path):
    """Load test data from JSON file."""
    with open(infer_data_path, "r") as f:
        test_data = json.load(f)
    return test_data

def run_inference(model, processor, tokenizer, test_data, infer_data_path, model_name, use_unsloth):
    """Run inference on test data and return results."""
    results = []
    for item in tqdm(test_data["data"]):
        video_path = os.path.join(os.path.dirname(os.path.dirname(infer_data_path)), item["video_path"])
        # Concatenate question and choices
        question = item["question"] + " " + " ".join(item.get("choices", []))
        
        if model_name == "placeholder":
            response = model.predict(video_path, question)
        elif use_unsloth:
            response = predict_unsloth(model, tokenizer, video_path, question)
        else:
            response = predict_conversation(model, processor, video_path, question)
        
        results.append({"id": item.get("id", ""), "answer": response})
        # print(f"Processed {item.get('id', 'unknown')} -> {response}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Run inference for Road Buddy Challenge")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file (default: configs/config.yaml)"
    )
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    model_name = config.get("model_name", "DAMO-NLP-SG/VideoLLaMA3-7B")
    infer_data_path = config.get("infer_data_path", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/public_test/public_test.json"))
    output_path = config.get("output_path", None)
    attn_implementation = config.get("attn_implementation", "flash_attention_2")
    use_quantization = config.get("use_8bit_quantization", False)
    use_unsloth = config.get("use_unsloth", False)
    
    # Get HuggingFace token from environment
    hf_token = os.getenv("HF_TOKEN")
    
    # Load model
    print("Loading model...")
    model, processor, tokenizer = load_model(model_name, attn_implementation, use_quantization, use_unsloth, hf_token)
    
    # Load test data
    print("Loading test data...")
    test_data = load_test_data(infer_data_path)
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, processor, tokenizer, test_data, infer_data_path, model_name, use_unsloth)
    
    # Save results
    print("Saving results...")
    save_submission_csv(results, infer_data_path, output_path)

if __name__ == "__main__":
    main()