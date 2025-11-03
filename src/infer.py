import yaml
import os
import torch
import json
from transformers import AutoProcessor, AutoModelForCausalLM

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs/config.yaml")

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def predict_conversation(model, processor, video_path, question):
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
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

def main():
    config = load_config(CONFIG_PATH)
    model_name = config.get("model_name", "DAMO-NLP-SG/VideoLLaMA3-7B")
    infer_data_path = config.get("infer_data_path", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/public_test/public_test.json"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load public test conversations
    with open(infer_data_path, "r") as f:
        test_data = json.load(f)
    for item in test_data["data"][:10]:  # Limit to first 10 for quick testing
        video_path = os.path.join(os.path.dirname(os.path.dirname(infer_data_path)), item["video_path"])
        print(f"Processing video: {video_path}")
        question = item["question"]
        response = predict_conversation(model, processor, video_path, question)
        print(f"ID: {item.get('id', '')}")
        print(f"Question: {question}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()