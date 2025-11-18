import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"
# os.environ["UNSLOTH_DISABLE_STATISTICS"] = "0"
# os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

import argparse
import json
from tqdm import tqdm
import torch
import unsloth
from models.utils import load_model
from infer import predict_answer
from utils.utils import load_json_data, video_abs_path


DEBUG = True

PROMPT_TEMPLATE = """You are an expert AI assistant specializing in Vietnamese traffic law. Your task is to generate a dataset for finetuning.

Your job is to generate the `reasoning_steps` that logically explain why a "Known_Correct_Answer" is the correct solution for a given "Problem". The "Problem" is a multiple-choice question.

**CRITICAL INSTRUCTIONS:**
1.  Your *entire* output MUST be a single, valid JSON object.
2.  Do NOT write any text before or after the JSON (e.g., no "Here is the JSON...").
3.  The `reasoning_steps` must be a brief list of logical steps. These steps must explain *why* the correct answer is right and *why* the other options are wrong, referencing Vietnamese traffic rules where possible.
4.  The `final_answer` in your JSON *must exactly match* the "Known_Correct_Answer" provided.

---
**TASK:**

[Problem]:
Question: {question}
Choices:
{choices}

[Known_Correct_Answer]:
"{answer}"

[Your JSON Output]:
"""

def format_choices(choices):
    # Support both dict and list
    if isinstance(choices, dict):
        return "\n".join([f"- {key}: {value}" for key, value in choices.items()])
    elif isinstance(choices, list):
        return "\n".join([f"- {value}" for value in choices])
    else:
        return str(choices)

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning for the dataset (Unsloth).")
    parser.add_argument("--data_path", type=str, default="data/train/train.json", help="Path to the input data JSON file.")
    parser.add_argument("--output_path", type=str, default="data/train/train_thinking.json", help="Path to save the output JSON file.")
    parser.add_argument("--model_name", type=str, 
                        default="unsloth/Qwen3-VL-32B-Thinking", 
                        help="Hugging Face model id/path.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max new tokens for generation.")
    parser.add_argument("--use_unsloth", action="store_true", default=True, help="Force Unsloth preprocessing path.")
    parser.add_argument("--quantization_mode", type=str, default="4bit", choices=["4bit", "8bit", "none"],
                        help="Quantization mode (default: 4bit). Use 'none' to disable.")
    args = parser.parse_args()

    # Prepare quantization config (default 4bit)
    quantization = None
    if args.quantization_mode != "none":
        quantization = {"enabled": True, "mode": args.quantization_mode}

    # Load Unsloth model + processor + tokenizer
    model, processor, tokenizer = load_model(
        model_name=args.model_name,
        attn_implementation="flash_attention_2",
        quantization=quantization,
        use_unsloth=args.use_unsloth,
        hf_token=os.getenv("HF_TOKEN"),
        inference_mode=True,
    )

    # Load input data (supports either {"data": [...]} or [...] formats)
    raw = load_json_data(args.data_path)
    items = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
    if not isinstance(items, list):
        raise ValueError("Input JSON must be a list of items or an object with a 'data' list.")

    results = []
    if DEBUG:
        items = items[:5]

    for original_item in tqdm(items, desc="Generating Reasoning (Unsloth)"):
        question = original_item['question']
        choices = original_item['choices']
        answer = original_item['answer']
        video_path = video_abs_path(original_item['video_path'])

        prompt_text = PROMPT_TEMPLATE.format(
            question=question,
            choices=format_choices(choices),
            answer=answer,
        )

        # Build a single message dict (preprocess_for_inference wraps it into a list)
        message = {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": 8},
                {"type": "text", "text": prompt_text},
            ],
        }

        # Generate with shared predict_answer (Unsloth path selected by use_unsloth=True)
        response = predict_answer(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            messages=message,
            model_name=args.model_name,
            use_unsloth=args.use_unsloth,
            max_new_tokens=args.max_new_tokens,
        )

        generated_text = response.strip()
        # Clean up potential markdown code fences
        if generated_text.startswith("```json"):
            generated_text = generated_text[7:].strip()
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3].strip()
        elif generated_text.startswith("```"):
            generated_text = generated_text[3:].strip()
            if generated_text.endswith("```"):
                generated_text = generated_text[:-3].strip()

        try:
            reasoning_data = json.loads(generated_text)
            new_item = dict(original_item)
            new_item["reasoning"] = reasoning_data.get("reasoning_steps", [])
            results.append(new_item)
        except json.JSONDecodeError:
            print(f"Failed to parse JSON for item {original_item.get('id', '<no-id>')}. Raw output: {generated_text}")
            new_item = dict(original_item)
            new_item["reasoning"] = ["Error: Failed to generate valid JSON reasoning."]
            results.append(new_item)

    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Reasoning generation complete. Output saved to {args.output_path}")

if __name__ == '__main__':
    main()
