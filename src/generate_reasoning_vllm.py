import argparse
import json
import os
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

from utils.utils import load_json_data, video_abs_path

os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

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
    return "\n".join([f"- {key}: {value}" for key, value in choices.items()])

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

def main():
    parser = argparse.ArgumentParser(description="Generate reasoning for the dataset.")
    parser.add_argument("--data_path", type=str, default="data/train/train.json", help="Path to the input data JSON file.")
    parser.add_argument("--output_path", type=str, default="data/train/train_thinking.json", help="Path to save the output JSON file.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference.")
    args = parser.parse_args()

    checkpoint_path = "Qwen/Qwen3-VL-32B-Thinking-FP8"
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    
    llm = LLM(
        model=checkpoint_path,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
        enforce_eager=False,
        tensor_parallel_size=torch.cuda.device_count(),
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=256,
        top_k=-1,
        stop_token_ids=[],
    )

    test_data = load_json_data(args.data_path)
    results = []
    
    if DEBUG:
        test_data = test_data[:5]
    
    for i in tqdm(range(0, len(test_data), args.batch_size), desc="Generating Reasoning"):
        batch = test_data[i:i+args.batch_size]
        
        batch_messages = []
        for item in batch:
            question = item['question']
            choices = item['choices']
            answer = item['answer']
            video_path = video_abs_path(item['video_path'])

            formatted_choices = format_choices(choices)
            prompt_text = PROMPT_TEMPLATE.format(question=question, choices=formatted_choices, answer=choices[answer])

            messages = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": prompt_text},
                ],
            }]
            batch_messages.append(messages)

        inputs = [prepare_inputs_for_vllm(msg, processor) for msg in batch_messages]
        
        outputs = llm.generate(inputs, sampling_params=sampling_params)

        for original_item, output in zip(batch, outputs):
            generated_text = output.outputs[0].text
            try:
                # Clean up potential markdown code fences
                if generated_text.strip().startswith("```json"):
                    generated_text = generated_text.strip()[7:-3].strip()
                
                reasoning_data = json.loads(generated_text)
                new_item = original_item.copy()
                new_item['reasoning'] = reasoning_data.get('reasoning_steps', [])
                results.append(new_item)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON for item {original_item['id']}. Raw output: {generated_text}")
                new_item = original_item.copy()
                new_item['reasoning'] = ["Error: Failed to generate valid JSON reasoning."]
                results.append(new_item)

    with open(args.output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Reasoning generation complete. Output saved to {args.output_path}")

if __name__ == '__main__':
    main()
