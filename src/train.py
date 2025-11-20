import os
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch


from unsloth import FastVisionModel
from transformers import TrainingArguments

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    SFTTrainer = None
    SFTConfig = None

from utils.utils import load_config, normalize_quantization
from models.utils import build_lora_model, load_model
from data import RoadBuddyVideoDataset, ConversationSample
from dotenv import load_dotenv

load_dotenv()


def get_training_args(output_model_path: str, config: Dict[str, Any]):
    total_epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    lr = float(config.get("learning_rate", 2e-4))
    gradient_accumulation = config.get("grad_accum_steps", 8)
    warmup = config.get("warmup_steps", 5)
    max_steps = config.get("max_steps")  # optional override
    push_to_hub = config.get("push_to_hub", False)
    hub_model_id = config.get("hub_model_id", None)
    
    print("Initial LR:", lr)

    # Prefer SFTConfig directly for TRL
    if SFTConfig is None:
        # Fallback to TrainingArguments if TRL is missing (will error earlier)
        return TrainingArguments(
            output_dir=output_model_path,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=lr,
            warmup_steps=warmup,
            num_train_epochs=total_epochs,
            # max_steps=10,
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            save_total_limit=config.get("save_total_limit", 3),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),
            optim="adamw_torch",
            report_to=["trackio"] if config.get("use_wandb", False) else ["none"],
            push_to_hub=push_to_hub,
            hub_model_id=hub_model_id,
        )
    return SFTConfig(
        output_dir=output_model_path,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        warmup_steps=warmup,
        num_train_epochs=total_epochs,
        # max_steps=10,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        optim="adamw_8bit",
        report_to=["trackio"] if config.get("use_wandb", False) else ["none"],

        # Required for Unsloth vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.get("max_length", 2048),
        
        # Optional extras:
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 3407),
        
        # Push to hub settings:
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        max_seq_length=50000
    )


def collate_unsloth(samples: List[ConversationSample]):
    """Minimal collator placeholder when UnslothVisionDataCollator not available.
    Falls back to simple list of messages for trainer expecting text format.
    """
    return {"messages": [s.messages for s in samples]}


def main():
    parser = argparse.ArgumentParser(description="Unsloth video finetuning for Road Buddy")
    parser.add_argument("--config", type=str, default="configs/config_unsloth.yaml", help="Config path")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HF token if needed")
    args = parser.parse_args()

    config = load_config(args.config)
    use_unsloth = config.get("use_unsloth", True)
    model_name = config.get("model_name", "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
    attn_impl = config.get("attn_implementation", "flash_attention_2")

    quantization = normalize_quantization(config, use_unsloth=use_unsloth)

    # Read train_data_path and output_model_path from config, allow CLI override
    train_data_path = config.get("train_data_path", "data/train/train.json")
    output_model_path = config.get("output_model_path", "models/unsloth_lora")
    
    # Push to hub configuration
    push_to_hub = config.get("push_to_hub", False)
    hub_model_id = config.get("hub_model_id", None)

    # Wandb configuration
    use_wandb = config.get("use_wandb", False)
    wandb_run_name = config.get("wandb_run_name", None)

    # Initialize wandb if enabled
    if use_wandb:
        # Get tokens from environment
        # wandb_api_key = os.getenv("WANDB_API_KEY")
        # wandb = __import__('wandb')
        import trackio as wandb
        wandb.init(
            project="road-buddy",
            space_id='tryourbest/road-buddy-tracking',
            name=wandb_run_name,
            config=config,  # Log all config settings
            private=True,
        )
        # print(f"Wandb run initialized: {wandb.run.name}")
        # print(f"Wandb run URL: {wandb.run.url}")

    # Load base model (training mode)
    model, processor, tokenizer = load_model(
        model_name=model_name,
        attn_implementation=attn_impl,
        quantization=quantization,
        use_unsloth=use_unsloth,
        hf_token=args.hf_token,
        inference_mode=False,
    )

    # Apply LoRA adapters if Unsloth
    model, tokenizer = build_lora_model(model, tokenizer, config)

    dataset = RoadBuddyVideoDataset(train_data_path, use_unsloth=use_unsloth, signs_dir="data/traffic_signs", use_support_frame=True)

    # Prepare SFT trainer
    if SFTTrainer is None:
        raise ImportError("trl not installed. Install with: pip install trl")

    if use_unsloth:
        try:
            from unsloth.trainer import UnslothVisionDataCollator
            data_collator = UnslothVisionDataCollator(model, tokenizer, max_seq_length=50000)
        except Exception:
            print("Falling back to simple collator; install unsloth_zoo for vision support.")
            data_collator = collate_unsloth
    else:
        data_collator = collate_unsloth

    training_args = get_training_args(output_model_path, config)

    # Convert dataset to list of dict for SFTTrainer expected format
    hf_style_dataset = [ {"messages": sample.messages} for sample in dataset ]

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=hf_style_dataset,
        data_collator=data_collator,
        args=training_args,
    )
    
    if model_name.startswith("qwen") and use_unsloth:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
            response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
        )

    print("Starting training...")
    train_result = trainer.train()
    print("Training complete. Saving LoRA adapters...")

    os.makedirs(output_model_path, exist_ok=True)
    if use_unsloth:
        # Save only adapters
        model.save_pretrained(output_model_path)
        tokenizer.save_pretrained(output_model_path)
    else:
        # Save full model if not using Unsloth
        model.save_pretrained(output_model_path)
        if tokenizer:
            tokenizer.save_pretrained(output_model_path)

    print("Saved to", output_model_path)
    print("Training metrics:", train_result)
    
    # Push to Hugging Face Hub if configured
    if push_to_hub:
        if hub_model_id is None:
            print("Warning: push_to_hub is True but hub_model_id is not set. Skipping push.")
        else:
            print(f"Pushing model to Hugging Face Hub: {hub_model_id}")
            model.push_to_hub(hub_model_id, token=args.hf_token)
            tokenizer.push_to_hub(hub_model_id, token=args.hf_token)
            print(f"Successfully pushed to {hub_model_id}")


if __name__ == "__main__":
    main()
