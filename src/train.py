import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset

from unsloth import FastVisionModel
from transformers import TrainingArguments

try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    SFTTrainer = None
    SFTConfig = None

from utils.utils import load_config, load_model, format_mcq_question
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ConversationSample:
    messages: List[Dict[str, Any]]


class RoadBuddyVideoDataset(Dataset):
    """Dataset converting train.json rows into Unsloth vision conversation format.

    Each item becomes a list of messages: user (video + text) then assistant (answer text).
    Unsloth expects: [{role: user, content: [...]}, {role: assistant, content: [...]}]
    Video is referenced by path; reading/decoding is handled later by vision_utils.
    """

    def __init__(self, data_path: str, root_dir: str = "data", use_unsloth: bool = True):
        with open(data_path, "r") as f:
            raw = json.load(f)
        self.items = raw["data"]
        self.root_dir = root_dir
        self.use_unsloth = use_unsloth

    def __len__(self):
        return len(self.items)

    def _video_abs_path(self, rel: str) -> str:
        # dataset uses paths like train/videos/... ensure absolute path
        p = os.path.join("data", rel) if not rel.startswith("data/") else rel
        return p

    def __getitem__(self, idx: int) -> ConversationSample:
        item = self.items[idx]
        video_rel = item["video_path"]  # eg train/videos/xxx.mp4
        video_path = self._video_abs_path(video_rel)
        question = format_mcq_question(item["question"], item.get("choices"))
        raw_ans = item.get("answer", "")
        # Target should be just the choice letter (A/B/C/D)
        if isinstance(raw_ans, str) and "." in raw_ans:
            answer_text = raw_ans.split(".", 1)[0].strip()
        else:
            answer_text = (raw_ans or "A").strip()[:1]

        if self.use_unsloth:
            user_content = [
                {"type": "video", "video": video_path},
                {"type": "text", "text": question},
            ]
        else:
            # Generic processor format (with metadata structure) if not using Unsloth
            user_content = [
                {"type": "video", "video": {"video_path": video_path, "fps": 1, "max_frames": 128}},
                {"type": "text", "text": question},
            ]

        sample = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": answer_text}]},
        ]
        return ConversationSample(messages=sample)


def build_lora_model(base_model, tokenizer, config: Dict[str, Any]):
    """Wrap base_model with LoRA adapters using Unsloth FastVisionModel APIs if requested."""
    if not config.get("use_unsloth", False):
        return base_model, tokenizer  # LoRA only supported via Unsloth path here
    from unsloth import FastVisionModel
    lora_params = config.get("lora", {})
    # Defaults tuned for moderate parameter count
    base_model = FastVisionModel.get_peft_model(
        base_model,
        finetune_vision_layers=lora_params.get("finetune_vision_layers", False),
        finetune_language_layers=lora_params.get("finetune_language_layers", True),
        finetune_attention_modules=lora_params.get("finetune_attention_modules", True),
        finetune_mlp_modules=lora_params.get("finetune_mlp_modules", True),
        r=lora_params.get("r", 16),
        lora_alpha=lora_params.get("lora_alpha", 16),
        lora_dropout=lora_params.get("lora_dropout", 0),
        bias=lora_params.get("bias", "none"),
        use_gradient_checkpointing=lora_params.get("use_gradient_checkpointing", "unsloth"),
        random_state=lora_params.get("random_state", 42),
        use_rslora=lora_params.get("use_rslora", False),
        loftq_config=lora_params.get("loftq_config", None),
    )
    return base_model, tokenizer


def get_training_args(output_dir: str, config: Dict[str, Any]):
    total_epochs = config.get("epochs", 1)
    batch_size = config.get("batch_size", 1)
    lr = config.get("learning_rate", 2e-4)
    gradient_accumulation = config.get("grad_accum_steps", 1)
    warmup = config.get("warmup_steps", 5)
    max_steps = config.get("max_steps")  # optional override

    # Prefer SFTConfig directly for TRL
    if SFTConfig is None:
        # Fallback to TrainingArguments if TRL is missing (will error earlier)
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation,
            learning_rate=lr,
            warmup_steps=warmup,
            num_train_epochs=total_epochs,
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            save_total_limit=config.get("save_total_limit", 3),
            fp16=config.get("fp16", False),
            bf16=config.get("bf16", True),
            optim="adamw_torch",
            report_to=["none"],
        )
    return SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=lr,
        warmup_steps=warmup,
        num_train_epochs=total_epochs,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 3),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
        optim="adamw_8bit",
        report_to=["none"],
        # eos_token="<|im_end|>" if 'qwen' in config.get("model_name", "").lower() else None,

        # Required for Unsloth vision finetuning:
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_length=config.get("max_length", 2048),
        
        # Optional extras:
        weight_decay=config.get("weight_decay", 0.0),
        lr_scheduler_type=config.get("lr_scheduler_type", "linear"),
        seed=config.get("seed", 3407),
    )


def collate_unsloth(samples: List[ConversationSample]):
    """Minimal collator placeholder when UnslothVisionDataCollator not available.
    Falls back to simple list of messages for trainer expecting text format.
    """
    return {"messages": [s.messages for s in samples]}


def main():
    parser = argparse.ArgumentParser(description="Unsloth video finetuning for Road Buddy")
    parser.add_argument("--config", type=str, default="configs/config_unsloth.yaml", help="Config path")
    parser.add_argument("--train_json", type=str, default="data/train/train.json", help="Training JSON path")
    parser.add_argument("--output_dir", type=str, default="models/unsloth_lora", help="Output dir for adapters")
    parser.add_argument("--hf_token", type=str, default=os.getenv("HF_TOKEN"), help="HF token if needed")
    args = parser.parse_args()

    config = load_config(args.config)
    use_unsloth = config.get("use_unsloth", True)
    model_name = config.get("model_name", "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
    attn_impl = config.get("attn_implementation", "flash_attention_2")

    def _normalize_quantization(cfg: Dict[str, Any]) -> Dict[str, Any]:
        q = cfg.get("quantization")
        if q is not None:
            return {
                "enabled": bool(q.get("enabled", False)),
                "mode": str(q.get("mode", "4bit" if use_unsloth else "8bit")).lower(),
            }
        if "use_8bit_quantization" in cfg:
            return {"enabled": bool(cfg.get("use_8bit_quantization", False)), "mode": "8bit"}
        if "use_quantization" in cfg:
            return {"enabled": bool(cfg.get("use_quantization", False)), "mode": "4bit"}
        return {"enabled": False, "mode": "4bit" if use_unsloth else "8bit"}

    quantization = _normalize_quantization(config)

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
    if use_unsloth:
        model, tokenizer = build_lora_model(model, tokenizer, config)

    dataset = RoadBuddyVideoDataset(args.train_json, use_unsloth=use_unsloth)

    # Prepare SFT trainer
    if SFTTrainer is None:
        raise ImportError("trl not installed. Install with: pip install trl")

    if use_unsloth:
        try:
            from unsloth.trainer import UnslothVisionDataCollator
            data_collator = UnslothVisionDataCollator(model, tokenizer)
        except Exception:
            print("Falling back to simple collator; install unsloth_zoo for vision support.")
            data_collator = collate_unsloth
    else:
        data_collator = collate_unsloth

    training_args = get_training_args(args.output_dir, config)

    # Convert dataset to list of dict for SFTTrainer expected format
    hf_style_dataset = [ {"messages": sample.messages} for sample in dataset ]

    trainer = SFTTrainer(
        model=model,
        # tokenizer=tokenizer,
        train_dataset=hf_style_dataset,
        data_collator=data_collator,
        args=training_args,
    )

    print("Starting training...")
    train_result = trainer.train()
    print("Training complete. Saving LoRA adapters...")

    os.makedirs(args.output_dir, exist_ok=True)
    if use_unsloth:
        # Save only adapters
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
    else:
        # Save full model if not using Unsloth
        model.save_pretrained(args.output_dir)
        if tokenizer:
            tokenizer.save_pretrained(args.output_dir)

    print("Saved to", args.output_dir)
    print("Training metrics:", train_result)


if __name__ == "__main__":
    main()
