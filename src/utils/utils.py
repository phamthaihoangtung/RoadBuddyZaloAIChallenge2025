import yaml
import os
import torch
from typing import Callable, Dict, Optional, Tuple

# Lazy imports for transformers inside functions to keep this module light when unavailable


def load_config(path: str):
    """Load a YAML configuration file and return it as a dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def is_smolvlm2_model(model_name: str) -> bool:
    """Heuristic to detect SmolVLM2 family."""
    return "smolvlm2" in (model_name or "").lower()


def _maybe_add_token(kwargs: dict, hf_token: Optional[str]):
    if hf_token:
        kwargs["token"] = hf_token


def load_model(
    model_name: str,
    attn_implementation: str,
    use_quantization: bool,
    use_unsloth: bool,
    hf_token: Optional[str],
    inference_mode: bool = True,
):
    """Load and initialize the model and processor/tokenizer for inference or training.

    Returns a tuple of (model, processor, tokenizer). Exactly one of processor/tokenizer
    may be None depending on the model family.
    """
    # Deferred imports to avoid forcing transformers dependency where not needed
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    # Determine model family
    is_smol = is_smolvlm2_model(model_name)

    if model_name == "placeholder":
        from utils.placeholder_model import PlaceholderModel

        model = PlaceholderModel({"model_name": model_name})
        processor = None
        tokenizer = None
        return model, processor, tokenizer

    if use_unsloth:
        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "Unsloth requested but not installed. Install with: pip install unsloth"
            ) from e

        print(f"Loading Unsloth FastVisionModel from {model_name}...")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=use_quantization,
            use_gradient_checkpointing="unsloth",
            token=hf_token,
        )
        # Switch model behavior depending on mode
        if inference_mode:
            FastVisionModel.for_inference(model)
            model.eval()
        else:
            # Enable training mode and set Unsloth's training optimizations
            FastVisionModel.for_training(model)
            model.train()
        processor = None
        return model, processor, tokenizer

    # Shared kwargs for transformers models
    if is_smol:
        # SmolVLM2 uses AutoModelForImageTextToText
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            # Some SmolVLM2 repos expect this private argument; keep as-is from original code
            "_attn_implementation": attn_implementation,
        }
        _maybe_add_token(model_kwargs, hf_token)

        if use_quantization:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
            model_kwargs["quantization_config"] = quantization_config
            print("Loading SmolVLM2 model with 8-bit quantization...")
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Loading SmolVLM2 model...")

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=hf_token
        )
        tokenizer = None
        # Set mode
        model.eval() if inference_mode else model.train()
        return model, processor, tokenizer

    # Default family: causal LM with vision processor
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    _maybe_add_token(model_kwargs, hf_token)

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

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, token=hf_token
    )
    tokenizer = None
    # Set mode
    model.eval() if inference_mode else model.train()
    return model, processor, tokenizer


def format_mcq_question(question: str, choices: Optional[list]) -> str:
    """Combine question and choices into a single instruction-friendly string."""
    base_instr = (
        "Answer the multiple choice question based on the video provided. "
        "Please select one of the provided choices and respond with only the "
        "choice letter in A, B, C, or D."
    )
    if choices:
        return f"{base_instr}\n\n{question} {' '.join(choices)}"
    return f"{base_instr}\n\n{question}"


def preprocess_for_inference(
    model,
    processor,
    tokenizer,
    video_path: str,
    question_text: str,
    model_name: str,
    use_unsloth: bool,
) -> Tuple[Dict[str, torch.Tensor], Callable[[torch.Tensor], str]]:
    """Create tokenized inputs for generation and a decode function.

    Returns (inputs, decode_fn). inputs are on the correct device with proper dtype.
    decode_fn takes output_ids and returns a string.
    """
    is_smol = is_smolvlm2_model(model_name)

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

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. \n"
                "Answer the multiple choice question based on the video provided. \n"
                "Please select one of the provided choices and respond with only the choice "
                "letter in A, B, C, or D.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": video_path},
                    {"type": "text", "text": question_text},
                ],
            },
        ]

        image_input, video_input = vision_utils.process_vision_info(messages)
        input_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(
            input_text,
            images=image_input,
            videos=video_input,
            return_tensors="pt",
        ).to(model.device)

        def _decode(output_ids: torch.Tensor) -> str:
            return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return inputs, _decode

    if is_smol:
        # SmolVLM2 expects messages + apply_chat_template via processor
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_path},
                    {"type": "text", "text": question_text},
                ],
            }
        ]
        inputs = (
            processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            .to(model.device, dtype=torch.bfloat16)
        )

        def _decode(output_ids: torch.Tensor) -> str:
            return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        return inputs, _decode

    # Default processor conversation format
    conversation = [
        {
            "role": "system",
            "content": "You are a helpful assistant. "
            "Answer the multiple choice question based on the video provided."
            "Please select one of the provided choices and respond with only the choice "
            "letter in A, B, C, or D.",
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": {"video_path": video_path, "fps": 1, "max_frames": 128},
                },
                {"type": "text", "text": question_text},
            ],
        },
    ]
    inputs = processor(conversation=conversation, return_tensors="pt")
    # Move to device and set dtypes
    inputs = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
    if "pixel_values" in inputs and isinstance(inputs["pixel_values"], torch.Tensor):
        inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

    def _decode(output_ids: torch.Tensor) -> str:
        return processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return inputs, _decode
