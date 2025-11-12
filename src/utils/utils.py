import yaml
import os
import torch
from typing import Callable, Dict, Optional, Tuple
import unsloth
from unsloth import FastVisionModel

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
    quantization: Optional[Dict] = None,
    use_unsloth: bool = False,
    hf_token: Optional[str] = None,
    inference_mode: bool = True,
):
    """Load model and processor/tokenizer with optional quantization.

    quantization: dict with keys:
      - enabled: bool
      - mode: "4bit" | "8bit"
    For Unsloth, only 4-bit is honored via load_in_4bit.
    """
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        BitsAndBytesConfig,
    )

    # Normalize quantization config
    q = quantization or {}
    q_enabled = bool(q.get("enabled", False))
    q_mode = str(q.get("mode", "8bit")).lower()
    if q_enabled and q_mode not in ("4bit", "8bit"):
        print(f"Unknown quantization mode '{q_mode}', defaulting to 8bit.")
        q_mode = "8bit"

    # Detect local directory (contains config.json)
    is_local_path = (
        os.path.isdir(model_name)
        and os.path.isfile(os.path.join(model_name, "config.json"))
    )
    if is_local_path:
        print(f"Loading local model from: {model_name}")
    # Determine model family
    is_smol = is_smolvlm2_model(model_name)

    if model_name == "placeholder":
        from utils.placeholder_model import PlaceholderModel

        model = PlaceholderModel({"model_name": model_name})
        processor = None
        tokenizer = None
        return model, processor, tokenizer

    if use_unsloth:
        # Unsloth path also supports local directories
        try:
            from unsloth import FastVisionModel
        except ImportError as e:
            raise ImportError(
                "Unsloth requested but not installed. Install with: pip install unsloth"
            ) from e

        print(
            f"Loading Unsloth FastVisionModel from {'local path' if is_local_path else 'hub'}: {model_name}"
        )
        load_in_4bit = q_enabled and (q_mode == "4bit")
        if q_enabled and q_mode != "4bit":
            print("Warning: Unsloth currently supports 4-bit. Ignoring non-4bit quantization request.")
        model, tokenizer = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit=load_in_4bit,
            use_gradient_checkpointing="unsloth",
            token=None if is_local_path else hf_token,
        )
        # Switch model behavior depending on mode
        if inference_mode:
            FastVisionModel.for_inference(model)
            # model.eval()
        else:
            # Enable training mode and set Unsloth's training optimizations
            FastVisionModel.for_training(model)
            # model.train()
        processor = None
        return model, processor, tokenizer

    # Helper to build BitsAndBytesConfig
    def _build_bnb() -> Optional[BitsAndBytesConfig]:
        if not q_enabled:
            return None
        if q_mode == "8bit":
            print("Loading model with 8-bit quantization...")
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        elif q_mode == "4bit":
            print("Loading model with 4-bit quantization (nf4)...")
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        return None

    if is_smol:
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "_attn_implementation": attn_implementation,
        }
        if not is_local_path:
            _maybe_add_token(model_kwargs, hf_token)

        quantization_config = _build_bnb()
        if quantization_config is not None:
            model_kwargs["quantization_config"] = quantization_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16
            print("Loading SmolVLM2 model without quantization...")

        model = AutoModelForImageTextToText.from_pretrained(model_name, **model_kwargs)
        processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=None if is_local_path else hf_token
        )
        tokenizer = None
        # Set mode
        model.eval() if inference_mode else model.train()
        return model, processor, tokenizer

    # Default causal LM family
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    if not is_local_path:
        _maybe_add_token(model_kwargs, hf_token)

    quantization_config = _build_bnb()
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True, token=None if is_local_path else hf_token
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
            # {
            #     "role": "system",
            #     "content": "You are a helpful assistant. \n"
            #     "Answer the multiple choice question based on the video provided. \n"
            #     "Please select one of the provided choices and respond with only the choice "
            #     "letter in A, B, C, or D.",
            # },
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
            text=input_text,
            images=image_input,
            videos=video_input,
            padding=True,
            return_tensors="pt",
            # **video_kwargs
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
