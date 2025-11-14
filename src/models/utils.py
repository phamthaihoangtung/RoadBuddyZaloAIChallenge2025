import os
from typing import Any, Dict, Optional, Tuple

import torch
from transformers import BitsAndBytesConfig

def _maybe_add_token(kwargs: dict, hf_token: Optional[str]):
    if hf_token:
        kwargs["token"] = hf_token


def _is_local_model_path(model_name: str) -> bool:
    """Check if model_name points to a local directory with config.json."""
    return (
        os.path.isdir(model_name)
        and os.path.isfile(os.path.join(model_name, "config.json"))
    )


def _parse_quantization_config(quantization: Optional[Dict]) -> Tuple[bool, str]:
    """Parse and normalize quantization configuration.
    
    Returns:
        Tuple of (is_enabled, mode) where mode is "4bit" or "8bit"
    """
    config = quantization or {}
    is_enabled = bool(config.get("enabled", False))
    mode = str(config.get("mode", "8bit")).lower()
    
    if is_enabled and mode not in ("4bit", "8bit"):
        print(f"Unknown quantization mode '{mode}', defaulting to 8bit.")
        mode = "8bit"
    
    return is_enabled, mode


def _build_bnb_config(is_enabled: bool, mode: str) -> Optional["BitsAndBytesConfig"]:
    """Build BitsAndBytesConfig for quantization."""
    from transformers import BitsAndBytesConfig
    
    if not is_enabled:
        return None
    
    if mode == "8bit":
        print("Loading model with 8-bit quantization...")
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
    elif mode == "4bit":
        print("Loading model with 4-bit quantization (nf4)...")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    
    return None


def _load_placeholder_model(model_name: str):
    """Load placeholder model for testing."""
    from models.placeholder_model import PlaceholderModel
    
    model = PlaceholderModel({"model_name": model_name})
    return model, None, None


def _load_unsloth_model(
    model_name: str,
    quantization_enabled: bool,
    quantization_mode: str,
    is_local: bool,
    hf_token: Optional[str],
    inference_mode: bool,
):
    """Load model using Unsloth FastVisionModel."""
    try:
        from unsloth import FastVisionModel
    except ImportError as e:
        raise ImportError(
            "Unsloth requested but not installed. Install with: pip install unsloth"
        ) from e
    
    print(f"Loading Unsloth FastVisionModel from {'local path' if is_local else 'hub'}: {model_name}")
    
    # Unsloth only supports 4-bit quantization
    load_in_4bit = quantization_enabled and (quantization_mode == "4bit")
    if quantization_enabled and quantization_mode != "4bit":
        print("Warning: Unsloth currently supports 4-bit. Ignoring non-4bit quantization request.")
    
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,
        use_gradient_checkpointing="unsloth",
        token=None if is_local else hf_token,
    )
    
    # Configure model for inference or training
    if inference_mode:
        FastVisionModel.for_inference(model)
    else:
        FastVisionModel.for_training(model)
    
    return model, None, tokenizer


def _load_transformers_model(
    model_name: str,
    attn_implementation: str,
    quantization_config: Optional["BitsAndBytesConfig"],
    is_local: bool,
    hf_token: Optional[str],
    inference_mode: bool,
):
    """Load model using standard Transformers library."""
    from transformers import AutoProcessor, AutoModelForCausalLM
    
    # Build model kwargs
    model_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
        "attn_implementation": attn_implementation,
    }
    
    if not is_local:
        _maybe_add_token(model_kwargs, hf_token)
    
    # Add quantization or dtype
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16
    
    # Load model and processor
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True,
        token=None if is_local else hf_token,
    )
    
    # Set model mode
    model.eval() if inference_mode else model.train()
    
    return model, processor, None


def load_model(
    model_name: str,
    attn_implementation: str,
    quantization: Optional[Dict] = None,
    use_unsloth: bool = False,
    hf_token: Optional[str] = None,
    inference_mode: bool = True,
):
    """Load model and processor/tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model ID or local path
        attn_implementation: Attention implementation (e.g., "flash_attention_2")
        quantization: Dict with keys:
            - enabled: bool
            - mode: "4bit" | "8bit"
        use_unsloth: Use Unsloth's FastVisionModel (only supports 4-bit)
        hf_token: HuggingFace API token (not used for local paths)
        inference_mode: If True, set model to eval mode; otherwise train mode

    Returns:
        Tuple of (model, processor, tokenizer)
    """
    # Check if loading from local directory
    is_local = _is_local_model_path(model_name)
    if is_local:
        print(f"Loading local model from: {model_name}")
    
    # Handle placeholder model
    if model_name == "placeholder":
        return _load_placeholder_model(model_name)
    
    # Parse quantization settings
    quant_enabled, quant_mode = _parse_quantization_config(quantization)
    
    # Load using Unsloth if requested
    if use_unsloth:
        return _load_unsloth_model(
            model_name, quant_enabled, quant_mode, is_local, hf_token, inference_mode
        )
    
    # Load using standard Transformers
    bnb_config = _build_bnb_config(quant_enabled, quant_mode)
    return _load_transformers_model(
        model_name, attn_implementation, bnb_config, is_local, hf_token, inference_mode
    )

def build_lora_model(base_model, tokenizer, config: Dict[str, Any]):
    if not config.get("use_unsloth", False):
        print("LoRA training only supported via Unsloth path currently.")
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