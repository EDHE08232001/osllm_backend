"""
ai.py - model loading, response generation, and conversation helpers.

This module loads runtime settings from config.yml and exposes helpers for
model initialization, inference, validation, and conversation context building.
"""

from transformers import AutoProcessor, AutoModelForCausalLM
import torch
import logging
import yaml
import os

# ============================================================================
# SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# Load config at module level
config_path = os.path.join(os.path.dirname(__file__), "config.yml")

try:
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
except FileNotFoundError:
    logger.error(f"config.yml not found at {config_path}")
    raise
except yaml.YAMLError as e:
    logger.error(f"Invalid YAML in config.yml: {e}")
    raise

# Get config values
MODEL_ID = config["model"]["id"]
MODEL_DTYPE = config["model"]["dtype"]
DEVICE_MAP = config["model"].get("device_map", "auto")
MAX_TOKENS = config["inference"].get("max_tokens", 1024)
TEMPERATURE = config["inference"]["temperature"]
TOP_P = config["inference"].get("top_p", 0.9)
TOP_K = config["inference"].get("top_k", 0)

# ============================================================================
# GLOBAL MODEL STATE
# ============================================================================

_model = None
_processor = None
_model_loaded = False

# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model():
    """Load the Hugging Face processor and causal language model once.

    Raises:
        RuntimeError: If loading the model or processor fails.
    """
    global _model, _processor, _model_loaded

    if _model_loaded:
        logger.info("Model already loaded")
        return

    logger.info(f"Loading model '{MODEL_ID}' with dtype '{MODEL_DTYPE}' on device map '{DEVICE_MAP}'...")

    try:
        logger.info(f"Loading processor from: {MODEL_ID}")
        _processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True
        )

        logger.info(f"Loading model from: {MODEL_ID}")
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            dtype=MODEL_DTYPE,  # ✅ FIXED: was torch_dtype, should be dtype
            device_map=DEVICE_MAP,
            trust_remote_code=config["model"].get("trust_remote_code", True)
        )

        _model_loaded = True
        logger.info("✓ Model loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model '{MODEL_ID}'") from e

# ============================================================================
# MODEL INFERENCE
# ============================================================================

def generate_response(messages: list, max_tokens: int = None) -> str:
    """Generate a model response from a chat-style message list.

    Args:
        messages: Chat messages in Hugging Face chat template format.
        max_tokens: Optional override for the maximum number of new tokens.

    Returns:
        The decoded model response string.

    Raises:
        RuntimeError: If generation fails.
    """
    global _model, _processor

    if not _model_loaded:
        load_model()

    tokens = max_tokens if max_tokens is not None else MAX_TOKENS

    try:
        # Apply chat template
        text = _processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        # Tokenize
        inputs = _processor(text, return_tensors="pt").to(_model.device)
        input_len = inputs["input_ids"].shape[-1]  # ✅ FIXED: was shape[1], now shape[-1] for flexibility

        logger.info(f"Input token length: {input_len}")
        logger.info(f"Generating response with max tokens: {tokens}")

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": tokens,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        }

        # Add top_k only if it's > 0
        if TOP_K > 0:  # ✅ FIXED: Better handling of top_k
            gen_kwargs["top_k"] = TOP_K

        # Generate response
        outputs = _model.generate(**inputs, **gen_kwargs)

        # Decode response
        response = _processor.decode(
            outputs[0][input_len:],
            skip_special_tokens=True
        )

        logger.info(f"Response generated successfully ({len(response)} chars)")
        return response

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise RuntimeError("Failed to generate response") from e

# ============================================================================
# INPUT VALIDATION
# ============================================================================

def validate_message(message: str) -> tuple[bool, str]:
    """Validate a user message against the configured length limits.

    Args:
        message: The incoming user message.

    Returns:
        A tuple containing a success flag and a human-readable message.
    """
    max_len = config["conversation"].get("max_message_length", 2000)
    min_len = config["conversation"].get("min_message_length", 1)

    if not message:
        return False, "Message cannot be empty."

    if not isinstance(message, str):
        return False, "Message must be a string."

    if len(message) > max_len:
        return False, f"Message exceeds maximum length of {max_len} characters."

    if len(message) < min_len:
        return False, f"Message is shorter than minimum length of {min_len} characters."

    return True, "Message is valid."

# ============================================================================
# RESPONSE CLEANING
# ============================================================================

def clean_response(response: str) -> str:
    """Normalize whitespace in a generated response.

    Args:
        response: Raw text returned by the model.

    Returns:
        The response with repeated whitespace collapsed to single spaces.
    """
    response = " ".join(response.split())
    return response

# ============================================================================
# CONVERSATION CONTEXT
# ============================================================================

def build_conversation_context(user_message: str, history: list) -> list:
    """Append the current user message to a copy of the conversation history.

    Args:
        user_message: The latest user message.
        history: Existing chat messages to preserve.

    Returns:
        A new list containing the prior history plus the new user message.
    """
    context = history.copy() if history else []
    context.append({"role": "user", "content": user_message})
    return context

# ============================================================================
# DEVICE CHECK (Optional)
# ============================================================================

def check_device():
    """Log the current device configuration for debugging."""
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Current device: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available. Using CPU.")