"""
inference.py
Standard Amazon SageMaker PyTorch / Hugging Face Inference Toolkit entrypoint.

Functions required by SageMaker PyTorch Container:
- model_fn(model_dir): Loads DistilGPT2 and tokenizer.
- input_fn(request_body, request_content_type): Parses & validates input.
- predict_fn(input_data, model_and_tokenizer): Runs PyTorch generation.
- output_fn(prediction, accept): Formats output response with DRAFT label.
"""

import os
import json
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_CONTEXT_CHARS = 1000
MIN_CONTEXT_CHARS = 3


def model_fn(model_dir: str):
    """
    Loads the model and tokenizer from model_dir into PyTorch evaluation mode.
    """
    logger.info(f"[SageMaker model_fn] Loading DistilGPT2 from {model_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if local model files exist in model_dir, else load distilgpt2
    model_source = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else "distilgpt2"
    
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.to(device)
    model.eval()
    
    logger.info(f"[SageMaker model_fn] Model successfully loaded on {device}")
    return {"model": model, "tokenizer": tokenizer, "device": device}


def input_fn(request_body: str, request_content_type: str = "application/json"):
    """
    Validates input payload and handles missing or excessively long input.
    """
    logger.info(f"[SageMaker input_fn] Content-Type: {request_content_type}")
    
    if request_content_type == "application/json":
        data = json.loads(request_body) if isinstance(request_body, (str, bytes)) else request_body
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}. Expected application/json.")

    # 1. Validation for Missing Input
    email_context = data.get("email_context")
    if email_context is None or not str(email_context).strip():
        raise ValueError("Missing input: 'email_context' cannot be empty.")

    email_context = str(email_context).strip()

    # 2. Validation for Too Short Input
    if len(email_context) < MIN_CONTEXT_CHARS:
        raise ValueError(f"Input too short: must be at least {MIN_CONTEXT_CHARS} characters.")

    # 3. Validation for Excessively Long Input (Report Question 9)
    if len(email_context) > MAX_CONTEXT_CHARS:
        raise ValueError(
            f"Excessively long input: Received {len(email_context)} characters. "
            f"Maximum limit is {MAX_CONTEXT_CHARS} characters."
        )

    # 4. Tone Validation
    tone = str(data.get("tone", "formal")).strip().lower()
    if tone not in ("formal", "friendly"):
        raise ValueError(f"Invalid tone '{tone}': Must be either 'formal' or 'friendly'.")

    return {
        "email_context": email_context,
        "tone": tone,
        "max_new_tokens": int(data.get("max_new_tokens", 80)),
        "temperature": float(data.get("temperature", 0.7)),
        "top_p": float(data.get("top_p", 0.9))
    }


def predict_fn(input_data: dict, model_artifacts: dict):
    """
    Runs text generation using DistilGPT2.
    """
    model = model_artifacts["model"]
    tokenizer = model_artifacts["tokenizer"]
    device = model_artifacts["device"]
    
    email_context = input_data["email_context"]
    tone = input_data["tone"]
    
    # Construct tone-specific prompt
    if tone == "formal":
        prompt = (
            "Task: Generate a polite and professional formal email reply based on the situation.\n\n"
            "Example 1:\n"
            "Situation: Requesting leave for one day\n"
            "Tone: Formal\n"
            "Draft Reply:\n"
            "Dear Manager,\n\n"
            "I am writing to formally request one day of leave on Friday due to personal matters. "
            "I will ensure all pressing tasks are completed before my departure.\n\n"
            "Sincerely,\n[Your Name]\n\n"
            f"Situation: {email_context}\n"
            "Tone: Formal\n"
            "Draft Reply:\n"
            "Dear "
        )
    else:
        prompt = (
            "Task: Generate a warm and friendly email reply based on the situation.\n\n"
            "Example 1:\n"
            "Situation: Thanking a colleague for help\n"
            "Tone: Friendly\n"
            "Draft Reply:\n"
            "Hi Alex,\n\n"
            "Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.\n\n"
            "Cheers,\n[Your Name]\n\n"
            f"Situation: {email_context}\n"
            "Tone: Friendly\n"
            "Draft Reply:\n"
            "Hi "
        )

    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **tokens,
            max_new_tokens=max(20, min(input_data["max_new_tokens"], 150)),
            temperature=max(0.1, min(input_data["temperature"], 1.5)),
            top_p=max(0.1, min(input_data["top_p"], 1.0)),
            repetition_penalty=1.15,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Postprocess output
    if decoded.startswith(prompt):
        continuation = decoded[len(prompt):]
    else:
        parts = decoded.split("Draft Reply:\n")
        continuation = parts[-1] if len(parts) > 1 else decoded

    salutation = "Dear " if tone == "formal" else "Hi "
    draft = (salutation + continuation).strip()
    
    for stop in ["\nSituation:", "Situation:", "Example", "Task:"]:
        if stop in draft:
            draft = draft.split(stop)[0].strip()

    has_closing = any(
        c in draft.lower() 
        for c in ["sincerely", "regards", "best regards", "cheers", "talk soon", "thanks"]
    )
    if not has_closing:
        closing = "\n\nBest regards,\n[Your Name]" if tone == "formal" else "\n\nCheers,\n[Your Name]"
        draft += closing

    return {
        "status": "success",
        "email_context": email_context,
        "tone": tone,
        "prompt": prompt,
        "draft_reply": draft,
        "label": "DRAFT",
        "requires_user_review": True,
        "sent_automatically": False,
        "endpoint_source": "Amazon SageMaker"
    }


def output_fn(prediction: dict, accept: str = "application/json"):
    """
    Formats the SageMaker output JSON response.
    """
    if accept == "application/json" or accept == "*/*":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
