"""
handler.py
Custom TorchServe Handler for Smart Email Reply Generation with DistilGPT2.

Follows official TorchServe Custom Service design:
- initialize(context)
- preprocess(data)
- inference(model_inputs)
- postprocess(inference_output)
- handle(data, context)
"""

import os
import json
import logging
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# Length thresholds for input validation (addresses Report Question 9)
MAX_CONTEXT_CHARS = 1000
MIN_CONTEXT_CHARS = 3


class EmailReplyHandler:
    """
    TorchServe custom handler for DistilGPT2 smart email reply generation.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.initialized = False

    def initialize(self, context=None):
        """
        Initializes the model and tokenizer from the model directory or Hugging Face.
        """
        properties = context.system_properties if context else {}
        gpu_id = properties.get("gpu_id")
        model_dir = properties.get("model_dir", ".")
        
        self.device = (
            torch.device(f"cuda:{gpu_id}")
            if torch.cuda.is_available() and gpu_id is not None
            else torch.device("cpu")
        )
        logger.info(f"Initializing EmailReplyHandler on device: {self.device}")

        # Check if local model weights exist in model_dir, otherwise fall back to 'distilgpt2'
        model_path = model_dir if os.path.exists(os.path.join(model_dir, "config.json")) else "distilgpt2"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True
        logger.info("DistilGPT2 model and tokenizer successfully loaded in TorchServe handler.")

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates missing, invalid, or excessively long input and builds prompts.
        Addresses Functional Requirements and Report Question 9.
        """
        processed_batch = []
        
        for item in data:
            # Handle TorchServe input payloads (can be bytearray, string, or dict)
            payload = item.get("data") or item.get("body")
            if isinstance(payload, (bytes, bytearray)):
                try:
                    payload = json.loads(payload.decode("utf-8"))
                except Exception as e:
                    raise ValueError(f"Malformed JSON payload: {str(e)}")
            elif isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    # Raw string passed as email context directly
                    payload = {"email_context": payload, "tone": "formal"}
            elif not isinstance(payload, dict):
                payload = {}

            # 1. Validation for Missing Input
            email_context = payload.get("email_context")
            if email_context is None or not str(email_context).strip():
                raise ValueError(
                    "Missing input validation error: 'email_context' is required and cannot be empty."
                )

            email_context = str(email_context).strip()
            
            # 2. Validation for Too Short Input
            if len(email_context) < MIN_CONTEXT_CHARS:
                raise ValueError(
                    f"Input too short: 'email_context' must have at least {MIN_CONTEXT_CHARS} characters."
                )

            # 3. Validation for Excessively Long Input (Report Question 9)
            if len(email_context) > MAX_CONTEXT_CHARS:
                raise ValueError(
                    f"Excessively long input validation error: Received {len(email_context)} characters. "
                    f"Maximum allowed limit is {MAX_CONTEXT_CHARS} characters."
                )

            # 4. Tone Validation
            raw_tone = payload.get("tone", "formal")
            tone = str(raw_tone).strip().lower()
            if tone not in ("formal", "friendly"):
                raise ValueError(
                    f"Invalid tone '{raw_tone}': Must be either 'formal' or 'friendly'."
                )

            # 5. Generation hyperparameter extraction with safe bounds
            max_new_tokens = int(payload.get("max_new_tokens", 80))
            max_new_tokens = max(20, min(max_new_tokens, 200))
            
            temperature = float(payload.get("temperature", 0.7))
            temperature = max(0.1, min(temperature, 1.5))
            
            top_p = float(payload.get("top_p", 0.9))
            top_p = max(0.1, min(top_p, 1.0))

            # 6. Prompt Construction
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

            tokens = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            processed_batch.append({
                "tokens": tokens,
                "prompt": prompt,
                "email_context": email_context,
                "tone": tone,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p
            })

        return processed_batch

    def inference(self, model_inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Runs model forward generation for each preprocessed item.
        """
        results = []
        with torch.no_grad():
            for item in model_inputs:
                tokens = item["tokens"]
                outputs = self.model.generate(
                    **tokens,
                    max_new_tokens=item["max_new_tokens"],
                    temperature=item["temperature"],
                    top_p=item["top_p"],
                    repetition_penalty=1.15,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                results.append({
                    "raw_output": decoded,
                    "prompt": item["prompt"],
                    "email_context": item["email_context"],
                    "tone": item["tone"]
                })
        return results

    def postprocess(self, inference_output: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Cleans generated reply and formats the TorchServe response payload.
        Ensures the result is prominently labelled as a draft requiring user review.
        """
        responses = []
        for item in inference_output:
            raw = item["raw_output"]
            prompt = item["prompt"]
            tone = item["tone"]

            # Remove prompt
            if raw.startswith(prompt):
                continuation = raw[len(prompt):]
            else:
                parts = raw.split("Draft Reply:\n")
                continuation = parts[-1] if len(parts) > 1 else raw

            salutation = "Dear " if tone == "formal" else "Hi "
            draft = (salutation + continuation).strip()

            # Clean stop tokens
            for stop in ["\nSituation:", "Situation:", "Example", "Task:"]:
                if stop in draft:
                    draft = draft.split(stop)[0].strip()

            # Add closing sign-off if missing
            has_closing = any(
                c in draft.lower() 
                for c in ["sincerely", "regards", "best regards", "cheers", "talk soon", "thanks"]
            )
            if not has_closing:
                closing = "\n\nBest regards,\n[Your Name]" if tone == "formal" else "\n\nCheers,\n[Your Name]"
                draft += closing

            responses.append({
                "status": "success",
                "email_context": item["email_context"],
                "tone": tone,
                "prompt_used": prompt,
                "draft_reply": draft,
                "label": "DRAFT",
                "requires_user_review": True,
                "sent_automatically": False,
                "disclaimer": "This draft email requires human review and editing. The application does not send emails automatically."
            })
        return responses

    def handle(self, data, context):
        """
        Entry point invoked by TorchServe engine.
        """
        if not self.initialized:
            self.initialize(context)

        if data is None:
            return [{"code": 400, "error": "Empty request data received."}]

        try:
            model_inputs = self.preprocess(data)
            inference_out = self.inference(model_inputs)
            return self.postprocess(inference_out)
        except ValueError as ve:
            logger.warning(f"Validation error in handler: {str(ve)}")
            return [{"status": "error", "code": 400, "error": str(ve)}]
        except Exception as e:
            logger.error(f"Unexpected error in handler: {str(e)}", exc_info=True)
            return [{"status": "error", "code": 500, "error": f"Internal inference error: {str(e)}"}]
