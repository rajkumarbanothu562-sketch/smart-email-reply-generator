"""
generator.py
Core text generation module using PyTorch and DistilGPT2.
"""

import time
from typing import Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.prompt_builder import build_prompt, postprocess_generated_reply, validate_input


class EmailReplyGenerator:
    """
    NLP Generation Engine for DistilGPT2 email drafting.
    """
    _instance: Optional["EmailReplyGenerator"] = None

    def __init__(self, model_name_or_path: str = "distilgpt2", device: Optional[str] = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name_or_path
        print(f"[Generator] Loading tokenizer for {model_name_or_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # Ensure pad token is set (DistilGPT2 doesn't have one by default)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"[Generator] Loading model {model_name_or_path} on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        self.model.eval()
        print("[Generator] Model initialized and ready for inference.")

    @classmethod
    def get_instance(cls, model_name_or_path: str = "distilgpt2") -> "EmailReplyGenerator":
        """Singleton accessor to avoid reallocating weights across requests."""
        if cls._instance is None:
            cls._instance = cls(model_name_or_path)
        return cls._instance

    def generate_reply(
        self,
        email_context: str,
        tone: str = "formal",
        max_new_tokens: int = 80,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.15
    ) -> Dict[str, Any]:
        """
        Generates an email reply draft from user context and tone.
        """
        # Validate inputs
        clean_context, clean_tone = validate_input(email_context, tone)
        
        # Build prompt
        prompt = build_prompt(clean_context, clean_tone)
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_token_count = inputs["input_ids"].shape[1]
        
        start_time = time.perf_counter()
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=max(0.1, min(temperature, 1.5)),
                top_p=max(0.1, min(top_p, 1.0)),
                repetition_penalty=repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        raw_output = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        draft_reply = postprocess_generated_reply(raw_output, prompt, clean_tone)
        
        return {
            "status": "success",
            "email_context": clean_context,
            "tone": clean_tone,
            "prompt": prompt,
            "raw_output": raw_output,
            "draft_reply": draft_reply,
            "is_draft": True,
            "requires_user_review": True,
            "latency_ms": elapsed_ms,
            "input_tokens": input_token_count,
            "output_tokens": output_tokens.shape[1],
            "model": self.model_name,
            "device": self.device
        }
