"""
prompt_builder.py
Prompt construction, input validation, and text post-processing for DistilGPT2.
"""

from typing import Dict, Any, Tuple
import re

MAX_INPUT_LENGTH = 1000  # Maximum allowed characters for input context
MIN_INPUT_LENGTH = 3     # Minimum required characters


class ValidationError(ValueError):
    """Custom exception raised when input validation fails."""
    pass


def validate_input(email_context: Any, tone: Any) -> Tuple[str, str]:
    """
    Validates missing, empty, invalid, or excessively long user input.
    
    Args:
        email_context: The incoming message or situation entered by the user.
        tone: The selected tone ("formal" or "friendly").
        
    Returns:
        Tuple of sanitized (email_context, tone).
        
    Raises:
        ValidationError: If input is missing, empty, or exceeds length limits.
    """
    # 1. Check missing or non-string input
    if email_context is None:
        raise ValidationError("Missing input: 'email_context' field is required.")
    
    if not isinstance(email_context, str):
        raise ValidationError("Invalid input type: 'email_context' must be a string.")
    
    # 2. Check empty or whitespace-only input
    sanitized_context = email_context.strip()
    if len(sanitized_context) < MIN_INPUT_LENGTH:
        raise ValidationError(
            f"Input too short: 'email_context' must be at least {MIN_INPUT_LENGTH} characters."
        )
    
    # 3. Check excessively long input
    if len(sanitized_context) > MAX_INPUT_LENGTH:
        raise ValidationError(
            f"Input exceeds maximum limit: received {len(sanitized_context)} characters; "
            f"maximum allowed is {MAX_INPUT_LENGTH} characters."
        )
    
    # 4. Check and sanitize tone
    if not tone or not isinstance(tone, str):
        sanitized_tone = "formal"  # Default fallback
    else:
        norm_tone = tone.strip().lower()
        if norm_tone in ("formal", "friendly"):
            sanitized_tone = norm_tone
        else:
            raise ValidationError(
                f"Invalid tone '{tone}': must be either 'formal' or 'friendly'."
            )
            
    return sanitized_context, sanitized_tone


def build_prompt(email_context: str, tone: str) -> str:
    """
    Constructs an optimized prompt for DistilGPT2 based on the selected tone.
    
    DistilGPT2 is a causal language model; framing with tone-specific
    exemplars and a deterministic seed salutation primes the generation
    to maintain tone fidelity, correct vocabulary, and email formatting.
    """
    context, validated_tone = validate_input(email_context, tone)
    
    if validated_tone == "formal":
        # Formal priming: Polite, professional, polite salutation, business phrasing
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
            "Example 2:\n"
            "Situation: Unable to attend quarterly review meeting\n"
            "Tone: Formal\n"
            "Draft Reply:\n"
            "Dear Colleagues,\n\n"
            "Thank you for the meeting invitation. Regrettably, I will be unable to attend "
            "the upcoming session due to a prior commitment. Please share the meeting minutes at your convenience.\n\n"
            "Best regards,\n[Your Name]\n\n"
            f"Situation: {context}\n"
            "Tone: Formal\n"
            "Draft Reply:\n"
            "Dear "
        )
    else:
        # Friendly priming: Conversational, warm, enthusiastic, casual salutation
        prompt = (
            "Task: Generate a warm and friendly email reply based on the situation.\n\n"
            "Example 1:\n"
            "Situation: Thanking a colleague for help\n"
            "Tone: Friendly\n"
            "Draft Reply:\n"
            "Hi Alex,\n\n"
            "Thanks so much for lending a hand with the presentation today! I really appreciate your time and support.\n\n"
            "Cheers,\n[Your Name]\n\n"
            "Example 2:\n"
            "Situation: Grabbing lunch next Tuesday\n"
            "Tone: Friendly\n"
            "Draft Reply:\n"
            "Hey Chris,\n\n"
            "Sounds awesome! I'd love to catch up over lunch next Tuesday. Let me know what time works best for you.\n\n"
            "Talk soon,\n[Your Name]\n\n"
            f"Situation: {context}\n"
            "Tone: Friendly\n"
            "Draft Reply:\n"
            "Hi "
        )
        
    return prompt


def postprocess_generated_reply(raw_generated_text: str, prompt: str, tone: str) -> str:
    """
    Cleans raw generated text from DistilGPT2:
    - Extracts the text following the prompt's 'Draft Reply:' section
    - Restores the seed salutation ("Dear " or "Hi ")
    - Strips subsequent prompt artifacts or next-example leakages
    - Ensures a clean closing and sign-off
    """
    # 1. Remove the prompt prefix
    if raw_generated_text.startswith(prompt):
        continuation = raw_generated_text[len(prompt):]
    else:
        # Fallback split
        parts = raw_generated_text.split("Draft Reply:\n")
        continuation = parts[-1] if len(parts) > 1 else raw_generated_text

    # Re-attach the primed salutation prefix
    salutation_prefix = "Dear " if tone.lower() == "formal" else "Hi "
    draft = salutation_prefix + continuation
    
    # Clean stop tokens
    for stop_token in ["\nSituation:", "Situation:", "Example", "Task:", "Tone:"]:
        if stop_token in draft:
            draft = draft.split(stop_token)[0]

    # Sanitize unusual unicode direction/control characters
    draft = re.sub(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", draft)

    # 3. Clean up sentences: trim dangling trailing incomplete sentences
    draft = draft.strip()
    
    # Check if ends with punctuation (. ! ?)
    if draft and draft[-1] not in (".", "!", "?"):
        # Find last sentence-ending punctuation
        match = re.search(r"^(.*[.!?])", draft, re.DOTALL)
        if match:
            draft = match.group(1).strip()
            
    # 4. Ensure an appropriate closing exists if model didn't emit one
    has_closing = any(
        closing in draft.lower() 
        for closing in ["sincerely", "regards", "best regards", "cheers", "talk soon", "thanks", "warm regards"]
    )
    if not has_closing:
        default_closing = "\n\nBest regards,\n[Your Name]" if tone.lower() == "formal" else "\n\nCheers,\n[Your Name]"
        draft = draft + default_closing
        
    return draft.strip()
