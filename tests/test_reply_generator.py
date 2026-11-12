"""
test_reply_generator.py
Unit and integration tests for Smart Email Reply Generator.
Tests prompt construction, validation of missing and excessively long inputs,
TorchServe handler lifecycle, and SageMaker inference pipeline.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompt_builder import validate_input, build_prompt, postprocess_generated_reply, ValidationError
from src.generator import EmailReplyGenerator
from serve.handler import EmailReplyHandler
from sagemaker.inference import input_fn, predict_fn, output_fn


class TestEmailReplyGenerator(unittest.TestCase):

    # -------------------------------------------------------------
    # 1. Input Validation Tests (Report Question 9)
    # -------------------------------------------------------------
    def test_validation_missing_input_none(self):
        """Verifies handler and builder reject None context."""
        with self.assertRaises(ValidationError) as ctx:
            validate_input(None, "formal")
        self.assertIn("Missing input", str(ctx.exception))

    def test_validation_missing_input_empty(self):
        """Verifies handler and builder reject empty string context."""
        with self.assertRaises(ValidationError) as ctx:
            validate_input("   ", "formal")
        self.assertIn("Input too short", str(ctx.exception))

    def test_validation_excessively_long_input(self):
        """Verifies handler and builder reject input exceeding 1000 characters."""
        long_context = "This is a very long email context message. " * 30  # ~1290 chars
        self.assertGreater(len(long_context), 1000)
        with self.assertRaises(ValidationError) as ctx:
            validate_input(long_context, "formal")
        self.assertIn("exceeds maximum limit", str(ctx.exception))

    def test_validation_invalid_tone(self):
        """Verifies handler rejects unsupported tone types."""
        with self.assertRaises(ValidationError) as ctx:
            validate_input("Valid meeting context", "aggressive")
        self.assertIn("Invalid tone", str(ctx.exception))

    def test_validation_valid_inputs(self):
        """Verifies clean context and tone pass validation."""
        context, tone = validate_input("  Unable to attend meeting  ", "Friendly  ")
        self.assertEqual(context, "Unable to attend meeting")
        self.assertEqual(tone, "friendly")

    # -------------------------------------------------------------
    # 2. Prompt Construction Tests (Report Question 7)
    # -------------------------------------------------------------
    def test_prompt_construction_formal(self):
        """Verifies formal prompt contains polite formal instructions and salutation."""
        prompt = build_prompt("I have a doctor appointment.", "formal")
        self.assertIn("polite and professional formal email reply", prompt)
        self.assertIn("Tone: Formal", prompt)
        self.assertTrue(prompt.endswith("Draft Reply:\nDear "))

    def test_prompt_construction_friendly(self):
        """Verifies friendly prompt contains casual instructions and greeting."""
        prompt = build_prompt("I have a doctor appointment.", "friendly")
        self.assertIn("warm and friendly email reply", prompt)
        self.assertIn("Tone: Friendly", prompt)
        self.assertTrue(prompt.endswith("Draft Reply:\nHi "))

    # -------------------------------------------------------------
    # 3. TorchServe Handler Tests
    # -------------------------------------------------------------
    def test_torchserve_handler_validation(self):
        """Tests TorchServe handler input validation error responses."""
        handler = EmailReplyHandler()
        # Missing context test
        bad_req = [{"data": {"email_context": "", "tone": "formal"}}]
        resp = handler.handle(bad_req, context=None)
        self.assertEqual(resp[0].get("code"), 400)
        self.assertIn("Missing input", resp[0].get("error"))

        # Excessively long context test
        long_req = [{"data": {"email_context": "x" * 1200, "tone": "formal"}}]
        resp2 = handler.handle(long_req, context=None)
        self.assertEqual(resp2[0].get("code"), 400)
        self.assertIn("Excessively long input", resp2[0].get("error"))

    def test_torchserve_handler_end_to_end(self):
        """Tests TorchServe handler end-to-end inference and draft labelling."""
        handler = EmailReplyHandler()
        handler.initialize()
        req = [{"data": {"email_context": "Unable to attend tomorrow's sync meeting", "tone": "friendly"}}]
        resp = handler.handle(req, context=None)
        self.assertEqual(len(resp), 1)
        item = resp[0]
        self.assertEqual(item.get("status"), "success")
        self.assertEqual(item.get("label"), "DRAFT")
        self.assertTrue(item.get("requires_user_review"))
        self.assertFalse(item.get("sent_automatically"))
        self.assertTrue(len(item.get("draft_reply")) > 10)

    # -------------------------------------------------------------
    # 4. SageMaker Pipeline Tests
    # -------------------------------------------------------------
    def test_sagemaker_inference_pipeline(self):
        """Tests SageMaker input_fn and predict_fn functions."""
        from sagemaker.inference import model_fn
        artifacts = model_fn(".")
        input_json = '{"email_context": "Thanking team for project release", "tone": "friendly"}'
        parsed = input_fn(input_json, "application/json")
        pred = predict_fn(parsed, artifacts)
        out_str, content_type = output_fn(pred, "application/json")
        self.assertEqual(content_type, "application/json")
        self.assertIn("draft_reply", out_str)
        self.assertIn("DRAFT", out_str)


if __name__ == "__main__":
    unittest.main()
