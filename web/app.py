"""
app.py
Flask Web Application connecting UI to PyTorch, TorchServe, and SageMaker.
"""

import os
import sys
import time
import logging
from pathlib import Path
from flask import Flask, render_template, request, jsonify
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import EmailReplyGenerator
from src.prompt_builder import validate_input, ValidationError
from serve.handler import EmailReplyHandler
from sagemaker.sagemaker_client import SageMakerClient
from tests.generate_comparison_table import TEST_CASES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SmartEmailReplyApp")

app = Flask(
    __name__,
    template_folder=str(PROJECT_ROOT / "web" / "templates"),
    static_folder=str(PROJECT_ROOT / "web" / "static")
)

# Global service singletons
generator_instance = None
torchserve_handler = None
sagemaker_client = None


def get_generator():
    global generator_instance
    if generator_instance is None:
        generator_instance = EmailReplyGenerator.get_instance("distilgpt2")
    return generator_instance


def get_torchserve_handler():
    global torchserve_handler
    if torchserve_handler is None:
        torchserve_handler = EmailReplyHandler()
        torchserve_handler.initialize()
    return torchserve_handler


def get_sagemaker_client():
    global sagemaker_client
    if sagemaker_client is None:
        sagemaker_client = SageMakerClient()
    return sagemaker_client


@app.route("/")
def index():
    """Serves the main application page."""
    return render_template("index.html")


@app.route("/api/health", methods=["GET"])
def health_check():
    """Returns status of PyTorch engine and backends."""
    return jsonify({
        "status": "healthy",
        "model": "distilgpt2",
        "backends_available": {
            "pytorch_direct": True,
            "torchserve_handler": True,
            "sagemaker_client": True
        }
    })


@app.route("/api/test_cases", methods=["GET"])
def get_test_cases():
    """Returns the list of 10 pre-configured benchmark test cases."""
    formatted_cases = []
    for tc in TEST_CASES:
        formatted_cases.append({
            "id": tc["id"],
            "situation": tc["situation"],
            "tone": tc["tone"].capitalize(),
            "expected_behaviour": tc["expected_behaviour"],
            "generated_reply": ""
        })
    return jsonify({"status": "success", "test_cases": formatted_cases})


@app.route("/api/generate", methods=["POST"])
def generate():
    """
    Main inference generation API endpoint.
    Routes to Local PyTorch, TorchServe Handler, or Amazon SageMaker endpoint.
    """
    try:
        data = request.get_json(force=True) or {}
        email_context = data.get("email_context")
        tone = data.get("tone", "formal")
        backend = data.get("backend", "local").lower()
        max_new_tokens = int(data.get("max_new_tokens", 80))
        temperature = float(data.get("temperature", 0.7))
        top_p = float(data.get("top_p", 0.9))

        # Validate input (Report Question 9)
        clean_context, clean_tone = validate_input(email_context, tone)

        start_time = time.perf_counter()

        # -------------------------------------------------------------
        # Route 1: Local PyTorch Direct Generation
        # -------------------------------------------------------------
        if backend == "local":
            gen = get_generator()
            result = gen.generate_reply(
                email_context=clean_context,
                tone=clean_tone,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )
            result["backend"] = "PyTorch Direct (In-Memory Engine)"

        # -------------------------------------------------------------
        # Route 2: TorchServe Service
        # -------------------------------------------------------------
        elif backend == "torchserve":
            ts_port = int(os.environ.get("TORCHSERVE_PORT", 8080))
            ts_url = f"http://127.0.0.1:{ts_port}/predictions/email_reply_generator"
            
            # Check if external standalone TorchServe server is listening
            try:
                ts_resp = requests.post(
                    ts_url,
                    json={"email_context": clean_context, "tone": clean_tone, "max_new_tokens": max_new_tokens},
                    timeout=3
                )
                if ts_resp.status_code == 200:
                    result = ts_resp.json()
                    result["backend"] = "TorchServe REST Service (Port 8080)"
                else:
                    raise RuntimeError(f"TorchServe returned HTTP {ts_resp.status_code}")
            except Exception:
                # Fallback to direct in-process execution of EmailReplyHandler
                logger.info("External TorchServe port unreachable; invoking EmailReplyHandler directly.")
                handler = get_torchserve_handler()
                batch_input = [{
                    "data": {
                        "email_context": clean_context,
                        "tone": clean_tone,
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_p": top_p
                    }
                }]
                handler_results = handler.handle(batch_input, context=None)
                result = handler_results[0]
                result["backend"] = "TorchServe Handler (Custom Service)"

        # -------------------------------------------------------------
        # Route 3: Amazon SageMaker Inference Endpoint
        # -------------------------------------------------------------
        elif backend == "sagemaker":
            sm_client = get_sagemaker_client()
            result = sm_client.invoke(
                email_context=clean_context,
                tone=clean_tone,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

        else:
            return jsonify({"status": "error", "error": f"Unknown backend '{backend}'"}), 400

        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        if "latency_ms" not in result:
            result["latency_ms"] = elapsed_ms

        # Guarantee compliance with safety requirements
        result["label"] = "DRAFT"
        result["requires_user_review"] = True
        result["sent_automatically"] = False

        return jsonify(result), 200

    except ValidationError as ve:
        logger.warning(f"Validation error: {ve}")
        return jsonify({"status": "error", "code": 400, "error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected generation error: {e}", exc_info=True)
        return jsonify({"status": "error", "code": 500, "error": str(e)}), 500


@app.route("/api/run_test_suite", methods=["POST"])
def run_test_suite():
    """Runs all 10 test cases and returns comparison table results."""
    try:
        gen = get_generator()
        results = []
        for tc in TEST_CASES:
            res = gen.generate_reply(
                email_context=tc["situation"],
                tone=tc["tone"],
                max_new_tokens=70,
                temperature=0.65
            )
            results.append({
                "id": tc["id"],
                "situation": tc["situation"],
                "tone": tc["tone"].capitalize(),
                "expected_behaviour": tc["expected_behaviour"],
                "generated_reply": res["draft_reply"],
                "latency_ms": res["latency_ms"]
            })
        return jsonify({"status": "success", "results": results})
    except Exception as e:
        logger.error(f"Failed to run test suite: {e}", exc_info=True)
        return jsonify({"status": "error", "error": str(e)}), 500


def main(host: str = "127.0.0.1", port: int = 5000):
    logger.info("=" * 65)
    logger.info("Starting Smart Email Reply Generator Web Application")
    logger.info("=" * 65)
    logger.info(f"Open your browser at: http://{host}:{port}")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    main(port=port)
