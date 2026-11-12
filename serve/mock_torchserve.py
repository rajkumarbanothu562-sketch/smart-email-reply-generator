"""
mock_torchserve.py
Standalone HTTP service implementing the TorchServe REST API contract
using EmailReplyHandler directly.

Endpoints:
- GET  /ping                              (TorchServe Health Check)
- GET  /models                            (Model Inventory)
- POST /predictions/email_reply_generator (Inference Endpoint)
"""

import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from serve.handler import EmailReplyHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TorchServeServer")

app = Flask("TorchServeServer")

# Initialize the TorchServe Handler
handler = EmailReplyHandler()
handler.initialize()


@app.route("/ping", methods=["GET"])
def ping():
    """TorchServe standard health check endpoint."""
    return jsonify({
        "status": "Healthy",
        "service": "TorchServe (DistilGPT2 Email Reply Service)",
        "model_loaded": handler.initialized
    }), 200


@app.route("/models", methods=["GET"])
def models():
    """TorchServe model inventory endpoint."""
    return jsonify({
        "models": [
            {
                "modelName": "email_reply_generator",
                "modelUrl": "email_reply_generator.mar",
                "minWorkers": 1,
                "maxWorkers": 2,
                "batchSize": 1,
                "status": "READY"
            }
        ]
    }), 200


@app.route("/predictions/email_reply_generator", methods=["POST"])
def predict():
    """
    TorchServe inference endpoint. Accepts standard TorchServe JSON or form data.
    """
    try:
        if request.is_json:
            raw_payload = request.get_json(force=True)
        else:
            raw_payload = request.form.to_dict() or json.loads(request.get_data(as_text=True))

        # Format as TorchServe request batch
        ts_batch = [{"data": raw_payload}]
        
        # Invoke TorchServe handler
        results = handler.handle(ts_batch, context=None)
        
        if not results:
            return jsonify({"status": "error", "error": "No output from handler"}), 500
            
        first_result = results[0]
        if first_result.get("status") == "error" or "error" in first_result:
            code = first_result.get("code", 400)
            return jsonify(first_result), code

        return jsonify(first_result), 200

    except Exception as e:
        logger.error(f"Inference exception: {str(e)}", exc_info=True)
        return jsonify({"status": "error", "code": 500, "error": str(e)}), 500


def run_torchserve(port: int = 8080):
    logger.info(f"Starting TorchServe API Service on http://127.0.0.1:{port}")
    logger.info(f"Health check: http://127.0.0.1:{port}/ping")
    logger.info(f"Prediction:   POST http://127.0.0.1:{port}/predictions/email_reply_generator")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    run_torchserve(port)
