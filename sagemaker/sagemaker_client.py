"""
sagemaker_client.py
Client for invoking Amazon SageMaker real-time endpoint or local container simulation.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class SageMakerClient:
    """
    Client supporting both remote Amazon SageMaker runtime invocations
    and a local SageMaker container simulator executing inference.py.
    """

    def __init__(self, endpoint_name: Optional[str] = None, region_name: str = "us-east-1"):
        self.endpoint_name = endpoint_name or os.environ.get("SAGEMAKER_ENDPOINT_NAME", "distilgpt2-email-reply-generator")
        self.region_name = region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        self._sagemaker_runtime = None
        self._local_artifacts = None
        self._init_client()

    def _init_client(self):
        """Attempts to initialize boto3 sagemaker-runtime client."""
        try:
            import boto3
            session = boto3.Session(region_name=self.region_name)
            # Verify if credentials are configured
            creds = session.get_credentials()
            if creds and creds.access_key:
                self._sagemaker_runtime = session.client("sagemaker-runtime")
                logger.info(f"Initialized boto3 SageMaker runtime for region {self.region_name}")
            else:
                logger.info("No active AWS credentials found. Running in SageMaker Container Simulation mode.")
        except Exception as e:
            logger.info(f"AWS Boto3 initialization note: {e}. Running in SageMaker Container Simulation mode.")

    def _init_local_simulator(self):
        """Lazy loads inference.py model artifacts for local simulation."""
        if self._local_artifacts is None:
            from sagemaker.inference import model_fn
            logger.info("Loading DistilGPT2 weights for SageMaker local simulation...")
            self._local_artifacts = model_fn(".")

    def invoke(self, email_context: str, tone: str = "formal", **kwargs) -> Dict[str, Any]:
        """
        Sends prediction request to SageMaker endpoint or local container emulator.
        """
        payload = {
            "email_context": email_context,
            "tone": tone,
            "max_new_tokens": kwargs.get("max_new_tokens", 80),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }

        # Try remote SageMaker endpoint if client is active
        if self._sagemaker_runtime is not None:
            try:
                response = self._sagemaker_runtime.invoke_endpoint(
                    EndpointName=self.endpoint_name,
                    ContentType="application/json",
                    Accept="application/json",
                    Body=json.dumps(payload)
                )
                raw_body = response["Body"].read().decode("utf-8")
                result = json.loads(raw_body)
                result["backend"] = "Amazon SageMaker (Remote AWS Endpoint)"
                result["endpoint_name"] = self.endpoint_name
                return result
            except Exception as e:
                logger.warning(
                    f"Remote SageMaker call to '{self.endpoint_name}' failed ({e}). "
                    "Falling back to SageMaker Local Container Simulation."
                )

        # Local SageMaker Simulator (executes the exact inference.py lifecycle)
        self._init_local_simulator()
        from sagemaker.inference import input_fn, predict_fn, output_fn
        
        parsed_input = input_fn(json.dumps(payload), "application/json")
        prediction = predict_fn(parsed_input, self._local_artifacts)
        json_output, _ = output_fn(prediction, "application/json")
        result = json.loads(json_output)
        result["backend"] = "Amazon SageMaker (Local Container Simulation)"
        result["endpoint_name"] = self.endpoint_name
        return result
