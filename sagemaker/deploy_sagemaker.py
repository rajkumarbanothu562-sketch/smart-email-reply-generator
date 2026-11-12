"""
deploy_sagemaker.py
Automates packaging DistilGPT2 and inference code into model.tar.gz,
and deploying to an Amazon SageMaker real-time inference endpoint.
"""

import os
import sys
import tarfile
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def package_model_artifact(output_path: Path) -> Path:
    """
    Creates model.tar.gz containing code/inference.py for SageMaker PyTorch container.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    tar_gz_path = output_path / "model.tar.gz"
    inference_script = PROJECT_ROOT / "sagemaker" / "inference.py"
    requirements_file = PROJECT_ROOT / "requirements.txt"

    print(f"[*] Packaging SageMaker model artifact to: {tar_gz_path}")
    with tarfile.open(tar_gz_path, "w:gz") as tar:
        # Add inference script under code/
        tar.add(inference_script, arcname="code/inference.py")
        if requirements_file.exists():
            tar.add(requirements_file, arcname="code/requirements.txt")
            
    print(f"[OK] Successfully created: {tar_gz_path} ({tar_gz_path.stat().st_size} bytes)")
    return tar_gz_path


def deploy_to_sagemaker(
    role_arn: str,
    bucket_name: str,
    endpoint_name: str = "distilgpt2-email-reply-generator",
    instance_type: str = "ml.m5.large"
):
    """
    Deploys the model artifact to AWS SageMaker using boto3 / sagemaker SDK.
    """
    try:
        import boto3
        import sagemaker
        from sagemaker.pytorch import PyTorchModel
    except ImportError as e:
        print(f"[ERROR] Required packages not found: {e}")
        print("Run: pip install boto3 sagemaker")
        sys.exit(1)

    session = boto3.Session()
    s3_client = session.client("s3")
    
    # 1. Package
    artifact_dir = PROJECT_ROOT / "sagemaker" / "dist"
    tar_path = package_model_artifact(artifact_dir)

    # 2. Upload to S3
    s3_key = f"models/{endpoint_name}/model.tar.gz"
    print(f"[*] Uploading artifact to s3://{bucket_name}/{s3_key}...")
    s3_client.upload_file(str(tar_path), bucket_name, s3_key)
    s3_model_data = f"s3://{bucket_name}/{s3_key}"
    print(f"[OK] Uploaded to {s3_model_data}")

    # 3. Deploy
    print(f"[*] Initializing SageMaker PyTorchModel...")
    sagemaker_session = sagemaker.Session(boto_session=session)
    pytorch_model = PyTorchModel(
        model_data=s3_model_data,
        role=role_arn,
        framework_version="2.0.0",
        py_version="py310",
        entry_point="code/inference.py",
        sagemaker_session=sagemaker_session
    )

    print(f"[*] Deploying real-time endpoint '{endpoint_name}' on {instance_type}...")
    predictor = pytorch_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    print(f"[SUCCESS] Endpoint successfully deployed: {predictor.endpoint_name}")
    return predictor


def main():
    parser = argparse.ArgumentParser(description="Package and deploy DistilGPT2 to Amazon SageMaker.")
    parser.add_argument("--package-only", action="store_true", help="Only build model.tar.gz without uploading or deploying.")
    parser.add_argument("--role-arn", type=str, default="", help="AWS IAM execution role ARN for SageMaker.")
    parser.add_argument("--bucket", type=str, default="", help="AWS S3 bucket for model artifact upload.")
    parser.add_argument("--endpoint-name", type=str, default="distilgpt2-email-reply-generator", help="SageMaker endpoint name.")
    parser.add_argument("--instance-type", type=str, default="ml.m5.large", help="SageMaker instance type.")

    args = parser.parse_args()

    artifact_dir = PROJECT_ROOT / "sagemaker" / "dist"
    package_model_artifact(artifact_dir)

    if args.package_only:
        print("[*] Completed packaging (--package-only flag provided). Skipping cloud deployment.")
        return

    if not args.role_arn or not args.bucket:
        print("\n[NOTE] Cloud deployment requires --role-arn and --bucket parameters.")
        print("Example usage:")
        print("  python sagemaker/deploy_sagemaker.py --role-arn arn:aws:iam::123456789012:role/SageMakerRole --bucket my-sagemaker-bucket")
        print("\nFor local testing without AWS credentials, the application provides a seamless")
        print("SageMaker container simulation mode that runs the exact same inference.py pipeline.")
        return

    deploy_to_sagemaker(
        role_arn=args.role_arn,
        bucket_name=args.bucket,
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type
    )


if __name__ == "__main__":
    main()
