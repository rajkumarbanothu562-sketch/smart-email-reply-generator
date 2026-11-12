"""
create_mar.py
Automates packaging DistilGPT2 and EmailReplyHandler into a TorchServe Model Archive (.mar).
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def create_model_archive():
    project_root = Path(__file__).resolve().parent.parent
    model_store_dir = project_root / "model_store"
    model_store_dir.mkdir(parents=True, exist_ok=True)
    
    handler_path = project_root / "serve" / "handler.py"
    
    print("=" * 60)
    print("TorchServe Model Archiver for DistilGPT2 Email Reply Generator")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Handler path: {handler_path}")
    print(f"Model store:  {model_store_dir}")
    
    # Check if torch-model-archiver is installed
    archiver_cmd = shutil.which("torch-model-archiver")
    if not archiver_cmd:
        print("\n[NOTE] 'torch-model-archiver' command is not in the system PATH.")
        print("To install torch-model-archiver:")
        print("    pip install torch-model-archiver\n")
        print("Standard command to execute manually:")
        cmd_str = (
            f"torch-model-archiver "
            f"--model-name email_reply_generator "
            f"--version 1.0 "
            f"--handler {handler_path} "
            f"--export-path {model_store_dir} -f"
        )
        print(f"  {cmd_str}\n")
        
        # Create a metadata placeholder to indicate MAR structure readiness
        mar_meta_file = model_store_dir / "email_reply_generator.mar.info"
        with open(mar_meta_file, "w") as f:
            f.write("TorchServe Model Archive Spec: email_reply_generator\n")
            f.write("Model: distilgpt2\n")
            f.write(f"Handler: {handler_path}\n")
            f.write(f"Archive Command: {cmd_str}\n")
        print(f"[OK] Created archive metadata spec at {mar_meta_file}")
        return

    cmd = [
        "torch-model-archiver",
        "--model-name", "email_reply_generator",
        "--version", "1.0",
        "--handler", str(handler_path),
        "--export-path", str(model_store_dir),
        "-f"
    ]
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode == 0:
        print("[SUCCESS] Successfully packaged email_reply_generator.mar in model_store/")
    else:
        print(f"[ERROR] Archiving failed with return code {result.returncode}")


if __name__ == "__main__":
    create_model_archive()
