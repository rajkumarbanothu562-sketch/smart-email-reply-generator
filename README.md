# Smart Email Reply Generator (Project 2)

An NLP text-generation web application that accepts an email message or situation and generates a suggested reply draft in a selected tone (**Formal** or **Friendly**) using **DistilGPT2**, deployed via **PyTorch**, **TorchServe**, and **Amazon SageMaker**.

---

## 🚀 Key Features

- **Tone-Conditioned Generation**: Generates contextual email replies tailored in either *Formal* (polite, respectful corporate style) or *Friendly* (warm, cordial conversational style) registers.
- **Human-in-the-Loop Safety**: Every generated output is prominently labelled as a **DRAFT** requiring user review; direct in-browser editing is provided; **the application never sends emails automatically**.
- **Three Inference Backends**:
  1. **Local PyTorch Direct**: In-memory PyTorch generation engine with causal LM attention caching.
  2. **TorchServe Service**: Custom `EmailReplyHandler` adhering to official TorchServe REST contracts (`/predictions/email_reply_generator` and `/ping`).
  3. **Amazon SageMaker Endpoint**: Full cloud deployment script with dual-mode support (remote AWS endpoint via `boto3` or local container simulator).
- **Rigorous Input Validation**: Guards against missing input, blank input, and excessively long input (>1000 characters).
- **10 Benchmark Scenarios**: Includes 5 formal and 5 friendly test cases with live execution and comparison table export.
- **Comprehensive Report**: `REPORT.md` answering Questions 7 through 11 in detail with diagrams and tables.

---

## 📂 Project Directory Structure

```text
smart-email-reply-generator/
├── src/
│   ├── __init__.py
│   ├── prompt_builder.py          # Prompt engineering, validation, and postprocessing
│   └── generator.py               # DistilGPT2 PyTorch generation engine
├── serve/
│   ├── __init__.py
│   ├── handler.py                 # Custom TorchServe Handler (EmailReplyHandler)
│   ├── create_mar.py              # Packaging script for .mar model archive
│   └── mock_torchserve.py         # TorchServe REST API server (:8080)
├── sagemaker/
│   ├── __init__.py
│   ├── inference.py               # SageMaker PyTorch entrypoint (model_fn, input_fn, predict_fn, output_fn)
│   ├── deploy_sagemaker.py        # SageMaker packaging (model.tar.gz) and deployment script
│   └── sagemaker_client.py        # Dual-mode client for remote AWS or local simulation
├── web/
│   ├── templates/
│   │   └── index.html             # Responsive UI with live prompt inspector and review editor
│   ├── static/
│   │   ├── style.css              # Styling with badges, counters, and responsive grid
│   │   └── app.js                 # Dynamic interaction, prompt preview, and test runner
│   └── app.py                     # Flask web server connecting UI to backends
├── tests/
│   ├── __init__.py
│   ├── test_reply_generator.py     # Unit and validation test suite
│   ├── generate_comparison_table.py# Generates Markdown and CSV comparison tables
│   ├── comparison_table.md        # Benchmark comparison table (Markdown)
│   └── comparison_table.csv       # Benchmark comparison table (CSV)
├── REPORT.md                      # Comprehensive answers to questions 7-11
├── README.md                      # Project documentation
├── requirements.txt               # Dependencies
├── run.py                         # One-click CLI entry point
└── start.bat                      # Windows launcher script
```

---

## 🛠️ Installation & Setup

1. **Activate Environment**:
   Ensure Python 3.10+ with PyTorch and Transformers is available:
   ```bash
   pip install -r requirements.txt
   ```

2. **Launch the Web Application**:
   ```bash
   python run.py
   # Or on Windows:
   start.bat
   ```
   Open your browser at: **`http://127.0.0.1:5000`**

---

## 🧪 Running Tests & Benchmarks

- **Run Automated Unit & Validation Tests**:
  ```bash
  python run.py --test
  ```
  Tests input validation, missing input rejection, long input guard (>1000 chars), prompt construction, TorchServe handler lifecycle, and SageMaker inference pipeline.

- **Run Benchmark Comparison Across 10 Test Cases**:
  ```bash
  python run.py --benchmark
  ```
  Generates `tests/comparison_table.md` and `tests/comparison_table.csv`.

---

## 📦 Serving Options

### 1. Local PyTorch Direct
Selected by default in the web UI. Uses in-memory DistilGPT2 instance.

### 2. Standalone TorchServe API
Start the TorchServe REST server:
```bash
python serve/mock_torchserve.py 8080
```
- Health Check: `GET http://127.0.0.1:8080/ping`
- Inference: `POST http://127.0.0.1:8080/predictions/email_reply_generator`

Package into `.mar` file:
```bash
python serve/create_mar.py
```

### 3. Amazon SageMaker
Package model artifact `model.tar.gz`:
```bash
python sagemaker/deploy_sagemaker.py --package-only
```
Deploy to AWS SageMaker endpoint:
```bash
python sagemaker/deploy_sagemaker.py --role-arn <YOUR_IAM_ROLE_ARN> --bucket <YOUR_S3_BUCKET>
```
If AWS credentials are not configured, the UI seamlessly falls back to the SageMaker Local Container simulator.

---

## 📄 Deliverables & Questions Answered

Detailed answers to all assigned project questions are provided in **`REPORT.md`**:
- **Question 7**: Tone influence on prompt structure and token distributions.
- **Question 8**: Difference between user email context and generated reply.
- **Question 9**: Handler validation logic for missing, short, and excessively long input.
- **Question 10**: Analysis of tone adherence with 4 concrete examples of adherence and drift.
- **Question 11**: Verification procedure testing model/tokenizer parity between TorchServe and SageMaker.
