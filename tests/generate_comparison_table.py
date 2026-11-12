"""
generate_comparison_table.py
Executes 5 Formal and 5 Friendly test cases using DistilGPT2,
records the constructed prompt, selected tone, and generated reply,
and generates formatted Markdown and CSV comparison tables.
"""

import sys
import csv
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.generator import EmailReplyGenerator

TEST_CASES = [
    # 5 Formal Test Cases
    {
        "id": "F1",
        "situation": "Requesting leave for one day due to a family medical appointment.",
        "tone": "formal",
        "expected_behaviour": "Generate a polite, professional leave-related draft."
    },
    {
        "id": "F2",
        "situation": "Requesting assignment extension for the Machine Learning final project due to illness.",
        "tone": "formal",
        "expected_behaviour": "Generate a respectful academic email draft."
    },
    {
        "id": "F3",
        "situation": "Inquiring about the status of a job application for the NLP Engineer role.",
        "tone": "formal",
        "expected_behaviour": "Generate a courteous professional status inquiry draft."
    },
    {
        "id": "F4",
        "situation": "Declining a vendor sales pitch meeting respectfully due to budget constraints.",
        "tone": "formal",
        "expected_behaviour": "Generate a polite, formal declination email draft."
    },
    {
        "id": "F5",
        "situation": "Submitting project status report and requesting executive review.",
        "tone": "formal",
        "expected_behaviour": "Generate a formal deliverable submission email draft."
    },
    # 5 Friendly Test Cases
    {
        "id": "FR1",
        "situation": "Unable to attend tomorrow's meeting because I have an appointment.",
        "tone": "friendly",
        "expected_behaviour": "Generate a courteous, conversational message."
    },
    {
        "id": "FR2",
        "situation": "Thanking a colleague for helping debug the deployment script today.",
        "tone": "friendly",
        "expected_behaviour": "Generate a brief appreciative reply."
    },
    {
        "id": "FR3",
        "situation": "Inviting teammates for a casual coffee break and project catch-up.",
        "tone": "friendly",
        "expected_behaviour": "Generate an informal, cheerful invitation draft."
    },
    {
        "id": "FR4",
        "situation": "Sharing notes from the conference and asking if they want to collaborate.",
        "tone": "friendly",
        "expected_behaviour": "Generate an enthusiastic, collaborative peer draft."
    },
    {
        "id": "FR5",
        "situation": "Congratulating a teammate on their work anniversary and promotion.",
        "tone": "friendly",
        "expected_behaviour": "Generate a warm, congratulatory email draft."
    }
]


def run_comparison():
    print("=" * 70)
    print("Generating Comparison Table: 5 Formal & 5 Friendly Test Cases")
    print("=" * 70)

    generator = EmailReplyGenerator.get_instance("distilgpt2")
    results = []

    for idx, tc in enumerate(TEST_CASES, start=1):
        print(f"\n[{idx}/10] Running Test Case {tc['id']} | Tone: {tc['tone'].upper()}")
        print(f"Situation: {tc['situation']}")
        
        res = generator.generate_reply(
            email_context=tc["situation"],
            tone=tc["tone"],
            max_new_tokens=70,
            temperature=0.65,
            top_p=0.85
        )
        
        reply = res["draft_reply"]
        prompt = res["prompt"]
        latency = res["latency_ms"]
        
        print(f"Latency: {latency} ms")
        print("Generated Reply Draft:")
        print("-" * 40)
        print(reply)
        print("-" * 40)
        
        results.append({
            "ID": tc["id"],
            "Situation": tc["situation"],
            "Tone": tc["tone"].capitalize(),
            "Expected Behaviour": tc["expected_behaviour"],
            "Prompt Summary": prompt.replace("\n", " ")[:120] + "...",
            "Full Prompt": prompt,
            "Generated Reply": reply,
            "Latency (ms)": latency,
            "Status": "Draft (Requires Review)"
        })

    # Save to Markdown
    md_file = PROJECT_ROOT / "tests" / "comparison_table.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write("# Smart Email Reply Generator - Test Cases & Comparison Table\n\n")
        f.write("> **Model**: DistilGPT2 | **Framework**: PyTorch / TorchServe / SageMaker\n")
        f.write("> **Total Cases**: 10 (5 Formal, 5 Friendly)\n\n")
        f.write("| ID | Email Situation | Tone | Generated Reply (Draft) | Latency (ms) |\n")
        f.write("|---|---|---|---|---|\n")
        for r in results:
            clean_reply = r["Generated Reply"].replace("\n", "<br>")
            f.write(f"| **{r['ID']}** | {r['Situation']} | `{r['Tone']}` | {clean_reply} | {r['Latency (ms)']} |\n")

        f.write("\n\n## Detailed Test Case Audit\n\n")
        for r in results:
            f.write(f"### Case {r['ID']}: {r['Situation']} ({r['Tone']})\n\n")
            f.write(f"- **Expected Behavior**: {r['Expected Behaviour']}\n")
            f.write(f"- **Constructed Prompt**:\n```text\n{r['Full Prompt']}\n```\n")
            f.write(f"- **Generated Draft Reply**:\n```text\n{r['Generated Reply']}\n```\n")
            f.write(f"- **Review Status**: {r['Status']} (Not sent automatically)\n\n")
            f.write("---\n\n")

    print(f"\n[OK] Markdown comparison table saved to: {md_file}")

    # Save to CSV
    csv_file = PROJECT_ROOT / "tests" / "comparison_table.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ID", "Situation", "Tone", "Expected Behaviour", "Generated Reply", "Latency (ms)", "Status"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "ID": r["ID"],
                "Situation": r["Situation"],
                "Tone": r["Tone"],
                "Expected Behaviour": r["Expected Behaviour"],
                "Generated Reply": r["Generated Reply"],
                "Latency (ms)": r["Latency (ms)"],
                "Status": r["Status"]
            })
    print(f"[OK] CSV comparison table saved to: {csv_file}")
    return results


if __name__ == "__main__":
    run_comparison()
