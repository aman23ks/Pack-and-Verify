#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

from cli import ask  # uses your fixed ask(question, budget, namespace)


# You can tweak these budgets however you want
BUDGETS = [2000, 2500, 3000, 4000, 5000, 6000, 8000, 10000]


def run_batch(input_csv: str, output_csv: str):
    in_path = Path(input_csv)
    if not in_path.exists():
        print(f"Input CSV not found: {input_csv}")
        sys.exit(1)

    rows = []
    with in_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Basic validation
            ns = (row.get("namespace") or "").strip()
            q = (row.get("question") or "").strip()
            if not ns or not q:
                print(f"Skipping row with missing namespace/question: {row}")
                continue
            rows.append({"namespace": ns, "question": q})

    if not rows:
        print("No valid rows found in input CSV.")
        sys.exit(0)

    # Prepare output fieldnames
    fieldnames = ["namespace", "question"]
    for budget in BUDGETS:
        fieldnames.append(f"answer_{budget}")
        fieldnames.append(f"used_tokens_{budget}")  # optional, can be blank for now

    out_path = Path(output_csv)
    with out_path.open("w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            ns = row["namespace"]
            question = row["question"]
            out_row = {
                "namespace": ns,
                "question": question,
            }

            print(f"\n===== Row {idx}: ns={ns} =====")
            print(f"Question: {question}")

            for budget in BUDGETS:
                print(f"\n--- Budget {budget} ---")
                try:
                    # ask() already prints, but we also capture the answer
                    ans = ask(question, budget, namespace=ns)
                    out_row[f"answer_{budget}"] = ans
                    # If you later want real used_tokens, you can modify ask() to return it too.
                    out_row[f"used_tokens_{budget}"] = ""  # placeholder
                except Exception as e:
                    print(f"[ERROR] Failed for budget {budget}: {e}")
                    out_row[f"answer_{budget}"] = f"[ERROR: {e}]"
                    out_row[f"used_tokens_{budget}"] = ""

            writer.writerow(out_row)

    print(f"\nDone. Results written to: {output_csv}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_eval.py <input_csv> <output_csv>")
        print("Example: python batch_eval.py Automation.csv Automation_results.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_csv = sys.argv[2]
    run_batch(input_csv, output_csv)


if __name__ == "__main__":
    main()
