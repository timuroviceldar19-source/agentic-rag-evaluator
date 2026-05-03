import argparse
import json
import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.services.benchmark import BenchmarkRunner, format_benchmark_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation benchmark.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=BACKEND_DIR / "benchmarks" / "sample_questions.json",
        help="Path to benchmark dataset JSON.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full report as JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the benchmark report.",
    )
    args = parser.parse_args()

    report = BenchmarkRunner().run(args.dataset.resolve())
    if args.json:
        output = json.dumps(report.model_dump(), indent=2)
    else:
        output = format_benchmark_report(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
