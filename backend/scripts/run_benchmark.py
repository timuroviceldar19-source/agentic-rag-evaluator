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
    args = parser.parse_args()

    report = BenchmarkRunner().run(args.dataset.resolve())
    if args.json:
        print(json.dumps(report.model_dump(), indent=2))
    else:
        print(format_benchmark_report(report))


if __name__ == "__main__":
    main()

