"""CLI entry point for the inference benchmark.

Usage:
    python scripts/benchmark_inference.py

See src/cnnClassifier/components/benchmarking.py for what this measures
(latency/throughput/param count across candidate backbones) and, just as
importantly, what it deliberately does not measure (accuracy - that needs a
real fine-tuning run per architecture, which this repo hasn't done).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from cnnClassifier.components.benchmarking import run_benchmark  # noqa: E402


def main():
    results = run_benchmark()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
