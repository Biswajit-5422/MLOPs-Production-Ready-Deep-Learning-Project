"""Inference latency/throughput benchmarking across candidate backbones.

This deliberately does NOT compare accuracy - that would require fine-tuning
each architecture on the real dataset, which needs a GPU and real training
time neither of which this was run with. What it *does* measure honestly:
given the same (untrained) 2-class classification head attached to each
ImageNet backbone, how do they compare on inference latency, throughput, and
parameter count? Weight values don't affect latency, only architecture and
FLOPs do, so this is a legitimate accuracy-independent comparison, and one
input to an eventual choice of architecture alongside real accuracy numbers.
"""
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

ARCHITECTURES = {
    "VGG16": tf.keras.applications.VGG16,
    "ResNet50": tf.keras.applications.ResNet50,
    "MobileNetV2": tf.keras.applications.MobileNetV2,
}


def _build_classifier(backbone_fn, image_size, classes=2, weights="imagenet"):
    backbone = backbone_fn(input_shape=image_size, weights=weights, include_top=False)
    backbone.trainable = False
    flatten = tf.keras.layers.Flatten()(backbone.output)
    prediction = tf.keras.layers.Dense(classes, activation="softmax")(flatten)
    return tf.keras.models.Model(inputs=backbone.input, outputs=prediction)


def benchmark_model(
    name, backbone_fn, image_size=(224, 224, 3), batch_sizes=(1, 8, 16), warmup=2, iterations=10
):
    model = _build_classifier(backbone_fn, image_size)
    num_params = model.count_params()

    results = {"model": name, "num_params": num_params, "batches": {}}

    for batch_size in batch_sizes:
        dummy_input = np.random.rand(batch_size, *image_size).astype("float32")

        for _ in range(warmup):
            model.predict(dummy_input, verbose=0)

        latencies_ms = []
        for _ in range(iterations):
            start = time.perf_counter()
            model.predict(dummy_input, verbose=0)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        latencies_ms.sort()
        mean_ms = sum(latencies_ms) / len(latencies_ms)
        p50_ms = latencies_ms[len(latencies_ms) // 2]
        p95_ms = latencies_ms[max(0, int(len(latencies_ms) * 0.95) - 1)]
        throughput = batch_size / (mean_ms / 1000)

        results["batches"][str(batch_size)] = {
            "mean_ms": round(mean_ms, 2),
            "p50_ms": round(p50_ms, 2),
            "p95_ms": round(p95_ms, 2),
            "throughput_images_per_sec": round(throughput, 2),
        }

    del model
    tf.keras.backend.clear_session()
    return results


def run_benchmark(output_path="benchmarks/inference_benchmark.json", architectures=None):
    architectures = architectures or ARCHITECTURES
    all_results = []
    for name, backbone_fn in architectures.items():
        print(f"Benchmarking {name}...")
        all_results.append(benchmark_model(name, backbone_fn))

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    run_benchmark()
