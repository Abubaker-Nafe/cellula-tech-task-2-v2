from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Save a figure without opening a desktop window.
import matplotlib.pyplot as plt
import numpy as np


def quantize_int8(weights: np.ndarray) -> tuple[np.ndarray, np.float32]:
    """Map float32 weights to signed INT8 using one symmetric scale."""
    max_magnitude = float(np.max(np.abs(weights)))
    scale = np.float32(max_magnitude / 127 if max_magnitude else 1.0)
    integers = np.clip(np.rint(weights / scale), -127, 127).astype(np.int8)
    return integers, scale


def main() -> None:
    rng = np.random.default_rng(42)
    original = rng.normal(loc=0.0, scale=0.15, size=(256, 256)).astype(np.float32)
    quantized, scale = quantize_int8(original)
    recovered = quantized.astype(np.float32) * scale

    original_bytes = original.nbytes
    quantized_bytes = quantized.nbytes + scale.nbytes  # Include the stored scale.
    mean_absolute_error = float(np.mean(np.abs(original - recovered)))
    max_absolute_error = float(np.max(np.abs(original - recovered)))

    print(f"Weights: {original.size:,} (simulated, not an entire model)")
    print(f"FP32 weight storage: {original_bytes:,} bytes")
    print(f"INT8 weight storage + scale: {quantized_bytes:,} bytes")
    print(f"Storage reduction: {100 * (1 - quantized_bytes / original_bytes):.2f}%")
    print(f"Scale: {scale:.6f}")
    print(f"Mean absolute reconstruction error: {mean_absolute_error:.6f}")
    print(f"Maximum absolute reconstruction error: {max_absolute_error:.6f}")

    figure, (weights_axis, storage_axis) = plt.subplots(1, 2, figsize=(10, 4))
    sample = np.arange(60)
    weights_axis.plot(sample, original.flat[:60], label="FP32 weights", linewidth=1.3)
    weights_axis.plot(sample, recovered.flat[:60], "--", label="Recovered from INT8", linewidth=1.1)
    weights_axis.set(title="First 60 weights", xlabel="Weight index", ylabel="Value")
    weights_axis.legend(fontsize=8)

    sizes_kib = [original_bytes / 1024, quantized_bytes / 1024]
    bars = storage_axis.bar(["FP32", "INT8 + scale"], sizes_kib, color=["#4063a6", "#23a49a"])
    storage_axis.set(title="Weight storage", ylabel="KiB")
    storage_axis.bar_label(bars, fmt="%.1f KiB", padding=4)
    storage_axis.set_ylim(0, max(sizes_kib) * 1.2)

    figure.suptitle("Post-training symmetric INT8 quantization")
    figure.tight_layout()
    graph_path = Path(__file__).with_name("Task2_0_quantization_graph.png")
    figure.savefig(graph_path, dpi=180)
    plt.close(figure)
    print(f"Graph saved: {graph_path}")


if __name__ == "__main__":
    main()
