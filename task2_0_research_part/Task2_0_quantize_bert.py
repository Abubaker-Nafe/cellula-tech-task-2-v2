"""Compare a real BERT model before and after dynamic INT8 quantization.

Install: python -m pip install "transformers[torch]" matplotlib
Run:     python task2_0_research_part/Task2_0_quantize_bert.py
Lighter: python task2_0_research_part/Task2_0_quantize_bert.py --model distilbert/distilbert-base-uncased

The first run downloads the chosen pretrained model. The saved checkpoint
comparison includes every model tensor, so the reduction will be smaller than
the theoretical 75% for weights: embeddings and other layers stay in FP32.
The temporary checkpoints are deleted when the script finishes.
"""

import argparse
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoModel, AutoTokenizer


def checkpoint_size(model: torch.nn.Module, path: Path) -> int:
    """Measure the serialized model state, including quantization metadata."""
    torch.save(model.state_dict(), path)
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize BERT Linear layers to INT8")
    parser.add_argument("--model", default="google-bert/bert-base-uncased")
    args = parser.parse_args()

    print(f"Loading {args.model} on the CPU (first run downloads it)...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    fp32_model = AutoModel.from_pretrained(args.model).cpu().eval()
    inputs = tokenizer("Quantization reduces model storage.", return_tensors="pt")

    with torch.inference_mode():
        fp32_output = fp32_model(**inputs).last_hidden_state

    # Post-training quantization: pretrained weights are converted without retraining.
    # Only torch.nn.Linear modules are converted; embeddings/LayerNorm stay FP32.
    int8_model = torch.ao.quantization.quantize_dynamic(
        fp32_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
    ).eval()
    converted_layers = sum(
        isinstance(module, torch.ao.nn.quantized.dynamic.Linear)
        for module in int8_model.modules()
    )
    if converted_layers == 0:
        raise RuntimeError("No Linear layers were converted to INT8.")

    with torch.inference_mode():
        int8_output = int8_model(**inputs).last_hidden_state

    difference = (fp32_output - int8_output).abs()
    with tempfile.TemporaryDirectory(prefix="bert_int8_") as directory:
        folder = Path(directory)
        fp32_bytes = checkpoint_size(fp32_model, folder / "fp32_state.pt")
        int8_bytes = checkpoint_size(int8_model, folder / "int8_state.pt")

    mib = 1024**2
    print(f"Converted Linear layers: {converted_layers}")
    print(f"FP32 checkpoint: {fp32_bytes / mib:.2f} MiB")
    print(f"INT8 checkpoint: {int8_bytes / mib:.2f} MiB")
    print(f"Checkpoint reduction: {100 * (1 - int8_bytes / fp32_bytes):.2f}%")
    print(f"Mean absolute output difference: {difference.mean().item():.6f}")
    print(f"Maximum absolute output difference: {difference.max().item():.6f}")

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(
        ["FP32", "Dynamic INT8"],
        [fp32_bytes / mib, int8_bytes / mib],
        color=["#4063a6", "#23a49a"],
    )
    axis.bar_label(bars, fmt="%.1f MiB", padding=4)
    axis.set(
        ylabel="Serialized state size (MiB)",
        title=f"{args.model.split('/')[-1]} checkpoint size",
    )
    axis.set_ylim(0, max(fp32_bytes, int8_bytes) / mib * 1.2)
    figure.tight_layout()
    graph_path = Path(__file__).with_name("Task2_0_bert_quantization_graph.png")
    figure.savefig(graph_path, dpi=180)
    plt.close(figure)
    print(f"Graph saved: {graph_path}")
    print("Output difference measures numerical change, not task accuracy.")


if __name__ == "__main__":
    main()
