"""Load the Task 1 LSTMs and classify text with an optional image.

Extract task2_artifacts.zip into a task2_artifacts/ folder beside this file.
Run: python Task2_predict_text.py "Your text here"
Run: python Task2_predict_text.py "Your text here" --caption "An image caption"
Run: python Task2_predict_text.py "Your text here" --image image.jpg
"""

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json


ARTIFACTS = Path(__file__).resolve().parent / "task2_artifacts"


@lru_cache(maxsize=2)
def load_classifier(mode: str = "text"):
    if mode not in ("text", "combined"):
        raise ValueError(f"Unknown classifier mode: {mode}")

    settings = json.loads((ARTIFACTS / "settings.json").read_text(encoding="utf-8"))
    tokenizer = tokenizer_from_json(
        (ARTIFACTS / f"{mode}_tokenizer.json").read_text(encoding="utf-8")
    )
    model = load_model(ARTIFACTS / f"{mode}_lstm.keras", compile=False)
    return model, tokenizer, settings


def _classify(input_text: str, mode: str) -> tuple[str, float]:
    model, tokenizer, settings = load_classifier(mode)
    sequence = tokenizer.texts_to_sequences([input_text])
    padded = pad_sequences(
        sequence,
        maxlen=settings[f"{mode}_max_length"],
        padding=settings["padding"],
        truncating=settings["truncating"],
    )

    scores = model.predict(padded, verbose=0)[0]
    if len(scores) != len(settings["class_names"]):
        raise ValueError("The saved model and class list have different sizes.")

    predicted_index = int(np.argmax(scores))
    return settings["class_names"][predicted_index], float(scores[predicted_index])


def classify_text(text: str) -> tuple[str, float]:
    if not text.strip():
        raise ValueError("Please provide some text to classify.")
    return _classify(text, "text")


def classify_text_with_caption(text: str, caption: str) -> tuple[str, float]:
    if not text.strip() or not caption.strip():
        raise ValueError("Provide both text and an image caption.")
    # The Task 1 combined model was trained on query + " " + image description.
    return _classify(f"{text} {caption}", "combined")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify text with an optional image")
    parser.add_argument("text", help="Text to classify")
    image_input = parser.add_mutually_exclusive_group()
    image_input.add_argument("--caption", help="Caption for the accompanying image")
    image_input.add_argument("--image", type=Path, help="Path to a local image")
    args = parser.parse_args()

    caption = args.caption
    if args.image is not None:
        from PIL import Image
        from imagecaption import generate_caption

        with Image.open(args.image) as image:
            caption = generate_caption(image)
        print(f"Image caption: {caption}")

    label, score = (
        classify_text(args.text)
        if caption is None
        else classify_text_with_caption(args.text, caption)
    )
    print(f"Prediction: {label} (model score: {score:.3f})")
