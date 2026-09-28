"""Create an image caption with BLIP-1.

Run on its own: python imagecaption.py path/to/image.jpg
The Streamlit app will import generate_caption from this module later.
"""

import argparse
from functools import lru_cache
from pathlib import Path

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


MODEL_NAME = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def load_caption_model():
    """Load the processor and model once per Python process."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    return processor, model, device


def generate_caption(image: Image.Image) -> str:
    """Return a plain English caption for a PIL image."""
    processor, model, device = load_caption_model()
    inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=40)

    return processor.decode(generated_ids[0], skip_special_tokens=True).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test BLIP image captioning")
    parser.add_argument("image_path", type=Path, help="Path to a local image")
    args = parser.parse_args()

    with Image.open(args.image_path) as image:
        print(generate_caption(image))
