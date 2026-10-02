import csv
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
import streamlit as st

from Task2_1_predict_text import classify_text, classify_text_with_caption


HISTORY_PATH = Path(__file__).resolve().parent / "classification_history.csv"
HISTORY_COLUMNS = [
    "timestamp_utc",
    "input_type",
    "user_text",
    "image_caption",
    "classification",
    "model_score",
]


def save_classification(text: str, caption: str | None, label: str, score: float) -> None:
    """Add one completed submission to the CSV database."""
    with HISTORY_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        if file.tell() == 0:
            writer.writeheader()
        writer.writerow(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "input_type": "text + image" if text and caption else "text" if text else "image",
                "user_text": text,
                "image_caption": caption or "",
                "classification": label,
                "model_score": f"{score:.3f}",
            }
        )


def read_history() -> list[dict[str, str]]:
    if not HISTORY_PATH.exists():
        return []
    with HISTORY_PATH.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


st.set_page_config(page_title="Content Classification", page_icon="🔎")
st.title("Content Classification")
st.write("Enter text, upload an image, or use both.")

text = st.text_area("Text", placeholder="Type a message to classify...")
uploaded_image = st.file_uploader("Image (optional)", type=["jpg", "jpeg", "png"])

if st.button("Classify", type="primary"):
    text = text.strip()
    if not text and uploaded_image is None:
        st.warning("Enter text or upload an image first.")
        st.stop()

    caption = None
    if uploaded_image is not None:
        try:
            with Image.open(uploaded_image) as opened_image:
                image = opened_image.convert("RGB")
        except (OSError, ValueError):
            st.error("The uploaded file could not be opened as an image.")
            st.stop()

        st.image(image, caption=uploaded_image.name)
        from task2_1_imagecaption import generate_caption

        with st.spinner("Generating image caption..."):
            caption = generate_caption(image)
        st.write("Generated caption:", caption)

    with st.spinner("Classifying content..."):
        if text and caption:
            label, score = classify_text_with_caption(text, caption)
        elif text:
            label, score = classify_text(text)
        else:
            # Provisional: the text LSTM was trained on queries, not stand-alone captions.
            label, score = classify_text(caption)

    save_classification(text, caption, label, score)
    st.subheader("Result")
    st.write(f"**Category:** {label}")
    st.write(f"Model score: {score:.3f}")
    if caption and not text:
        st.info("Image-only classifications are preliminary and need separate evaluation.")
    st.caption("These are content labels, not assessments of real-world safety.")

with st.expander("View saved history"):
    history = read_history()
    if history:
        st.dataframe(history[::-1], hide_index=True)
    else:
        st.write("No submissions saved yet.")
