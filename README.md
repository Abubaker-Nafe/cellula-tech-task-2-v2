# Week 2, Task 2: Content Classification

This Streamlit app classifies submitted text, an uploaded image, or both. It uses the LSTM models saved from Task 1. For images, `imagecaption.py` uses BLIP to generate a caption before classification.

## Deployed app

Open the app at [cellula-tech-task-2.streamlit.app](https://cellula-tech-task-2.streamlit.app/). It is deployed on Streamlit Community Cloud from `Abubaker-Nafe/cellula-tech-task-2`, branch `main`, with `app.py` as the entry point.

## Setup (VS Code on Windows)

Use Python 3.11. Put these files in the same folder:

```text
Task-2/
├── app.py
├── Task2_predict_text.py
├── imagecaption.py
├── requirements.txt
├── README.md
└── task2_artifacts/
    ├── settings.json
    ├── text_lstm.keras
    ├── text_tokenizer.json
    ├── combined_lstm.keras
    └── combined_tokenizer.json
```

If the artifacts are in a ZIP file, extract its five files into the `task2_artifacts` folder beside the scripts. Open a terminal in `Task-2` and install the dependencies:

```bat
python -m pip install -r requirements.txt
```

The first image run downloads the BLIP model, so it needs an internet connection and enough free disk space. Later runs use the cached download.

## Run locally

```bat
python -m streamlit run app.py --server.fileWatcherType none
```

Open the Local URL printed in the terminal (usually `http://localhost:8501`). Enter text, upload a JPG or PNG image, or provide both, then click **Classify**. Open **View saved history** to see previous results. To stop the server, press `Ctrl+C`; restart it after editing the code because file watching is disabled in this command.

Each successful submission is appended to `classification_history.csv` in the same folder. The file is created automatically and stores the input type, submitted text, generated caption, classification, model score, and UTC timestamp.

On Streamlit Community Cloud, [local file storage is not guaranteed to persist](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data), so the deployed app's CSV history may disappear after a restart or redeployment. For permanent shared history, use external persistent storage.

## Optional command-line checks

```bat
python Task2_predict_text.py "How can I learn about elections?"
python imagecaption.py "sea.jpg"
python Task2_predict_text.py "Is it safe to swim here?" --image "sea.jpg"
```

For a known caption, use `--caption "ocean waves"` instead of `--image "sea.jpg"`. The CLI classifier requires text even when supplying an image; the Streamlit app also accepts an image by itself. Image-only classifications are preliminary because they use a text model trained on user queries. Labels describe content categories, not whether a real-world activity is safe.

## Submission note

The Task 2 brief says submitted code must be `.py` files and does not allow notebooks. Check whether the submission also accepts the supporting model artifacts, `requirements.txt`, and this README; keep these setup files locally if the submission accepts only `.py` files.
