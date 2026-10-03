# Week 2, Task 2: Content Classification

This repository keeps the Task 2.0 quantization work separate from the Task 2.1 content-classification app. The Streamlit app classifies submitted text, an uploaded image, or both. It uses the LSTM models saved from Task 1. For images, `task2_1_imagecaption.py` uses BLIP to generate a caption before classification.

## Deployed app

Open the app at [cellula-tech-task-2.streamlit.app](https://cellula-tech-task-2.streamlit.app/). It is deployed on Streamlit Community Cloud from `Abubaker-Nafe/cellula-tech-task-2`, branch `main`. After this folder reorganization, use `task_2_1_Toxic_content_classification_project/app.py` as the entrypoint file. If an existing deployment still points to the former root-level `app.py`, delete that Streamlit app before applying the repository move, then redeploy it with the new entrypoint path. Streamlit documents this as a change to the app's GitHub coordinates.

## Setup (VS Code on Windows)

Use Python 3.11. The repository is organized into two project parts:

```text
Task-2/
├── task2_0_research_part/
│   ├── Task0_Quantization_Research.pdf
│   ├── Task2_0_quantization_demo.py
│   ├── Task2_0_quantize_bert.py
│   ├── Task2_0_quantization_graph.png
│   └── Task2_0_bert_quantization_graph.png
└── task_2_1_Toxic_content_classification_project/
    ├── app.py
    ├── Task2_1_predict_text.py
    ├── task2_1_imagecaption.py
    ├── requirements.txt
    ├── README.md
    ├── classification_history.csv
    ├── images/
    │   └── sea.jpg
    └── task2_1_artifacts/
        ├── settings.json
        ├── text_lstm.keras
        ├── text_tokenizer.json
        ├── combined_lstm.keras
        └── combined_tokenizer.json
```

If the artifacts are in a ZIP file, extract its five files into the `task2_1_artifacts` folder beside the scripts. Open a terminal in the repository root (`Task-2`) and install the dependencies:

```bat
python -m pip install -r task_2_1_Toxic_content_classification_project/requirements.txt
```

The first image run downloads the BLIP model, so it needs an internet connection and enough free disk space. Later runs use the cached download.

## Run locally

```bat
python -m streamlit run task_2_1_Toxic_content_classification_project/app.py --server.fileWatcherType none
```

Open the Local URL printed in the terminal (usually `http://localhost:8501`). Enter text, upload a JPG or PNG image, or provide both, then click **Classify**. Open **View saved history** to see previous results. To stop the server, press `Ctrl+C`; restart it after editing the code because file watching is disabled in this command.

Each successful submission is appended to `classification_history.csv` in the same folder. The file is created automatically and stores the input type, submitted text, generated caption, classification, model score, and UTC timestamp.

On Streamlit Community Cloud, [local file storage is not guaranteed to persist](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data), so the deployed app's CSV history may disappear after a restart or redeployment. For permanent shared history, use external persistent storage.

## Optional command-line checks

```bat
python task_2_1_Toxic_content_classification_project/Task2_1_predict_text.py "How can I learn about elections?"
python task_2_1_Toxic_content_classification_project/task2_1_imagecaption.py "task_2_1_Toxic_content_classification_project/images/sea.jpg"
python task_2_1_Toxic_content_classification_project/Task2_1_predict_text.py "Is it safe to swim here?" --image "task_2_1_Toxic_content_classification_project/images/sea.jpg"
```

For a known caption, use `--caption "ocean waves"` instead of `--image "images\sea.jpg"`. The CLI classifier requires text even when supplying an image; the Streamlit app also accepts an image by itself. Image-only classifications are preliminary because they use a text model trained on user queries. Labels describe content categories, not whether a real-world activity is safe.

## Task 2.0 quantization checks

Run these commands from the `Task-2` folder:

```bat
python task2_0_research_part/Task2_0_quantization_demo.py
python task2_0_research_part/Task2_0_quantize_bert.py
```

The BERT check downloads `google-bert/bert-base-uncased` the first time. For a lighter download, add `--model distilbert/distilbert-base-uncased`. Both scripts save their graph beside the corresponding Task 2.0 code.

## Submission note

The Task 2 brief says submitted code must be `.py` files and does not allow notebooks. Check whether the submission also accepts the supporting model artifacts, `requirements.txt`, and this README; keep these setup files locally if the submission accepts only `.py` files.
