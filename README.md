# 🥗 AI-Powered Nutritional Tracker

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([Streamlit App](https://nutritionaltracker.streamlit.app/))
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-yellow)](YOUR_HUGGINGFACE_REPO_URL_HERE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

A Generative AI application that extracts structured nutritional information from images of food labels. This project leverages a fine-tuned **LongT5 model** to parse OCR output into clean, structured JSON data.

## 🚀 Live Demo

**[Click here to try the App on Streamlit Cloud]([Here! ](https://nutritionaltracker.streamlit.app/))**

---

## 📸 How It Works

This application allows users to upload an image of a nutrition facts label, crop it for better accuracy, and instantly generate a structured breakdown of the nutritional content.

### Usage Instructions:
1.  **Upload:** Upload a clear image of a food product's nutrition label (PNG, JPG, JPEG).
2.  **Adjust:** Use the built-in **Rotate** and **Crop** tools to highlight *only* the nutritional table.
3.  **Scan:** Enter the product barcode (optional) for record-keeping.
4.  **Process:** Click **Run Model**.
    * The app first runs **OCR (Doctr)** to extract raw text.
    * It then passes the text to a **LongT5 Model** (hosted on Hugging Face) to generate structured JSON.

---

## 🏗️ Architecture: The "Hybrid" Approach

This project demonstrates a modern MLOps architecture designed to bypass standard Git file size limits:

* **Logic (GitHub):** The source code, OCR logic, and Streamlit UI.
* **Weights (Hugging Face Hub):** The 3.6GB model checkpoints (`model.safetensors`) are hosted on the Hugging Face Hub.
* **Compute (Streamlit Cloud):** The app pulls the model dynamically from Hugging Face at runtime using the `transformers` library.

---

## 🛠️ Local Installation

If you want to run this project locally on your machine:

**1. Clone the repository**
```bash
git clone [https://github.com/MrTig/NutritionalTracker-App.git](https://github.com/MrTig/NutritionalTracker-App.git)
cd NutritionalTracker-App
```
**2. Install Dependencies**
```bash
pip install -r requirements.txt
```
**3. Run the App**
```bash
streamlit run poc.py
```
The first run will take a moment to download the model weights from Hugging Face.

📦 Tech Stack
Frontend: Streamlit

OCR: Doctr (Document Text Recognition)

LLM: LongT5 (Fine-tuned)

Model Hosting: Hugging Face Hub

Orchestration: Python, PyTorch, Transformers

📝 License
[MIT](https://choosealicense.com/licenses/mit/)