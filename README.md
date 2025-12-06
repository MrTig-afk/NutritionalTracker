# 🥗 Nutrition Tracker – Turning Nutrition Labels into Data

Tracking calories and macros can be a **frustrating, manual, and error-prone process**.
I wanted to change that — by building a system that **reads nutrition labels directly** and converts them into structured, usable data.

This project is a **Proof of Concept** showing how OCR and data engineering can simplify everyday nutrition tracking.

---

## 🚀 Features

* **OCR-Powered Extraction**
  Reads nutrition labels using open-source OCR and a custom seq2seq model.

* **Computer Vision Preprocessing**
  Image augmentation, cropping, and cleaning to improve OCR accuracy.

* **Data Normalization**
  ETL-style pipeline to clean and standardize OCR outputs.

* **Integration with OpenFoodFacts**
  Fetches additional structured nutritional data for consistency and enrichment.

* **Model Training**
  Custom seq2seq architecture trained on 1,000+ samples with bounding-box–based OCR preprocessing.

---

## 📊 Current Progress

* Training loss of **0.318** achieved over 3 epochs
* Processing speed: ~**1.6 samples/sec**
* End-to-end pipeline for ingestion → OCR → preprocessing → structured data

⚠️ This is still a **work in progress**:

* Model requires further **fine-tuning** for real-world performance
* Additional datasets and preprocessing steps planned
* Scalability testing underway

---

## 💡 Why This Matters

Most nutrition apps rely on **barcode scanning**, which leaves out custom or international food products without barcodes.
This project focuses on **direct label reading**, making it adaptable to more diverse food sources and contexts.

By combining **data engineering, OCR, and applied ML**, this approach opens doors for:

* Smarter calorie and macro tracking
* Integration with healthcare or fitness systems
* Personalized nutrition recommendations

---

## 🔧 Tech Stack

* Python
* OpenCV (image preprocessing & augmentation)
* DocTR OCR
* OpenFoodFacts API
* PyTorch / TensorFlow (for training)
* Streamlit (for demo)

---

## ⚡ Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/nutrition-tracker.git
cd nutrition-tracker
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create .env file to store the model directory path (checkpoints)

```bash
MODEL_DIR=path_to_checkpoints_folder
```

### 5. Launch demo app

```bash
streamlit run app.py
```

---

## 🛠️ Next Steps

* Fine-tune OCR + seq2seq model for better accuracy
* Expand dataset 
* Improve front-end UX for easier adoption

---

## 🙌 Contributions

This is a POC and open to improvement.
If you’d like to collaborate on dataset expansion, model fine-tuning, or deployment, feel free to open an issue or reach out!

---

### 📌 Disclaimer

This project is a **personal experiment** and not intended for clinical or medical use.
Always consult verified sources for dietary and nutritional guidance.
