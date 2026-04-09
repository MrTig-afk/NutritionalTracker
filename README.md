
# 🥗 NutriScan Pipeline
**Cloud-Native Ingestion & Analytical Warehousing**

NutriScan is an end-to-end Data Engineering platform designed to automate the extraction, archival, and analysis of nutritional data from physical labels. This project implements a **Modern Data Stack** architecture, moving away from traditional row-based storage to a decoupled Data Lakehouse model.

### 🔗 Live Deployment
* **Frontend (Vercel):** `https://nutritional-tracker-delta.vercel.app/`

---

## 🏗️ Pipeline Architecture
The system follows a decoupled **ELT (Extract, Load, Transform)** workflow:

1.  **Ingestion Layer:** A React-based UI designed for high-density data ingestion and real-time pipeline status monitoring.
2.  **Object Storage (Data Lake):** Raw images are archived in **AWS S3** (`ap-southeast-2`). This serves as the "Source of Truth," bypassing the ephemeral storage limitations of web hosts like Render and ensuring data persistence.
3.  **Transformation Layer (AI):** **Gemini 2.5 Flash** performs multimodal inference to convert unstructured label imagery into structured JSON entities.
4.  **Analytical Warehouse:** Metadata is persisted in **DuckDB**. Using a **Schema-on-Read** approach, the raw JSON payload is stored in a native `JSON` column, allowing for flexible SQL querying without rigid schema constraints.

---

## 🛠️ Tech Stack
| Component | Technology |
| :--- | :--- |
| **Frontend** | React, Tailwind CSS, Lucide Icons |
| **Backend** | FastAPI (Python), Boto3 SDK |
| **Data Lake** | AWS S3 |
| **Warehouse** | DuckDB (OLAP Optimized) |
| **LLM Engine** | Google Gemini 2.5 Flash |
| **Hosting** | Vercel (Frontend) & Render (Backend) |

---

## 🚀 How to Use

### 1. Local Setup
* **Configure `.env`**: Add your `GOOGLE_API_KEY`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and `S3_BUCKET_NAME`.
* **Backend**: 
   ```bash
   cd backend
   pip install -r requirements.txt
   python init_db.py  # Initializes the DuckDB warehouse
   python main.py     # Starts the FastAPI server
   ```
* **Frontend**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### 2. Running the Pipeline
1. **Upload**: Select or drag-and-drop a clear image of a nutrition label into the **Staging Area**.
2. **Execute**: Click **RUN_INGESTION**. The system will simultaneously upload the raw asset to **AWS S3** and trigger the **Gemini 2.5** extraction.
3. **Analyze**: View the parsed results in the UI. Metadata is automatically committed to your local **DuckDB** warehouse for future analytical work.
4. **Audit**: Expand the **RAW_JSON_PAYLOAD** section at the bottom to view the exact JSON blob stored in the database.

---

## ⚡ Technical Highlights
* **Storage Decoupling:** Implemented S3 archival to handle unstructured image data, keeping the database lightweight and performant.
* **Schema Resilience:** Developed a transformation layer in the frontend to dynamically map inconsistent OCR keys (e.g., "per 100g" vs "per 100ml") to structured UI views.
* **Environment Management:** Utilized decoupled DDL scripts (`init_db.py`) for reproducible warehouse initialization across different environments.
