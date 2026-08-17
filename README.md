# BookLens 📚🔍

**BookLens** is an AI-powered multilingual book recommendation and discovery engine. It combines semantic vector search (`multilingual-e5-small`), emotion profiling, zero-shot and neural classification, and persona-driven filtering with an interactive Streamlit dashboard.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.10 or 3.11** (Python 3.11 recommended)
- `pip` and `virtualenv`

### 2. Environment Setup

Clone the repository and navigate into the project directory:

```bash
cd book_lens
```

Create and activate a virtual environment:

```bash
# Create virtual environment
python3.11 -m venv .venv

# Activate on macOS/Linux:
source .venv/bin/activate

# Or on Windows:
# .venv\Scripts\activate
```

Install all dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional)* Set up Jupyter kernel for notebooks:

```bash
python -m ipykernel install --user --name booklens --display-name "Python 3 (booklens)"
```

### 3. Environment Variables (Optional)

Copy the example environment file if you wish to configure live Google Books API keys:

```bash
cp .env.example .env
```

---

## 🖥️ Running the Application

### Launch Dashboard
To start the Streamlit web dashboard:

```bash
python scripts/run_dashboard.py
```
*Alternatively, run directly with Streamlit:*
```bash
streamlit run Dashboard.py
```

Open your browser at `http://localhost:8501`.

---

## ⚙️ Offline Pipeline & Notebooks

BookLens uses a two-stage architecture: an **offline pipeline** for data preparation, embedding generation, and model training, and an **online app** for low-latency recommendations.

### Run Full Pipeline
To run all stages end-to-end (`01` through `05`):

```bash
python scripts/run_pipeline.py
```

### Pipeline Options
```bash
# Run pipeline and immediately launch the dashboard on completion:
python scripts/run_pipeline.py --then-run-dashboard

# Resume pipeline from a specific stage:
python scripts/run_pipeline.py --from-stage embeddings --to-stage train_classifiers
```

### Notebook Stages:
- `notebooks/01_data_ingestion.ipynb`: Merges and cleans multi-source book datasets.
- `notebooks/02_data_exploration.ipynb`: Analyzes distributions, genres, and taxonomy balance.
- `notebooks/03_embeddings_and_index.ipynb`: Generates multilingual embeddings and builds ChromaDB vector store.
- `notebooks/04_label_bootstrapping.ipynb`: Bootstraps category and emotion pseudo-labels with teacher models.
- `notebooks/05_train_classifiers.ipynb`: Trains from-scratch neural network classifiers.

---

## 📁 Project Structure

```
book_lens/
├── app/                 # Streamlit dashboard components, pages, config, & charts
├── datasets/            # Raw datasets (Google Books, Nepali books, 7k books)
├── data/                # Processed catalog, spot check data, and vector index (local)
├── models/              # Trained weights (.npz, .npy) and metrics (local)
├── notebooks/           # Offline data ingestion, indexing, and training notebooks
├── scripts/             # CLI runners (run_dashboard.py, run_pipeline.py)
├── src/                 # ML from scratch, persona engine, and live API tools
├── Dashboard.py         # Streamlit main entry point
├── requirements.txt     # Complete Python dependencies
└── README.md            # Project documentation
```
