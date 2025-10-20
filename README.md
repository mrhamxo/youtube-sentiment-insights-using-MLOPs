# YouTube Sentiment Insights using MLOps

A production-ready sentiment analysis system for YouTube comments, built with MLOps best practices including DVC pipeline orchestration, MLflow experiment tracking, and automated model deployment.

## 🎯 Overview

This project analyzes YouTube comments to classify sentiment (Positive, Neutral, Negative) using a LightGBM classifier with TF-IDF features. The system includes a complete ML pipeline with data versioning, experiment tracking, model registry, and a Flask API with Streamlit frontend for real-time predictions.

## 🏗️ Architecture

```
├── data/                    # Data storage (tracked by DVC)
├── src/
│   ├── data/               # Data ingestion & preprocessing
│   └── model/              # Model building, evaluation & registration
├── flask_app/              # REST API for predictions
├── frontend/               # Streamlit dashboard
└── dvc.yaml                # ML pipeline definition
```

## ✨ Features

- **Automated ML Pipeline**: End-to-end pipeline with DVC for reproducibility
- **Experiment Tracking**: MLflow integration with DagsHub for experiment management
- **Model Registry**: Automated model versioning and staging
- **REST API**: Flask-based API for sentiment predictions
- **Interactive Dashboard**: Streamlit frontend with visualizations
- **Sentiment Analysis**: Pie charts, word clouds, and trend graphs
- **Containerization**: Docker support for easy deployment

## 🚀 Quick Start

### Prerequisites

```bash
python 3.9+
pip
docker (optional)
```

### Installation

```bash
# Clone repository
git clone https://github.com/mr.hamxa942/youtube-sentiment-insights-using-MLOPs.git
cd youtube-sentiment-insights-using-MLOPs

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
```

### Run Pipeline

```bash
# Initialize DVC
dvc repro

# This runs the complete pipeline:
# 1. Data ingestion
# 2. Data preprocessing
# 3. Model building
# 4. Model evaluation
# 5. Model registration
```

### Start API Server

```bash
# Run Flask API
cd flask_app
python app.py

# Server runs on http://localhost:5000
```

### Launch Dashboard

```bash
# Run Streamlit frontend
cd frontend
streamlit run streamlit.py

# Dashboard available at http://localhost:8501
```

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Predict sentiment for comments |
| `/predict_with_timestamps` | POST | Predict with timestamp data |
| `/generate_chart` | POST | Generate sentiment distribution chart |
| `/generate_wordcloud` | POST | Generate word cloud visualization |
| `/generate_trend_graph` | POST | Generate sentiment trend over time |

### Example Request

```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"comments": ["This video is amazing!", "Not worth my time"]}
)
print(response.json())
```

## 🔧 Configuration

Model hyperparameters are defined in `params.yaml`:

```yaml
model_building:
  ngram_range: [1, 3]
  max_features: 1000
  learning_rate: 0.09
  max_depth: 20
  n_estimators: 367
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t youtube-sentiment .

# Run container
docker run -p 5000:5000 youtube-sentiment
```

## 📈 Model Performance

The LightGBM model is trained on preprocessed Reddit comments with:
- TF-IDF vectorization (1000 features, trigrams)
- Balanced class weights
- L1/L2 regularization

Model metrics and artifacts are tracked in MLflow and available on DagsHub.

## 🛠️ Technology Stack

- **ML Framework**: LightGBM, scikit-learn
- **MLOps**: DVC, MLflow, DagsHub
- **API**: Flask, Flask-CORS
- **Frontend**: Streamlit
- **NLP**: NLTK, TF-IDF
- **Visualization**: Matplotlib, WordCloud, Seaborn

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Muhammad Hamza**  
Email: mr.hamxa942@gmail.com
