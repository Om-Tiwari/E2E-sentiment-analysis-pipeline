# End-to-End Sentiment Analysis Pipeline

A complete sentiment analysis solution with Flask API, featuring:
- IMDB movie review classification
- Database integration (SQLite)
- Browser testing interface
- Production-ready error handling

## Project Structure
```
sentiment-analysis/
├── model/                   # Machine Learning artifacts
│   ├── model.pkl            # Trained classifier
│   └── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── src/                     # Application source
│   ├── app.py               # Flask REST API
│   ├── data_ingestion.py    # Data loading script
│   └── model_training.py    # Model training code
├── pyproject.toml           # Dependency specifications
├── uv.lock                  # Version-locked dependencies
├── imdb_reviews.db          # SQLite database (auto-generated)
└── .gitignore               # Ignore build artifacts
```

## 🚀 Installation

1. **Install UV package manager**:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Set up environment**:
```bash
uv venv .venv            # Create virtual environment
source .venv/bin/activate # Activate (Linux/Mac)
# .\.venv\Scripts\activate # Windows
```

3. **Install dependencies**:
```bash
uv pip sync              # Install from uv.lock
```

## ▶️ Running the Application

```bash
python src/app.py
```

**Access endpoints**:
- Browser interface: `http://localhost:5000`
- API endpoint: `POST http://localhost:5000/predict`
- Health check: `GET http://localhost:5000/health`

## 🧪 Testing the API

### Browser Testing
1. Visit `http://localhost:5000`
2. Enter review text in the form
3. Click "Predict Sentiment"

### Command Line (curl)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text": "This movie completely captivated me!"}'
```

### Python (requests)
```python
import requests

response = requests.post(
    "http://localhost:5000/predict",
    json={"review_text": "Worst film I've ever seen."}
)
print(response.json())
```

## 🔧 Development Workflow

**Add new dependencies**:
```bash
uv add <package>  # Adds to pyproject.toml
uv lock        # Update lock file
```

**Rebuild environment**:
```bash
uv sync  # Fresh install from lock file
```

## 🛠️ Troubleshooting

**Common Issues**:
1. **404 Errors**: Ensure server is running and endpoint URLs are correct
2. **405 Errors**: Use POST method for `/predict` endpoint
3. **Model Loading Failures**:
   - Verify model files exist in `model/`
   - Check file paths in `app.py`

**Health Check**:
```bash
curl http://localhost:5000/health
# Should return {"model_loaded":true,"status":"healthy","tfidf_loaded":true}
```

## 📚 Documentation
- Model Architecture: Logistic Regression with TF-IDF features
- Accuracy: ~89% on IMDB test set
- Data Source: Hugging Face Datasets ("imdb")

```
