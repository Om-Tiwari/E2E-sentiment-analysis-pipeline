from flask import Flask, request, jsonify, render_template_string
import pickle
import re
from bs4 import BeautifulSoup
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS if needed

# Load model and vectorizer
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("model/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
except Exception as e:
    raise RuntimeError(f"Failed to load model files: {str(e)}")

# Preprocessing function
def clean_text(text):
    try:
        text = BeautifulSoup(text, "html.parser").get_text()
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower().strip()
    except Exception as e:
        raise ValueError(f"Text cleaning failed: {str(e)}")

@app.route('/')
def home():
    # Simple HTML form for browser testing
    return render_template_string('''
        <html>
            <head><title>Sentiment Analysis Demo</title></head>
            <body>
                <h1>Test Sentiment Analysis</h1>
                <form id="predForm">
                    <textarea name="review_text" rows="5" cols="50" 
                        placeholder="Enter movie review..."></textarea><br>
                    <button type="submit">Predict Sentiment</button>
                </form>
                <div id="result"></div>
                <script>
                    document.getElementById('predForm').onsubmit = async (e) => {
                        e.preventDefault();
                        const formData = new FormData(e.target);
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {'Content-Type': 'application/json'},
                            body: JSON.stringify({
                                review_text: formData.get('review_text')
                            })
                        });
                        const result = await response.json();
                        document.getElementById('result').innerHTML = `
                            <h3>Prediction: ${result.sentiment_prediction}</h3>
                            ${response.ok ? '' : '<p style="color:red">Error: ' + result.error + '</p>'}
                        `;
                    };
                </script>
            </body>
        </html>
    ''')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        if 'review_text' not in data:
            return jsonify({"error": "Missing 'review_text' field"}), 400
            
        raw_text = data['review_text']
        if not isinstance(raw_text, str) or len(raw_text.strip()) == 0:
            return jsonify({"error": "Invalid text input"}), 400
            
        cleaned_text = clean_text(raw_text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        
        return jsonify({
            "sentiment_prediction": prediction,
            "cleaned_text": cleaned_text
        })
        
    except Exception as e:
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "tfidf_loaded": tfidf is not None
    }), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)