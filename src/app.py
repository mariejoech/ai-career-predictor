from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os
import traceback

app = Flask(__name__)
CORS(app)

# Global variables for model components
model = None
scaler = None
job_encoder = None
job_mapping = None

def load_model_components():
    """Load all model components"""
    global model, scaler, job_encoder, job_mapping
    
    try:
        model = load_model('models/intelligence_career_predictor.h5')
        scaler = joblib.load('models/intelligence_scaler.pkl')
        job_encoder = joblib.load('models/job_encoder.pkl')
        job_mapping = joblib.load('models/job_mapping.pkl')
        print("✅ Model components loaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Error loading model components: {e}")
        print("Make sure to run 'python src/train_model.py' first to train the model.")
        return False

# Load model components at startup
model_loaded = load_model_components()

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🎯 AI Career Predictor</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 30px;
        }
        
        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }
        
        .form-group input {
            width: 100%;
            padding: 12px 15px;
            border: 2px solid #e1e5e9;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        
        .form-group input:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        
        .predict-btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 40px;
            font-size: 18px;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s ease;
            display: block;
            margin: 0 auto;
            font-weight: 600;
        }
        
        .predict-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        }
        
        .predict-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 15px;
            border-left: 5px solid #667eea;
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
            font-size: 1.4em;
        }
        
        .confidence {
            background: #e3f2fd;
            padding: 10px 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 600;
        }
        
        .recommendations {
            margin-top: 20px;
        }
        
        .recommendations h4 {
            margin-bottom: 15px;
            color: #333;
        }
        
        .job-item {
            background: white;
            padding: 12px 15px;
            margin-bottom: 8px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .job-name {
            font-weight: 500;
        }
        
        .job-percentage {
            color: #667eea;
            font-weight: 600;
        }
        
        .error {
            color: #dc3545;
            background: #f8d7da;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .loading::after {
            content: '';
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 2px solid #667eea;
            border-radius: 50%;
            border-top: 2px solid transparent;
            animation: spin 1s linear infinite;
            margin-left: 10px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .info-section {
            background: #e8f4fd;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
        }
        
        .info-section h3 {
            color: #1976d2;
            margin-bottom: 10px;
        }
        
        @media (max-width: 768px) {
            .form-grid {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
            
            .content {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎯 AI Career Predictor</h1>
            <p>Discover your ideal career path based on Multiple Intelligence Theory</p>
        </div>
        
        <div class="content">
            <div class="info-section">
                <h3>📚 How it works</h3>
                <p>This AI system analyzes your intelligence profile across 8 different types of intelligence to predict the most suitable career paths for you. Enter your scores for each intelligence type below (use the suggested ranges for best results).</p>
            </div>
            
            <form id="predictionForm">
                <div class="form-grid">
                    <div class="form-group">
                        <label for="linguistic">📝 Linguistic Intelligence (5-14)</label>
                        <input type="number" id="linguistic" min="5" max="14" step="0.1" placeholder="e.g., 10.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="musical">🎵 Musical Intelligence (5-14)</label>
                        <input type="number" id="musical" min="5" max="14" step="0.1" placeholder="e.g., 9.0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="bodily">🤸 Bodily-Kinesthetic Intelligence (11-16)</label>
                        <input type="number" id="bodily" min="11" max="16" step="0.1" placeholder="e.g., 13.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="logical">🧮 Logical-Mathematical Intelligence (11-16)</label>
                        <input type="number" id="logical" min="11" max="16" step="0.1" placeholder="e.g., 14.0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="spatial">🎨 Spatial Intelligence (6-20)</label>
                        <input type="number" id="spatial" min="6" max="20" step="0.1" placeholder="e.g., 15.0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="interpersonal">👥 Interpersonal Intelligence (11-14)</label>
                        <input type="number" id="interpersonal" min="11" max="14" step="0.1" placeholder="e.g., 12.5" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="intrapersonal">🧘 Intrapersonal Intelligence (13-19)</label>
                        <input type="number" id="intrapersonal" min="13" max="19" step="0.1" placeholder="e.g., 16.0" required>
                    </div>
                    
                    <div class="form-group">
                        <label for="naturalistic">🌿 Naturalistic Intelligence (15-19)</label>
                        <input type="number" id="naturalistic" min="15" max="19" step="0.1" placeholder="e.g., 17.0" required>
                    </div>
                </div>
                
                <button type="submit" class="predict-btn" id="submitBtn">
                    🔮 Predict My Career Path
                </button>
            </form>
            
            <div id="result" style="display: none;"></div>
        </div>
    </div>

    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const resultDiv = document.getElementById('result');
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.innerHTML = '🔄 Analyzing...';
            resultDiv.innerHTML = '<div class="loading">Analyzing your intelligence profile</div>';
            resultDiv.style.display = 'block';
            
            const scores = [
                parseFloat(document.getElementById('linguistic').value),
                parseFloat(document.getElementById('musical').value),
                parseFloat(document.getElementById('bodily').value),
                parseFloat(document.getElementById('logical').value),
                parseFloat(document.getElementById('spatial').value),
                parseFloat(document.getElementById('interpersonal').value),
                parseFloat(document.getElementById('intrapersonal').value),
                parseFloat(document.getElementById('naturalistic').value)
            ];
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ scores: scores })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    let html = `
                        <h3>🎯 Your Predicted Career Path</h3>
                        <div class="confidence">
                            <strong>Primary Recommendation:</strong> ${data.predicted_career}<br>
                            <strong>Confidence Level:</strong> ${data.confidence.toFixed(1)}%
                        </div>
                        
                        <div class="recommendations">
                            <h4>🏆 Top Career Recommendations</h4>
                    `;
                    
                    for (const [job, probability] of Object.entries(data.recommendations)) {
                        html += `
                            <div class="job-item">
                                <span class="job-name">${job}</span>
                                <span class="job-percentage">${probability.toFixed(1)}%</span>
                            </div>
                        `;
                    }
                    
                    html += '</div>';
                    resultDiv.innerHTML = html;
                } else {
                    resultDiv.innerHTML = `<div class="error">❌ Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultDiv.innerHTML = `<div class="error">❌ Network Error: ${error.message}</div>`;
            } finally {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.innerHTML = '🔮 Predict My Career Path';
            }
        });
        
        // Add sample data button
        function fillSampleData() {
            document.getElementById('linguistic').value = '10.0';
            document.getElementById('musical').value = '9.5';
            document.getElementById('bodily').value = '13.5';
            document.getElementById('logical').value = '14.0';
            document.getElementById('spatial').value = '15.0';
            document.getElementById('interpersonal').value = '12.5';
            document.getElementById('intrapersonal').value = '16.0';
            document.getElementById('naturalistic').value = '17.0';
        }
        
        // Add sample data button after form
        document.querySelector('.predict-btn').insertAdjacentHTML('afterend', 
            '<button type="button" onclick="fillSampleData()" style="background: #28a745; color: white; border: none; padding: 10px 20px; margin: 10px; border-radius: 5px; cursor: pointer;">📝 Fill Sample Data</button>'
        );
    </script>
</body>
</html>
"""

@app.route('/')
def home():
    """Home page with prediction interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy" if model_loaded else "unhealthy",
        "message": "Career Predictor API is running",
        "model_loaded": model_loaded,
        "model_classes": len(job_mapping) if job_mapping else 0
    })

@app.route('/predict', methods=['POST'])
def predict_career():
    """Main prediction endpoint"""
    if not model_loaded:
        return jsonify({
            "error": "Model not loaded. Please run 'python src/train_model.py' first to train the model."
        }), 500
        
    try:
        data = request.json
        intelligence_scores = data['scores']
        
        if len(intelligence_scores) != 8:
            return jsonify({"error": "Expected 8 intelligence scores"}), 400
        
        # Validate score ranges
        score_ranges = [(5, 14), (5, 14), (11, 16), (11, 16), (6, 20), (11, 14), (13, 19), (15, 19)]
        intelligence_names = ['Linguistic', 'Musical', 'Bodily', 'Logical', 'Spatial', 'Interpersonal', 'Intrapersonal', 'Naturalistic']
        
        for i, (score, (min_val, max_val)) in enumerate(zip(intelligence_scores, score_ranges)):
            if not (min_val <= score <= max_val):
                return jsonify({
                    "error": f"{intelligence_names[i]} score ({score}) is out of range [{min_val}-{max_val}]"
                }), 400
        
        # Preprocess input
        scores_array = np.array(intelligence_scores).reshape(1, -1)
        scaled_scores = scaler.transform(scores_array)
        
        # Make prediction
        prediction_prob = model.predict(scaled_scores, verbose=0)[0]
        
        # Get top 5 predictions
        top_indices = prediction_prob.argsort()[-5:][::-1]
        top_probabilities = prediction_prob[top_indices]
        
        recommendations = {}
        for idx, prob in zip(top_indices, top_probabilities):
            job_name = job_mapping[idx]
            recommendations[job_name] = float(prob * 100)
        
        return jsonify({
            "predicted_career": job_mapping[top_indices[0]],
            "recommendations": recommendations,
            "confidence": float(top_probabilities[0] * 100),
            "input_scores": intelligence_scores
        })
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "traceback": traceback.format_exc()
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
        
    return jsonify({
        "model_type": "Deep Neural Network",
        "input_features": 8,
        "output_classes": len(job_mapping),
        "intelligence_types": [
            "Linguistic", "Musical", "Bodily-Kinesthetic", 
            "Logical-Mathematical", "Spatial", "Interpersonal", 
            "Intrapersonal", "Naturalistic"
        ],
        "score_ranges": {
            "Linguistic": [5, 14],
            "Musical": [5, 14],
            "Bodily-Kinesthetic": [11, 16],
            "Logical-Mathematical": [11, 16],
            "Spatial": [6, 20],
            "Interpersonal": [11, 14],
            "Intrapersonal": [13, 19],
            "Naturalistic": [15, 19]
        },
        "total_jobs": len(job_mapping) if job_mapping else 0
    })

@app.route('/api/test', methods=['GET'])
def test_prediction():
    """Test endpoint with sample data"""
    if not model_loaded:
        return jsonify({"error": "Model not loaded"}), 500
    
    sample_scores = [10.0, 9.5, 13.5, 14.0, 15.0, 12.5, 16.0, 17.0]
    
    try:
        scores_array = np.array(sample_scores).reshape(1, -1)
        scaled_scores = scaler.transform(scores_array)
        prediction_prob = model.predict(scaled_scores, verbose=0)[0]
        
        top_indices = prediction_prob.argsort()[-3:][::-1]
        top_probabilities = prediction_prob[top_indices]
        
        recommendations = {
            job_mapping[idx]: float(prob * 100) 
            for idx, prob in zip(top_indices, top_probabilities)
        }
        
        return jsonify({
            "message": "Test prediction successful",
            "sample_scores": sample_scores,
            "predicted_career": job_mapping[top_indices[0]],
            "top_3_recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"error": f"Test failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Career Predictor API...")
    if model_loaded:
        print("✅ Model loaded successfully!")
        print("🌐 Open http://localhost:5000 in your browser")
    else:
        print("❌ Model not loaded. Run 'python src/train_model.py' first.")
    
    app.run(host='0.0.0.0', port=5000, debug=True)