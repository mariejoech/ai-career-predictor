# 🎯 AI Career Predictor

> An intelligent career recommendation system based on Multiple Intelligence Theory using Deep Learning and MLOps best practices.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)


## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Testing](#testing)
- [Model Performance](#model-performance)
- [MLOps Lifecycle](#mlops-lifecycle)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)


## 🎯 Overview

The AI Career Predictor is a machine learning system that analyzes an individual's Multiple Intelligence profile to recommend suitable career paths. Built following complete MLOps lifecycle practices, this system demonstrates end-to-end AI solution development from data preprocessing to production deployment.

### 🧠 Multiple Intelligence Theory

Based on Howard Gardner's theory, the system evaluates 8 types of intelligence:
- **Linguistic** 📝: Word smart
- **Musical** 🎵: Music smart  
- **Bodily-Kinesthetic** 🤸: Body smart
- **Logical-Mathematical** 🧮: Number smart
- **Spatial** 🎨: Picture smart
- **Interpersonal** 👥: People smart
- **Intrapersonal** 🧘: Self smart
- **Naturalistic** 🌿: Nature smart

## ✨ Features

### 🤖 AI/ML Features
- **Deep Neural Network** with 4 hidden layers and dropout regularization
- **Multi-class Classification** for 20+ career categories
- **Confidence Scoring** for prediction reliability
- **Intelligent Score Validation** with guided input ranges
- **Real-time Inference** with optimized model loading

### 🌐 Web Application
- **Interactive Web Interface** for easy predictions
- **Real-time Predictions** with instant results
- **Responsive Design** for all devices
- **Input Validation** with guided score ranges
- **Visual Recommendations** with confidence percentages
- **Sample Data Testing** for quick demonstration

### 🔧 MLOps & Production
- **Dockerized Deployment** for containerization
- **Real-time Monitoring** with performance tracking
- **Data Drift Detection** for model reliability
- **Health Check Endpoints** for system monitoring
- **Comprehensive Logging** for debugging and analysis
- **Model Versioning** with proper serialization

### 📊 Analytics & Visualization
- **Training History Plots** for model performance
- **Feature Correlation Heatmaps** for data insights
- **Career Distribution Charts** for dataset analysis
- **Intelligence Score Distributions** for data understanding

## 🛠 Technology Stack

### Backend
- **Python 3.9+** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **Flask** - Web framework with CORS support
- **Scikit-learn** - Machine learning utilities
- **Pandas & NumPy** - Data manipulation and analysis
- **Joblib** - Model serialization and persistence

### Frontend
- **HTML5/CSS3** - Modern web standards
- **JavaScript (ES6+)** - Interactive functionality
- **Responsive CSS** - Custom styling with gradients and animations
- **Real-time AJAX** - Asynchronous API communication

### DevOps & Deployment
- **Docker** - Containerization with multi-stage builds
- **Docker Compose** - Multi-container orchestration
- **Health Checks** - Automated system monitoring

### Monitoring & Analytics
- **Custom Monitoring System** - Performance tracking
- **JSON Logging** - Structured prediction logging
- **Matplotlib/Seaborn** - Statistical visualization
- **Data Drift Analysis** - Statistical distribution monitoring

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Option 1: Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-career-predictor.git
   cd ai-career-predictor
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare your dataset**
   ```bash
   # Place your dataset in the data/ folder
   mkdir -p data
   # Copy your Dataset Project 404.xlsx to data/
   ```

5. **Train the model**
   ```bash
   python src/train_model.py
   ```

6. **Run the application**
   ```bash
   python src/app.py
   ```

7. **Open in browser**
   ```
   http://localhost:5000
   ```

### Option 2: Docker Installation

1. **Clone and navigate**
   ```bash
   git clone https://github.com/yourusername/ai-career-predictor.git
   cd ai-career-predictor
   ```

2. **Build and run with Docker Compose**
   ```bash
   # Place your dataset in data/ folder first
   docker-compose up --build
   ```

3. **Access the application**
   ```
   http://localhost:5000
   ```

## 💻 Usage

### Web Interface

1. **Navigate to the application** in your browser
2. **Enter intelligence scores** using the guided ranges:
   - Linguistic Intelligence: 5-14
   - Musical Intelligence: 5-14  
   - Bodily-Kinesthetic: 11-16
   - Logical-Mathematical: 11-16
   - Spatial Intelligence: 6-20
   - Interpersonal: 11-14
   - Intrapersonal: 13-19
   - Naturalistic: 15-19

3. **Click "Predict My Career Path"**
4. **View your results** including:
   - Primary career recommendation
   - Confidence percentage
   - Top 5 alternative careers with scores

### API Usage

#### Predict Career
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "scores": [10.0, 9.5, 13.5, 14.0, 15.0, 12.5, 16.0, 17.0]
  }'
```

#### Health Check
```bash
curl http://localhost:5000/health
```

#### Model Information
```bash
curl http://localhost:5000/model_info
```

#### Test Endpoint
```bash
curl http://localhost:5000/api/test
```

## 🧠 Model Architecture

### Neural Network Design
```
Input Layer (8 features) 
    ↓
Dense Layer (128 neurons, ReLU) → Dropout (0.3)
    ↓  
Dense Layer (64 neurons, ReLU) → Dropout (0.3)
    ↓
Dense Layer (32 neurons, ReLU) → Dropout (0.2)
    ↓
Output Layer (N careers, Softmax)
```

### Training Configuration
- **Optimizer**: Adam with default learning rate
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Batch Size**: 32
- **Early Stopping**: Patience of 15 epochs with best weights restoration
- **Validation Split**: 20% of training data

### Data Preprocessing
- **Feature Scaling**: StandardScaler for intelligence scores
- **Label Encoding**: Categorical encoding for career labels
- **Missing Data**: Mean imputation for NaN values
- **Train/Test Split**: 80/20 stratified split

## 📚 API Documentation

### Endpoints

#### POST `/predict`
Predict career based on intelligence scores.

**Request Body:**
```json
{
  "scores": [10.0, 9.5, 13.5, 14.0, 15.0, 12.5, 16.0, 17.0]
}
```

**Response:**
```json
{
  "predicted_career": "Software Engineer",
  "recommendations": {
    "Software Engineer": 87.5,
    "Data Scientist": 12.3,
    "Teacher": 8.7,
    "Artist": 6.2,
    "Researcher": 4.1
  },
  "confidence": 87.5,
  "input_scores": [10.0, 9.5, 13.5, 14.0, 15.0, 12.5, 16.0, 17.0]
}
```

#### GET `/health`
Check system health status.

**Response:**
```json
{
  "status": "healthy",
  "message": "Career Predictor API is running",
  "model_loaded": true,
  "model_classes": 25
}
```

#### GET `/model_info`
Get detailed model information.

**Response:**
```json
{
  "model_type": "Deep Neural Network",
  "input_features": 8,
  "output_classes": 25,
  "intelligence_types": ["Linguistic", "Musical", ...],
  "score_ranges": {
    "Linguistic": [5, 14],
    "Musical": [5, 14],
    ...
  }
}
```

#### GET `/api/test`
Test endpoint with sample data.

**Response:**
```json
{
  "message": "Test prediction successful",
  "sample_scores": [10.0, 9.5, 13.5, 14.0, 15.0, 12.5, 16.0, 17.0],
  "predicted_career": "Software Engineer",
  "top_3_recommendations": {...}
}
```

## 🐳 Deployment

### Local Development
```bash
python src/app.py
```

### Docker Deployment
```bash
docker-compose up --build
```

### Production Deployment

#### Docker with External Database
```bash
# Build production image
docker build -t career-predictor:prod .

# Run with environment variables
docker run -p 5000:5000 \
  -e FLASK_ENV=production \
  -v $(pwd)/models:/app/models \
  career-predictor:prod
```

#### Cloud Deployment (Example)
```bash
# Deploy to cloud platform of choice
# Configure environment variables
# Set up persistent storage for models
# Configure load balancing if needed
```

## 📊 Monitoring

### Real-time Monitoring Features
The system includes comprehensive monitoring capabilities:

- **Performance Tracking**: Response time and accuracy monitoring
- **Data Drift Detection**: Statistical analysis of input distributions
- **Prediction Logging**: Detailed logs with timestamps and confidence scores
- **Health Monitoring**: System status and model availability checks

### Monitoring Components

#### Data Drift Detection
```python
# Automatic detection of input distribution changes
drift_result = monitor.check_data_drift(new_scores)
if drift_result['drift_detected']:
    print(f"Drift severity: {drift_result['severity']}")
```

#### Performance Logging
```python
# Comprehensive prediction logging
monitor.log_prediction(
    input_scores=scores,
    prediction=predicted_career,
    confidence=confidence_score
)
```

### Log Files
- `logs/predictions.json` - Detailed prediction history
- Model performance metrics tracked in memory
- System health status via `/health` endpoint

## 🧪 Testing

### Current Testing Approach
- **Manual Testing** via `/api/test` endpoint with sample data
- **Input Validation** with comprehensive range checking
- **Health Check Monitoring** for system status verification
- **Sample Data Validation** during model training process
- **Error Handling** with detailed error messages and stack traces

### Testing Examples
```bash
# Test with sample data
curl http://localhost:5000/api/test

# Test health status
curl http://localhost:5000/health

# Test with custom data
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"scores": [10,9,13,14,15,12,16,17]}'
```

### Validation Features
- Score range validation for each intelligence type
- Input data type checking
- Model availability verification
- Graceful error handling with user-friendly messages

## 📈 Model Performance

### Current Evaluation Approach
- **Train/Test Split**: 80/20 with stratified sampling to maintain class distribution
- **Early Stopping**: Prevents overfitting with patience=15 epochs
- **Validation Monitoring**: Real-time tracking of loss and accuracy during training
- **Best Weights Restoration**: Automatically restores best performing model weights

### Performance Metrics
- **Test Accuracy**: Reported after training completion (varies by dataset)
- **Multi-class Classification**: Successfully handles 20+ career categories
- **Confidence Scoring**: Probability-based recommendations with percentage confidence
- **Top-K Predictions**: Returns top 5 career recommendations with confidence scores

### Model Evaluation Features
- Automatic classification report generation for smaller datasets
- Confusion matrix analysis capability
- Training history visualization with accuracy and loss curves
- Model architecture summary with parameter counting

### Visualizations Generated
- Training/validation accuracy and loss curves
- Intelligence type correlation heatmap
- Career distribution in dataset
- Intelligence score distributions

## 🔄 MLOps Lifecycle

This project demonstrates implementation of key MLOps lifecycle stages:

### 1. Problem Definition ✅
- **Business Problem**: Career recommendation based on Multiple Intelligence Theory
- **Success Metrics**: Prediction accuracy and user satisfaction
- **Target Audience**: Students, career changers, HR professionals

### 2. Data Collection & Cleaning ✅
- **Data Source**: Excel dataset with intelligence scores and career labels
- **Preprocessing**: Automatic column detection and NaN value handling
- **Data Validation**: Score range validation and type checking

### 3. Exploratory Data Analysis ✅
- **Statistical Analysis**: Correlation analysis between intelligence types
- **Data Visualization**: Distribution plots and correlation heatmaps
- **Dataset Insights**: Career distribution and intelligence score patterns

### 4. Model Building & Training ✅
- **Architecture**: Deep neural network with dropout regularization
- **Training Strategy**: Early stopping with validation monitoring
- **Hyperparameters**: Optimized dropout rates and layer sizes

### 5. Evaluation & Validation ✅
- **Performance Metrics**: Accuracy, classification reports
- **Validation Strategy**: Train/test split with stratified sampling
- **Model Selection**: Best weights restoration based on validation loss

### 6. Deployment ✅
- **Containerization**: Docker deployment with health checks
- **API Development**: RESTful API with comprehensive endpoints
- **Web Interface**: User-friendly prediction interface

### 7. Monitoring ✅
- **Data Drift Detection**: Statistical monitoring of input distributions
- **Prediction Logging**: Comprehensive logging with structured data
- **Health Monitoring**: System status and model availability tracking

## 🚀 Future Improvements

### Testing Enhancements
- **Unit Testing**: Implement pytest framework for component testing
- **Integration Testing**: End-to-end API testing with automated test suites
- **Performance Testing**: Load testing and response time optimization
- **Model Testing**: Automated model validation and regression testing

### Advanced MLOps Features
- **CI/CD Pipeline**: GitHub Actions for automated testing and deployment
- **Model Versioning**: MLflow integration for experiment tracking
- **A/B Testing**: Compare different model versions in production
- **Automated Retraining**: Trigger retraining based on performance degradation

### Monitoring & Alerting
- **Email/Slack Alerts**: Automated notifications for system issues
- **Dashboard Development**: Real-time monitoring dashboard with Grafana
- **Advanced Metrics**: Business metrics and user engagement tracking
- **Log Analysis**: Automated log analysis and anomaly detection

### Model Improvements
- **Cross-Validation**: Implement k-fold cross-validation for robust evaluation
- **Feature Engineering**: Advanced feature selection and importance analysis
- **Ensemble Methods**: Combine multiple models for improved accuracy
- **Hyperparameter Tuning**: Automated optimization with Optuna or similar

### Data & Features
- **Data Augmentation**: Synthetic data generation for rare career categories
- **Feature Expansion**: Include demographic and educational features
- **Real-time Data**: Integration with live career market data
- **Feedback Loop**: User feedback collection for continuous improvement

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add documentation for new features
- Update README for significant changes
- Ensure Docker builds successfully
- Test locally before submitting


## 🙏 Acknowledgments

- Howard Gardner for Multiple Intelligence Theory framework
- AI Bootcamp instructors and mentors for guidance and support
- Open source contributors to TensorFlow, Flask, and other libraries
- The AI/ML community for inspiration and best practices
- Fellow bootcamp participants for collaboration and feedback


## 🎓 Project Context

This project was developed as part of an AI Bootcamp final project, demonstrating:
- Complete MLOps lifecycle implementation
- Real-world problem solving with AI
- Production-ready deployment practices
- Comprehensive documentation and testing

---

**Built with ❤️ using Python, TensorFlow, and MLOps best practices**