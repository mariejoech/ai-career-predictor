import json
import numpy as np
from datetime import datetime
import logging
import os

class ModelMonitor:
    def __init__(self, log_file='logs/predictions.json'):
        self.log_file = log_file
        self.predictions_log = []
        self.performance_metrics = {}
        
        # Create logs directory
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Load existing logs
        self._load_existing_logs()
        
    def _load_existing_logs(self):
        """Load existing prediction logs"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    self.predictions_log = json.load(f)
        except Exception as e:
            print(f"Could not load existing logs: {e}")
            self.predictions_log = []
    
    def log_prediction(self, input_scores, prediction, confidence, user_id=None):
        """Log each prediction for monitoring"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'input_scores': input_scores,
            'prediction': prediction,
            'confidence': confidence,
            'model_version': '1.0'
        }
        
        self.predictions_log.append(log_entry)
        
        # Save to file
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.predictions_log, f, indent=2)
        except Exception as e:
            print(f"Could not save prediction log: {e}")
    
    def check_data_drift(self, new_scores, reference_stats=None):
        """Simple data drift detection"""
        if reference_stats is None:
            # Use default reference statistics from training data
            reference_stats = {
                'means': [9.5, 9.5, 13.5, 13.5, 13.0, 12.5, 16.0, 17.0],
                'stds': [2.5, 2.5, 1.5, 1.5, 4.0, 1.0, 2.0, 1.5]
            }
        
        drift_detected = False
        drift_features = []
        drift_scores = []
        
        for i, (score, ref_mean, ref_std) in enumerate(zip(new_scores, 
                                                          reference_stats['means'], 
                                                          reference_stats['stds'])):
            # Calculate z-score
            z_score = abs(score - ref_mean) / ref_std if ref_std > 0 else 0
            
            # Check if score is more than 2 standard deviations from reference
            if z_score > 2:
                drift_detected = True
                drift_features.append(i)
                drift_scores.append(z_score)
                
        return {
            'drift_detected': drift_detected,
            'drift_features': drift_features,
            'drift_scores': drift_scores,
            'severity': 'high' if any(score > 3 for score in drift_scores) else 'medium' if drift_detected else 'low'
        }
    
    def get_performance_summary(self):
        """Get performance summary for monitoring"""
        if not self.predictions_log:
            return {"status": "No predictions logged yet"}
            
        confidences = [entry['confidence'] for entry in self.predictions_log]
        recent_predictions = self.predictions_log[-10:] if len(self.predictions_log) >= 10 else self.predictions_log
        
        # Count predictions by career
        career_counts = {}
        for entry in self.predictions_log:
            career = entry['prediction']
            career_counts[career] = career_counts.get(career, 0) + 1
        
        # Get most common prediction
        most_common_career = max(career_counts.items(), key=lambda x: x[1]) if career_counts else ("None", 0)
        
        return {
            "total_predictions": len(self.predictions_log),
            "average_confidence": np.mean(confidences),
            "min_confidence": min(confidences),
            "max_confidence": max(confidences),
            "last_prediction": self.predictions_log[-1]['timestamp'],
            "most_common_prediction": most_common_career[0],
            "prediction_distribution": career_counts,
            "recent_predictions": len(recent_predictions),
            "confidence_trend": "stable"  # Could implement actual trend analysis
        }
    
    def generate_model_report(self):
        """Generate a comprehensive monitoring report"""
        summary = self.get_performance_summary()
        
        report = {
            "report_generated": datetime.now().isoformat(),
            "model_health": "healthy" if summary.get("total_predictions", 0) > 0 else "needs_attention",
            "performance_summary": summary,
            "recommendations": self._generate_recommendations(summary),
            "data_quality_alerts": [],
            "system_status": "operational"
        }
        
        return report
    
    def _generate_recommendations(self, summary):
        """Generate recommendations based on monitoring data"""
        recommendations = []
        
        total_predictions = summary.get("total_predictions", 0)
        avg_confidence = summary.get("average_confidence", 0)
        
        if total_predictions < 10:
            recommendations.append("Collect more prediction data for better monitoring insights")
        
        if avg_confidence < 50:
            recommendations.append("Average confidence is low - consider model retraining")
        
        if avg_confidence > 95:
            recommendations.append("Very high confidence may indicate overfitting - validate with new data")
        
        recommendations.append("Regular model validation recommended every 30 days")
        recommendations.append("Monitor for data drift with new user inputs")
        
        return recommendations

def generate_improvement_suggestions():
    """Generate improvement suggestions for the MLOps pipeline"""
    suggestions = {
        "data_improvements": [
            "Collect more diverse career data from different industries and regions",
            "Include demographic factors (age, education level, work experience)",
            "Add job satisfaction and career success metrics as target variables",
            "Gather longitudinal data to track career changes over time",
            "Include salary and job market data for more comprehensive predictions"
        ],
        "model_improvements": [
            "Experiment with ensemble methods (Random Forest + Neural Network)",
            "Try different neural network architectures (CNN, LSTM, Transformer)",
            "Implement hyperparameter tuning with Optuna or Hyperband",
            "Add regularization techniques and dropout optimization",
            "Explore attention mechanisms for feature importance",
            "Implement model interpretability with SHAP or LIME"
        ],
        "deployment_improvements": [
            "Implement A/B testing for model versions",
            "Add real-time model retraining pipeline with MLflow",
            "Create monitoring dashboard with Grafana or Streamlit",
            "Implement automated model rollback on performance degradation",
            "Add model versioning and experiment tracking",
            "Implement blue-green deployment strategy"
        ],
        "evaluation_improvements": [
            "Add stratified cross-validation with temporal splits",
            "Implement fairness metrics across demographic groups",
            "Use business metrics (career satisfaction, job tenure)",
            "Add online learning evaluation with feedback loops",
            "Implement statistical significance testing for model comparisons",
            "Create comprehensive model validation framework"
        ],
        "monitoring_improvements": [
            "Implement real-time data drift detection",
            "Add model performance degradation alerts",
            "Create automated retraining triggers",
            "Implement user feedback collection system",
            "Add prediction explanation logging",
            "Create model performance benchmarking system"
        ]
    }
    return suggestions
