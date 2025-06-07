import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

class CareerPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.job_encoder = LabelEncoder()
        self.job_mapping = {}
        self.intelligence_columns = ['Lingu', 'Music', 'Bodil', 'Logic', 'Spati', 'Interp', 'Intrap', 'Natur']
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the dataset"""
        print("Loading dataset...")
        df = pd.read_excel(file_path, sheet_name='original')
        
        # Create directories
        os.makedirs('visualizations', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        print("Dataset shape:", df.shape)
        print("Columns:", list(df.columns))
        
        # Find intelligence columns
        column_indices = self._find_intelligence_columns(df)
        
        # Find job column
        job_col_idx = self._find_job_column(df)
        
        # Process data
        X, y = self._extract_features_and_target(df, column_indices, job_col_idx)
        
        return X, y, df, column_indices
    
    def _find_intelligence_columns(self, df):
        """Find intelligence score columns in dataset"""
        column_indices = {}
        
        # First try to find by partial name matching
        for col_name in self.intelligence_columns:
            for i, dataset_col in enumerate(df.columns):
                if col_name.lower() in str(dataset_col).lower():
                    column_indices[col_name] = i
                    print(f"Found '{col_name}' at column {i}: {dataset_col}")
                    break
        
        # If we don't find enough columns, use manual indices
        if len(column_indices) < 8:
            print("Using manual column indices...")
            manual_indices = {
                'Lingu': 4,   # Linguistic
                'Music': 5,   # Musical
                'Bodil': 6,   # Bodily/kinesthetic
                'Logic': 7,   # Logical
                'Spati': 8,   # Spatial
                'Interp': 9,  # Interpersonal
                'Intrap': 10, # Intrapersonal
                'Natur': 11   # Naturalistic
            }
            for col_name, idx in manual_indices.items():
                if idx < len(df.columns):
                    column_indices[col_name] = idx
                    print(f"Using column {idx}: {df.columns[idx]} for {col_name}")
            
        return column_indices
    
    def _find_job_column(self, df):
        """Find job/career column in dataset"""
        for i, col in enumerate(df.columns):
            if any(keyword in str(col).lower() for keyword in ['job', 'career', 'profession']):
                print(f"Found job column at index {i}: {col}")
                return i
        
        # Default fallback
        job_col_idx = 2
        print(f"Using default job column at index {job_col_idx}: {df.columns[job_col_idx]}")
        return job_col_idx
    
    def _extract_features_and_target(self, df, column_indices, job_col_idx):
        """Extract features and target from dataset"""
        X_columns = list(column_indices.values())
        
        # Convert to numeric and handle NaN
        for col_idx in X_columns:
            df.iloc[:, col_idx] = pd.to_numeric(df.iloc[:, col_idx], errors='coerce')
            nan_count = df.iloc[:, col_idx].isna().sum()
            if nan_count > 0:
                mean_val = df.iloc[:, col_idx].mean()
                df.iloc[:, col_idx] = df.iloc[:, col_idx].fillna(mean_val)
                print(f"Filled {nan_count} NaN values in column {col_idx} with mean: {mean_val:.2f}")
        
        X = df.iloc[:, X_columns].values
        y = df.iloc[:, job_col_idx].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Unique jobs: {len(np.unique(y))}")
        
        return X, y
    
    def create_visualizations(self, df, X, y, column_indices):
        """Create all visualizations"""
        print("Creating visualizations...")
        
        # Job distribution
        self._plot_job_distribution(y)
        
        # Intelligence distributions
        self._plot_intelligence_distributions(df, column_indices)
        
        # Correlation heatmap
        self._plot_correlation_heatmap(X)
        
        print("✅ All visualizations saved to 'visualizations/' folder")
    
    def _plot_job_distribution(self, y):
        """Plot job category distribution"""
        unique_jobs, counts = np.unique(y, return_counts=True)
        job_counts = dict(zip(unique_jobs, counts))
        sorted_jobs = sorted(job_counts.items(), key=lambda x: x[1], reverse=True)
        
        plt.figure(figsize=(15, 8))
        jobs_to_plot = [job for job, count in sorted_jobs[:20]]
        counts_to_plot = [count for job, count in sorted_jobs[:20]]
        plt.barh(jobs_to_plot, counts_to_plot)
        plt.xlabel('Number of Instances')
        plt.ylabel('Job Category')
        plt.title('Distribution of Top 20 Job Categories')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('visualizations/job_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print(f"📊 Job distribution saved. Top 5 jobs: {sorted_jobs[:5]}")
    
    def _plot_intelligence_distributions(self, df, column_indices):
        """Plot intelligence score distributions"""
        plt.figure(figsize=(15, 10))
        intelligence_names = list(column_indices.keys())
        
        for i, (intel_name, col_idx) in enumerate(column_indices.items()):
            plt.subplot(2, 4, i+1)
            data = df.iloc[:, col_idx]
            sns.histplot(data, kde=True)
            plt.title(f"{intel_name} Intelligence")
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            
        plt.tight_layout()
        plt.savefig('visualizations/intelligence_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("📊 Intelligence distributions saved")
    
    def _plot_correlation_heatmap(self, X):
        """Plot correlation between intelligence scores"""
        corr = np.corrcoef(X.T)
        intelligence_names = ['Linguistic', 'Musical', 'Bodily', 'Logical', 'Spatial', 'Interpersonal', 'Intrapersonal', 'Naturalistic']
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', mask=mask,
                   xticklabels=intelligence_names, yticklabels=intelligence_names)
        plt.title('Correlation Between Intelligence Types')
        plt.tight_layout()
        plt.savefig('visualizations/intelligence_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("📊 Intelligence correlation heatmap saved")
    
    def build_and_train_model(self, X, y):
        """Build and train the neural network"""
        print("Building and training model...")
        
        # Encode labels
        y_encoded = self.job_encoder.fit_transform(y)
        self.job_mapping = {i: job for i, job in enumerate(self.job_encoder.classes_)}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Build model
        input_dim = X_train.shape[1]
        output_dim = len(np.unique(y_encoded))
        
        print(f"Input dimensions: {input_dim}")
        print(f"Output classes: {output_dim}")
        
        self.model = Sequential([
            Dense(128, activation='relu', input_dim=input_dim),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(output_dim, activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model architecture:")
        self.model.summary()
        
        # Train model
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=1
        )
        
        print("\n🔥 Starting model training...")
        history = self.model.fit(
            X_train_scaled, y_train,
            batch_size=32,
            epochs=100,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Evaluate
        print("\n📊 Evaluating model...")
        y_pred = np.argmax(self.model.predict(X_test_scaled, verbose=0), axis=1)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Test Accuracy: {accuracy * 100:.2f}%")
        
        # Classification report for top classes
        unique_test_classes = np.unique(y_test)
        if len(unique_test_classes) <= 10:
            target_names = [self.job_mapping[i] for i in unique_test_classes]
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, labels=unique_test_classes, target_names=target_names))
        else:
            print(f"\nToo many classes ({len(unique_test_classes)}) for detailed classification report")
        
        # Plot training history
        self._plot_training_history(history)
        
        return history, accuracy
    
    def _plot_training_history(self, history):
        """Plot training history"""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('visualizations/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()  # Close the figure to free memory
        
        print("📊 Training history saved")
    
    def save_model(self):
        """Save model and components"""
        self.model.save('models/intelligence_career_predictor.h5')
        joblib.dump(self.scaler, 'models/intelligence_scaler.pkl')
        joblib.dump(self.job_encoder, 'models/job_encoder.pkl')
        joblib.dump(self.job_mapping, 'models/job_mapping.pkl')
        print("✅ Model and components saved successfully!")
    
    def predict_career(self, intelligence_scores):
        """Predict career from intelligence scores"""
        scores_array = np.array(intelligence_scores).reshape(1, -1)
        scaled_scores = self.scaler.transform(scores_array)
        prediction_prob = self.model.predict(scaled_scores, verbose=0)[0]
        
        # Get top 5 predictions
        top_indices = prediction_prob.argsort()[-5:][::-1]
        top_probabilities = prediction_prob[top_indices]
        
        recommendations = {
            self.job_mapping[idx]: float(prob * 100) 
            for idx, prob in zip(top_indices, top_probabilities)
        }
        
        return self.job_mapping[top_indices[0]], recommendations

def main():
    """Main execution function"""
    print("🚀 Starting Career Prediction Model Training...")
    print("=" * 50)
    
    # Initialize predictor
    predictor = CareerPredictor()
    
    # Load and preprocess data
    dataset_path = "data/Dataset Project 404.xlsx"
    
    try:
        X, y, df, column_indices = predictor.load_and_preprocess_data(dataset_path)
    except FileNotFoundError:
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please place your dataset file in the 'data' folder")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Create visualizations
    predictor.create_visualizations(df, X, y, column_indices)
    
    # Train model
    history, accuracy = predictor.build_and_train_model(X, y)
    
    # Save model
    predictor.save_model()
    
    # Demo prediction
    print("\n" + "=" * 50)
    print("🔮 Testing Model with Sample Data...")
    sample_scores = [10, 10, 13.5, 13.5, 13, 12.5, 16, 17]
    print(f"Sample intelligence scores: {sample_scores}")
    
    try:
        predicted_job, recommendations = predictor.predict_career(sample_scores)
        print(f"\n🎯 Predicted career: {predicted_job}")
        print("\n🏆 Top 5 recommendations:")
        for i, (job, prob) in enumerate(recommendations.items(), 1):
            print(f"  {i}. {job}: {prob:.2f}%")
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Training completed successfully!")
    print("\n📁 Generated files:")
    print("   • models/intelligence_career_predictor.h5")
    print("   • models/intelligence_scaler.pkl") 
    print("   • models/job_encoder.pkl")
    print("   • models/job_mapping.pkl")
    print("   • visualizations/job_distribution.png")
    print("   • visualizations/intelligence_distributions.png")
    print("   • visualizations/intelligence_correlation.png")
    print("   • visualizations/training_history.png")
    print("\n🚀 Next step: Run 'python src/app.py' to start the web interface!")

if __name__ == "__main__":
    main()