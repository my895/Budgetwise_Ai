#!/usr/bin/env python3
"""
Baseline Model for Budget Tool MVP
Simple statistical baseline using mean spending per category
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
try:
    from src.components.category_standardizer import standardizer
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.components.category_standardizer import standardizer

def setup_logging():
    """Setup logging for baseline model"""
    log_dir = Path("Checkpoints/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / "baseline_model.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

class BaselineModel:
    """
    Simple baseline model that predicts spending based on category averages
    """
    
    def __init__(self):
        self.category_means = {}
        self.total_mean = 0
        self.is_trained = False
    
    def fit(self, df):
        """Train the baseline model"""
        logging.info("Training baseline model...")
        
        # Ensure categories are standardized
        df_standardized = standardizer.standardize_dataframe(df, 'category')
        logging.info(f"Categories standardized for training: {sorted(df_standardized['category'].unique())}")
        
        # Calculate mean spending per category
        self.category_means = df_standardized.groupby('category')['amount'].mean().to_dict()
        self.total_mean = df_standardized['amount'].mean()
        
        logging.info(f"Category means calculated: {len(self.category_means)} standardized categories")
        logging.info(f"Overall mean spending: {self.total_mean:.2f}")
        
        self.is_trained = True
        return self
    
    def predict(self, categories):
        """Predict spending for given categories"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        predictions = []
        for category in categories:
            # Standardize the category before prediction
            standardized_category = standardizer.standardize_category(category)
            # Use category mean if available, otherwise use overall mean
            pred = self.category_means.get(standardized_category, self.total_mean)
            predictions.append(pred)
        
        return np.array(predictions)
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'category_means': self.category_means,
            'total_mean': self.total_mean,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"Baseline model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.category_means = model_data['category_means']
        self.total_mean = model_data['total_mean']
        self.is_trained = model_data['is_trained']
        
        logging.info(f"Baseline model loaded from {filepath}")
        return self

def train_baseline_model():
    """Train and save the baseline model"""
    setup_logging()
    logging.info("Starting baseline model training...")
    
    # Load preprocessed data
    data_path = Path("Checkpoints/logs/data_preprocessed.csv")
    if not data_path.exists():
        logging.error(f"Preprocessed data not found at {data_path}")
        print(f"❌ ERROR: Preprocessed data not found. Run data preprocessing first.")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    logging.info(f"Loaded data: {df.shape}")
    
    # Train baseline model
    model = BaselineModel()
    model.fit(df)
    
    # Save model
    model_dir = Path("Checkpoints/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "baseline_model.pkl"
    model.save(model_path)
    
    # Generate some sample predictions
    unique_categories = df['category'].unique()
    sample_predictions = model.predict(unique_categories[:5])
    
    print(f"✅ Baseline model trained successfully!")
    print(f"📊 Trained on {len(df)} records with {len(unique_categories)} categories")
    print(f"💾 Model saved to: {model_path}")
    print(f"🎯 Sample predictions for {unique_categories[:5]}: {sample_predictions}")
    
    return model

if __name__ == "__main__":
    model = train_baseline_model()
    if model:
        print("Baseline model training completed successfully!")
