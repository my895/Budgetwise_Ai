#!/usr/bin/env python3
"""
Machine Learning Model for Budget Tool MVP
Uses Random Forest and Linear Regression for spending prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
try:
    from src.components.category_standardizer import standardizer
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.components.category_standardizer import standardizer

def setup_logging():
    """Setup logging for ML model"""
    log_dir = Path("Checkpoints/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / "ml_model.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

class MLModel:
    """
    Machine Learning model for budget prediction
    """
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
    
    def prepare_features(self, df):
        """Prepare features for ML model"""
        # Create features
        features_df = df.copy()
        
        # Encode categorical variables
        features_df['category_encoded'] = self.label_encoder.fit_transform(features_df['category'])
        features_df['transaction_type_encoded'] = self.label_encoder.fit_transform(features_df['transaction_type'])
        
        # Extract date features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day'] = features_df['date'].dt.day
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        
        # Select feature columns
        self.feature_columns = ['category_encoded', 'transaction_type_encoded', 'month', 'day', 'day_of_week']
        X = features_df[self.feature_columns]
        
        return X, features_df['amount']
    
    def fit(self, df):
        """Train the ML model"""
        logging.info(f"Training {self.model_type} model...")
        
        # Ensure categories are standardized
        df_standardized = standardizer.standardize_dataframe(df, 'category')
        logging.info(f"Categories standardized for training: {sorted(df_standardized['category'].unique())}")
        
        # Prepare features
        X, y = self.prepare_features(df_standardized)
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Initialize and train model
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif self.model_type == 'linear_regression':
            self.model = LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_val_scaled)
        mae = mean_absolute_error(y_val, y_pred)
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        logging.info(f"Model performance - MAE: {mae:.2f}, MSE: {mse:.2f}, R²: {r2:.4f}")
        
        self.is_trained = True
        return self
    
    def predict(self, df):
        """Make predictions on new data"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Ensure categories are standardized
        df_standardized = standardizer.standardize_dataframe(df, 'category')
        
        # Prepare features
        features_df = df_standardized.copy()
        
        # Encode categorical variables (using fitted encoders)
        features_df['category_encoded'] = self.label_encoder.transform(features_df['category'])
        features_df['transaction_type_encoded'] = self.label_encoder.transform(features_df['transaction_type'])
        
        # Extract date features
        features_df['date'] = pd.to_datetime(features_df['date'])
        features_df['month'] = features_df['date'].dt.month
        features_df['day'] = features_df['date'].dt.day
        features_df['day_of_week'] = features_df['date'].dt.dayofweek
        
        # Select features and scale
        X = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        return predictions
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, filepath)
        logging.info(f"ML model saved to {filepath}")
    
    def load(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logging.info(f"ML model loaded from {filepath}")
        return self

def train_ml_models():
    """Train both Random Forest and Linear Regression models"""
    setup_logging()
    logging.info("Starting ML model training...")
    
    # Load preprocessed data
    data_path = Path("Checkpoints/logs/data_preprocessed.csv")
    if not data_path.exists():
        logging.error(f"Preprocessed data not found at {data_path}")
        print(f"❌ ERROR: Preprocessed data not found. Run data preprocessing first.")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    logging.info(f"Loaded data: {df.shape}")
    
    # Train Random Forest model
    print("🌲 Training Random Forest model...")
    rf_model = MLModel('random_forest')
    rf_model.fit(df)
    
    # Train Linear Regression model
    print("📈 Training Linear Regression model...")
    lr_model = MLModel('linear_regression')
    lr_model.fit(df)
    
    # Save models
    model_dir = Path("Checkpoints/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    rf_path = model_dir / "random_forest_model.pkl"
    lr_path = model_dir / "linear_regression_model.pkl"
    
    rf_model.save(rf_path)
    lr_model.save(lr_path)
    
    print(f"✅ ML models trained successfully!")
    print(f"📊 Trained on {len(df)} records")
    print(f"💾 Random Forest model saved to: {rf_path}")
    print(f"💾 Linear Regression model saved to: {lr_path}")
    
    return rf_model, lr_model

if __name__ == "__main__":
    models = train_ml_models()
    if models:
        print("ML model training completed successfully!")
