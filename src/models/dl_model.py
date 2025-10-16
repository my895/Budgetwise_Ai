#!/usr/bin/env python3
"""
Deep Learning Model for Budget Tool MVP
Uses TensorFlow/Keras for neural network-based spending prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
try:
    from src.components.category_standardizer import standardizer
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
    from src.components.category_standardizer import standardizer


def setup_logging():
    """Setup logging for DL model"""
    log_dir = Path("Checkpoints/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / "dl_model.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

class DLModel:
    """
    Deep Learning model for budget prediction using Neural Networks
    """
    
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []
        self.input_dim = 0
    
    def prepare_features(self, df):
        """Prepare features for DL model"""
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
        
        # Normalize date features
        features_df['month_norm'] = features_df['month'] / 12.0
        features_df['day_norm'] = features_df['day'] / 31.0
        features_df['day_of_week_norm'] = features_df['day_of_week'] / 7.0
        
        # Select feature columns
        self.feature_columns = [
            'category_encoded', 'transaction_type_encoded', 
            'month_norm', 'day_norm', 'day_of_week_norm'
        ]
        X = features_df[self.feature_columns]
        
        # Store input dimension
        self.input_dim = X.shape[1]
        
        return X, features_df['amount']
    
    def create_model(self):
        """Create the neural network model"""
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=(self.input_dim,)),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='linear')  # Linear activation for regression
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, df):
        """Train the DL model"""
        logging.info("Training Deep Learning model...")
        
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
        
        # Create model
        self.model = self.create_model()
        
        # Train model
        history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=50,
            batch_size=8,
            verbose=0
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_val_scaled, verbose=0).flatten()
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
        features_df['month_norm'] = features_df['date'].dt.month / 12.0
        features_df['day_norm'] = features_df['date'].dt.day / 31.0
        features_df['day_of_week_norm'] = features_df['date'].dt.dayofweek / 7.0
        
        # Select features and scale
        X = features_df[self.feature_columns]
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled, verbose=0).flatten()
        return predictions
    
    def save(self, filepath):
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save Keras model
        model_path = filepath.with_suffix('')
        self.model.save(model_path)
        
        # Save preprocessing objects
        preprocessing_data = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'input_dim': self.input_dim,
            'is_trained': self.is_trained
        }
        
        preprocessing_path = filepath.with_suffix('.preprocessing.pkl')
        joblib.dump(preprocessing_data, preprocessing_path)
        
        logging.info(f"DL model saved to {model_path}")
        logging.info(f"Preprocessing data saved to {preprocessing_path}")
    
    def load(self, filepath):
        """Load a trained model"""
        # Load Keras model
        model_path = filepath.with_suffix('')
        self.model = keras.models.load_model(model_path)
        
        # Load preprocessing objects
        preprocessing_path = filepath.with_suffix('.preprocessing.pkl')
        preprocessing_data = joblib.load(preprocessing_path)
        
        self.label_encoder = preprocessing_data['label_encoder']
        self.scaler = preprocessing_data['scaler']
        self.feature_columns = preprocessing_data['feature_columns']
        self.input_dim = preprocessing_data['input_dim']
        self.is_trained = preprocessing_data['is_trained']
        
        logging.info(f"DL model loaded from {model_path}")
        return self

def train_dl_model():
    """Train the deep learning model"""
    setup_logging()
    logging.info("Starting Deep Learning model training...")
    
    # Load preprocessed data
    data_path = Path("Checkpoints/logs/data_preprocessed.csv")
    if not data_path.exists():
        logging.error(f"Preprocessed data not found at {data_path}")
        print(f"❌ ERROR: Preprocessed data not found. Run data preprocessing first.")
        return None
    
    # Load data
    df = pd.read_csv(data_path)
    logging.info(f"Loaded data: {df.shape}")
    
    # Check if we have enough data for deep learning
    if len(df) < 10:
        logging.warning("Limited data available for deep learning training")
        print("⚠️  Warning: Limited data available. Deep learning may not perform optimally.")
    
    # Train DL model
    print("🧠 Training Deep Learning model...")
    dl_model = DLModel()
    dl_model.fit(df)
    
    # Save model
    model_dir = Path("Checkpoints/models")
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "deep_learning_model.h5"
    dl_model.save(model_path)
    
    print(f"✅ Deep Learning model trained successfully!")
    print(f"📊 Trained on {len(df)} records")
    print(f"💾 Model saved to: {model_path}")
    
    return dl_model

if __name__ == "__main__":
    model = train_dl_model()
    if model:
        print("Deep Learning model training completed successfully!")
