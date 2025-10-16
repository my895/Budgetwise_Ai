#!/usr/bin/env python3
"""
Model Evaluation for Budget Tool MVP
Evaluates and compares all trained models
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

def setup_logging():
    """Setup logging for model evaluation"""
    log_dir = Path("Checkpoints/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_dir / "model_evaluation.log",
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

class ModelEvaluator:
    """
    Evaluates and compares different models
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def load_models(self):
        """Load all trained models"""
        model_dir = Path("Checkpoints/models")
        
        # Load baseline model
        baseline_path = model_dir / "baseline_model.pkl"
        if baseline_path.exists():
            try:
                from src.models.baseline import BaselineModel
                baseline_model = BaselineModel()
                baseline_model.load(baseline_path)
                self.models['baseline'] = baseline_model
                logging.info("Baseline model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load baseline model: {e}")
        
        # Load Random Forest model
        rf_path = model_dir / "random_forest_model.pkl"
        if rf_path.exists():
            try:
                from src.models.ml_model import MLModel
                rf_model = MLModel('random_forest')
                rf_model.load(rf_path)
                self.models['random_forest'] = rf_model
                logging.info("Random Forest model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load Random Forest model: {e}")
        
        # Load Linear Regression model
        lr_path = model_dir / "linear_regression_model.pkl"
        if lr_path.exists():
            try:
                from src.models.ml_model import MLModel
                lr_model = MLModel('linear_regression')
                lr_model.load(lr_path)
                self.models['linear_regression'] = lr_model
                logging.info("Linear Regression model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load Linear Regression model: {e}")
        
        # Load Deep Learning model
        dl_path = model_dir / "deep_learning_model.h5"
        if dl_path.exists():
            try:
                from src.models.dl_model import DLModel
                dl_model = DLModel()
                dl_model.load(dl_path)
                self.models['deep_learning'] = dl_model
                logging.info("Deep Learning model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load Deep Learning model: {e}")
    
    def evaluate_model(self, model_name, model, test_data):
        """Evaluate a single model"""
        try:
            # Make predictions
            if model_name == 'baseline':
                # For baseline model, we need categories
                categories = test_data['category'].tolist()
                predictions = model.predict(categories)
            else:
                # For other models, pass the dataframe
                predictions = model.predict(test_data)
            
            # Get actual values
            actual = test_data['amount'].values
            
            # Calculate metrics
            mae = mean_absolute_error(actual, predictions)
            mse = mean_squared_error(actual, predictions)
            rmse = np.sqrt(mse)
            r2 = r2_score(actual, predictions)
            
            # Calculate percentage error
            mape = np.mean(np.abs((actual - predictions) / actual)) * 100
            
            results = {
                'model': model_name,
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'r2': float(r2),
                'mape': float(mape),
                'n_samples': len(actual)
            }
            
            logging.info(f"Evaluated {model_name}: MAE={mae:.2f}, R²={r2:.4f}")
            return results
            
        except Exception as e:
            logging.error(f"Failed to evaluate {model_name}: {e}")
            return None
    
    def evaluate_all_models(self):
        """Evaluate all loaded models"""
        logging.info("Starting model evaluation...")
        
        # Load test data
        data_path = Path("Checkpoints/logs/data_preprocessed.csv")
        if not data_path.exists():
            logging.error("Preprocessed data not found")
            print("❌ ERROR: Preprocessed data not found. Run data preprocessing first.")
            return None
        
        # Load data
        df = pd.read_csv(data_path)
        
        # For evaluation, we'll use the entire dataset as test data
        # In a real scenario, you'd have separate train/test splits
        test_data = df.copy()
        
        print(f"📊 Evaluating models on {len(test_data)} samples...")
        
        # Evaluate each model
        for model_name, model in self.models.items():
            print(f"🔍 Evaluating {model_name} model...")
            results = self.evaluate_model(model_name, model, test_data)
            if results:
                self.results[model_name] = results
        
        return self.results
    
    def generate_report(self):
        """Generate evaluation report"""
        if not self.results:
            print("❌ No evaluation results available")
            return
        
        print("\n" + "="*80)
        print("📈 MODEL EVALUATION REPORT")
        print("="*80)
        
        # Create results DataFrame for easy comparison
        results_df = pd.DataFrame(list(self.results.values()))
        results_df = results_df.set_index('model')
        
        # Sort by R² score (higher is better)
        results_df = results_df.sort_values('r2', ascending=False)
        
        print("\n🏆 MODEL PERFORMANCE COMPARISON:")
        print("-" * 80)
        print(f"{'Model':<20} {'R²':<8} {'MAE':<10} {'RMSE':<10} {'MAPE':<8}")
        print("-" * 80)
        
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<20} {row['r2']:<8.4f} {row['mae']:<10.2f} {row['rmse']:<10.2f} {row['mape']:<8.2f}%")
        
        # Best model
        best_model = results_df.index[0]
        best_r2 = results_df.iloc[0]['r2']
        
        print("-" * 80)
        print(f"🥇 Best performing model: {best_model} (R² = {best_r2:.4f})")
        
        # Save results
        self.save_results()
        
        return results_df
    
    def save_results(self):
        """Save evaluation results to file"""
        if not self.results:
            return
        
        # Save as JSON
        results_dir = Path("Checkpoints/evaluation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = results_dir / "evaluation_results.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save as CSV
        csv_path = results_dir / "evaluation_results.csv"
        results_df = pd.DataFrame(list(self.results.values()))
        results_df.to_csv(csv_path, index=False)
        
        logging.info(f"Evaluation results saved to {json_path} and {csv_path}")
        print(f"💾 Results saved to: {json_path}")

def run_evaluation():
    """Run complete model evaluation"""
    setup_logging()
    logging.info("Starting complete model evaluation...")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load models
    print("📥 Loading trained models...")
    evaluator.load_models()
    
    if not evaluator.models:
        print("❌ No trained models found. Train models first.")
        return None
    
    print(f"✅ Loaded {len(evaluator.models)} models: {list(evaluator.models.keys())}")
    
    # Evaluate models
    results = evaluator.evaluate_all_models()
    
    if not results:
        print("❌ Model evaluation failed")
        return None
    
    # Generate report
    report_df = evaluator.generate_report()
    
    print("\n✅ Model evaluation completed successfully!")
    return report_df

if __name__ == "__main__":
    results = run_evaluation()
    if results is not None:
        print("Model evaluation completed successfully!")
