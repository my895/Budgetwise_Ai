import pandas as pd
import logging
from pathlib import Path
from .category_standardizer import standardizer

def setup_logging(log_file):
    """Simple logger setup"""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(module)s | %(message)s"
    )

class DataPreprocessing:
    """
    Handles data preprocessing for the financial data pipeline.
    Processes the ingested data and prepares it for model training.
    """
    
    def __init__(self, input_path="Checkpoints/logs/data_ingested.csv"):
        self.input_path = Path(input_path)
        # Don't setup logging during __init__ to avoid import issues

    def initiate_preprocessing(self):
        """Initiate the data preprocessing process"""
        # Setup logging here instead of in __init__
        log_dir = Path("artifacts/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir / "preprocessing.log")
        
        logging.info("Starting Data Preprocessing process...")

        if not self.input_path.exists():
            logging.error(f"Input file '{self.input_path}' not found.")
            print(f"❌ ERROR: Input file not found at '{self.input_path}'. Run data ingestion first.")
            return None

        logging.info(f"Reading preprocessed data from '{self.input_path}'")
        df = pd.read_csv(self.input_path)
        logging.info(f"Original data shape: {df.shape}")
        
        # Basic preprocessing steps
        logging.info("Starting data preprocessing...")
        
        # 1. Convert date column to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # 2. STANDARDIZE CATEGORIES AGAIN (in case ingestion didn't do it)
        logging.info("Ensuring categories are standardized...")
        original_categories = df['category'].nunique()
        df = standardizer.standardize_dataframe(df, 'category')
        standardized_categories = df['category'].nunique()
        logging.info(f"Categories verified/standardized: {original_categories} -> {standardized_categories}")
        
        # 3. Handle missing values (if any)
        if df.isnull().sum().sum() > 0:
            logging.info("Found missing values, handling them...")
            df = df.dropna()
        
        # 4. Ensure amount is numeric
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
        
        # 5. Remove any rows with invalid amounts
        df = df[df['amount'] > 0]
        
        # 6. Sort by date and category
        df = df.sort_values(['date', 'category'])
        
        logging.info(f"Processed data shape: {df.shape}")
        logging.info(f"Final standardized categories: {sorted(df['category'].unique())}")
        logging.info("Data preprocessing completed successfully.")
        
        # Save preprocessed data
        output_path = Path("Checkpoints/logs/data_preprocessed.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        
        logging.info(f"Preprocessed data saved to '{output_path}'")
        return df

if __name__ == '__main__':
    # This allows you to test the script directly
    preprocessor = DataPreprocessing()
    preprocessed_data = preprocessor.initiate_preprocessing()
    if preprocessed_data is not None:
        print("Data Preprocessing ran successfully.")
        print("Preprocessed data preview:")
        print(preprocessed_data.head())