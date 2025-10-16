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

class DataIngestion:
    """
    Handles ingestion of financial data.
    CUSTOMIZED to handle pre-aggregated data with the format:
    'category', 'current_spend', 'percentage_of_total', 'recommendation', 'target_budget'
    """
    def __init__(self, data_path="data/raw/data.csv"):
        self.data_path = Path(data_path)
        # Don't create directories or setup logging during __init__
        # This will be done in the method call instead

    def initiate_data_ingestion(self):
        """Initiate the data ingestion process"""
        # Setup logging here instead of in __init__
        log_dir = Path("Checkpoints/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_dir / "ingestion.log")
        
        logging.info("Starting Data Ingestion process...")

        if not self.data_path.exists():
            logging.warning(f"'{self.data_path}' not found. The script requires your data file at this location.")
            print(f"❌ ERROR: Data file not found at '{self.data_path}'. Please place your data there.")
            return None
        
        logging.info(f"Reading data from '{self.data_path}'")
        df = pd.read_csv(self.data_path)
        logging.info("Original data read successfully. Now transforming to pipeline-compatible format.")

        # --- DATA TRANSFORMATION LOGIC ---
        # 1. Rename 'current_spend' to 'amount' to match the pipeline's expectation.
        df.rename(columns={'current_spend': 'amount'}, inplace=True)
        
        # 2. STANDARDIZE CATEGORIES - Merge similar categories
        logging.info("Standardizing categories...")
        original_categories = df['category'].nunique()
        df = standardizer.standardize_dataframe(df, 'category')
        standardized_categories = df['category'].nunique()
        logging.info(f"Categories standardized: {original_categories} -> {standardized_categories}")
        
        # 3. Add a 'date' column. Since none exists, we'll assume these totals
        #    are for the current month. This is CRUCIAL for the downstream
        #    time-series processing and forecasting steps.
        #    We use the first day of the current month.
        df['date'] = pd.to_datetime('today').strftime('%Y-%m-01')
        
        # 4. Add a 'transaction_type' column, assuming all spends are 'Expense'.
        df['transaction_type'] = 'Expense'
        
        # 5. Select and reorder columns to match the required format for the pipeline.
        #    This also drops the unused columns ('percentage_of_total', 'recommendation', 'target_budget').
        required_columns = ['date', 'amount', 'category', 'transaction_type']
        df = df[required_columns]
        
        logging.info("Data successfully transformed into the required format: ['date', 'amount', 'category', 'transaction_type']")
        logging.info(f"Standardized categories: {sorted(df['category'].unique())}")
        logging.info(f"Final data shape after transformation: {df.shape}")
        
        # Define the output path for the ingested data
        output_path = Path("Checkpoints/logs/data_ingested.csv")
        output_path.parent.mkdir(parents=True, exist_ok=True) # Ensure directory exists
        
        df.to_csv(output_path, index=False)
        logging.info(f"Data ingestion completed. Transformed data saved to '{output_path}'")
        return df

if __name__ == '__main__':
    # This allows you to test the script directly
    ingestor = DataIngestion()
    ingested_data = ingestor.initiate_data_ingestion()
    if ingested_data is not None:
        print("Data Ingestion ran successfully.")
        print("Transformed data preview:")
        print(ingested_data.head())