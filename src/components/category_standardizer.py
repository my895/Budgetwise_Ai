"""
Category Standardizer for Budget Tool MVP
Standardizes various category names into consistent categories
"""

import re
import pandas as pd
from typing import Dict, List

class CategoryStandardizer:
    """
    Standardizes category names by mapping variations to standard categories
    """
    
    def __init__(self):
        # Define standard categories and their variations
        self.category_mapping = {
            # Housing/Rent categories
            'RENT': [
                'rent', 'Rent', 'RENT', 'Rnt', 'Rentt', 'rentt', 'RENTT'
            ],
            
            # Food categories
            'FOOD': [
                'food', 'Food', 'FOOD', 'Fod', 'Foods', 'Foodd', 'foods', 'foodd'
            ],
            
            # Travel categories
            'TRAVEL': [
                'travel', 'Travel', 'TRAVEL', 'Traval', 'Travl', 'traval', 'travl'
            ],
            
            # Entertainment categories
            'ENTERTAINMENT': [
                'entertainment', 'Entertainment', 'ENTERTAINMENT',
                'Entertain', 'entertain', 'Entrtnmnt', 'entrtnmnt'
            ],
            
            # Utilities categories
            'UTILITIES': [
                'utilities', 'Utilities', 'UTILITIES', 'Utility', 'utility',
                'Utlities', 'Utilties', 'utlities', 'utilties'
            ],
            
            # Education categories
            'EDUCATION': [
                'education', 'Education', 'EDUCATION', 'EDU', 'edu',
                'Educaton', 'educaton'
            ],
            
            # Health categories
            'HEALTH': [
                'health', 'Health', 'HEALTH', 'Helth', 'helth'
            ],
            
            # Savings categories
            'SAVINGS': [
                'savings', 'Savings', 'SAVINGS', 'Saving', 'saving'
            ],
            
            # Others/Misc categories
            'OTHERS': [
                'others', 'Others', 'OTHERS', 'Other', 'other', 'Misc', 'misc'
            ],
            
            # Unknown categories
            'UNKNOWN': [
                'Unknown', 'unknown', 'UNKNOWN'
            ]
        }
        
        # Create reverse mapping for quick lookup
        self.reverse_mapping = {}
        for standard_category, variations in self.category_mapping.items():
            for variation in variations:
                self.reverse_mapping[variation.lower()] = standard_category
    
    def standardize_category(self, category: str) -> str:
        """
        Standardize a single category name
        
        Args:
            category: The category name to standardize
            
        Returns:
            The standardized category name
        """
        if not category or pd.isna(category):
            return 'UNKNOWN'
        
        # Clean the category name
        category_clean = str(category).strip()
        
        # Try exact match first (case insensitive)
        category_lower = category_clean.lower()
        if category_lower in self.reverse_mapping:
            return self.reverse_mapping[category_lower]
        
        # Try fuzzy matching for common variations
        category_clean_lower = re.sub(r'[^a-zA-Z]', '', category_clean.lower())
        
        # Check for common patterns
        if category_clean_lower in ['rent', 'rentt', 'rnt']:
            return 'RENT'
        elif category_clean_lower in ['food', 'fod', 'foods', 'foodd']:
            return 'FOOD'
        elif category_clean_lower in ['travel', 'traval', 'travl']:
            return 'TRAVEL'
        elif category_clean_lower in ['entertainment', 'entertain', 'entrtnmnt']:
            return 'ENTERTAINMENT'
        elif category_clean_lower in ['utilities', 'utility', 'utlities', 'utilties']:
            return 'UTILITIES'
        elif category_clean_lower in ['education', 'edu', 'educaton']:
            return 'EDUCATION'
        elif category_clean_lower in ['health', 'helth']:
            return 'HEALTH'
        elif category_clean_lower in ['savings', 'saving']:
            return 'SAVINGS'
        elif category_clean_lower in ['others', 'other', 'misc']:
            return 'OTHERS'
        else:
            return 'UNKNOWN'
    
    def standardize_dataframe(self, df, category_column: str = 'category'):
        """
        Standardize categories in a DataFrame
        
        Args:
            df: The DataFrame containing categories
            category_column: The name of the category column
            
        Returns:
            DataFrame with standardized categories
        """
        df_standardized = df.copy()
        df_standardized[category_column] = df_standardized[category_column].apply(
            self.standardize_category
        )
        return df_standardized
    
    def get_standard_categories(self) -> List[str]:
        """Get list of all standard categories"""
        return list(self.category_mapping.keys())
    
    def get_category_summary(self, df, category_column: str = 'category'):
        """
        Get summary of category standardization
        
        Args:
            df: Original DataFrame
            category_column: The name of the category column
            
        Returns:
            Dictionary with standardization summary
        """
        original_categories = df[category_column].unique()
        standardized_df = self.standardize_dataframe(df, category_column)
        standardized_categories = standardized_df[category_column].unique()
        
        # Create mapping summary
        mapping_summary = {}
        for orig_cat in original_categories:
            std_cat = self.standardize_category(orig_cat)
            if std_cat not in mapping_summary:
                mapping_summary[std_cat] = []
            mapping_summary[std_cat].append(orig_cat)
        
        return {
            'original_count': len(original_categories),
            'standardized_count': len(standardized_categories),
            'reduction': len(original_categories) - len(standardized_categories),
            'mapping_summary': mapping_summary,
            'original_categories': list(original_categories),
            'standardized_categories': list(standardized_categories)
        }

# Create a global instance for easy import
standardizer = CategoryStandardizer()
