"""
Data Preprocessing Module for Hospital Readmission Prediction
Handles data cleaning, missing value imputation, and basic feature processing
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import logging

class DataPreprocessor:
    """
    A class to preprocess hospital readmission data with comprehensive cleaning
    and transformation capabilities.
    
    Attributes:
        numerical_imputer: Imputer for numerical missing values
        categorical_imputer: Imputer for categorical missing values
        scaler: Standard scaler for numerical features
        label_encoder: Encoder for target variable
    """
    
    def __init__(self):
        """Initialize preprocessor with default strategies"""
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, file_path):
        """
        Load dataset from CSV file with error handling
        
        Args:
            file_path (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded dataset
            
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.EmptyDataError: If file is empty
        """
        try:
            self.logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            self.logger.info(f"Successfully loaded data with shape: {data.shape}")
            return data
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except pd.errors.EmptyDataError:
            self.logger.error("File is empty")
            raise
    
    def handle_missing_values(self, data, numerical_columns, categorical_columns):
        """
        Handle missing values in numerical and categorical columns
        
        Args:
            data (pd.DataFrame): Input data
            numerical_columns (list): List of numerical column names
            categorical_columns (list): List of categorical column names
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        self.logger.info("Handling missing values...")
        
        # Create a copy to avoid modifying original data
        data_clean = data.copy()
        
        # Impute numerical columns with median
        if numerical_columns:
            data_clean[numerical_columns] = self.numerical_imputer.fit_transform(
                data_clean[numerical_columns]
            )
            self.logger.info(f"Imputed numerical columns: {numerical_columns}")
        
        # Impute categorical columns with mode
        if categorical_columns:
            data_clean[categorical_columns] = self.categorical_imputer.fit_transform(
                data_clean[categorical_columns]
            )
            self.logger.info(f"Imputed categorical columns: {categorical_columns}")
        
        # Log missing values summary
        missing_after = data_clean.isnull().sum().sum()
        self.logger.info(f"Missing values after imputation: {missing_after}")
        
        return data_clean
    
    def encode_categorical_variables(self, data, categorical_columns, target_column=None):
        """
        Encode categorical variables using appropriate encoding strategies
        
        Args:
            data (pd.DataFrame): Input data
            categorical_columns (list): List of categorical column names
            target_column (str): Name of target column for special encoding
            
        Returns:
            pd.DataFrame: Data with encoded categorical variables
        """
        self.logger.info("Encoding categorical variables...")
        data_encoded = data.copy()
        
        for col in categorical_columns:
            if col == target_column:
                # Encode target variable
                data_encoded[col] = self.label_encoder.fit_transform(data_encoded[col])
                self.logger.info(f"Label encoded target column: {col}")
            else:
                # One-hot encode other categorical variables
                dummies = pd.get_dummies(data_encoded[col], prefix=col, drop_first=True)
                data_encoded = pd.concat([data_encoded, dummies], axis=1)
                data_encoded.drop(col, axis=1, inplace=True)
                self.logger.info(f"One-hot encoded column: {col}")
        
        return data_encoded
    
    def scale_numerical_features(self, data, numerical_columns):
        """
        Scale numerical features to standard normal distribution
        
        Args:
            data (pd.DataFrame): Input data
            numerical_columns (list): List of numerical column names
            
        Returns:
            pd.DataFrame: Data with scaled numerical features
        """
        self.logger.info("Scaling numerical features...")
        data_scaled = data.copy()
        
        if numerical_columns:
            data_scaled[numerical_columns] = self.scaler.fit_transform(
                data_scaled[numerical_columns]
            )
            self.logger.info(f"Scaled numerical columns: {numerical_columns}")
        
        return data_scaled

# Example usage
if __name__ == "__main__":
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Sample data (in practice, load from actual source)
    sample_data = pd.DataFrame({
        'age': [25, 30, np.nan, 40, 45],
        'blood_pressure': [120, np.nan, 140, 130, 125],
        'gender': ['M', 'F', 'M', np.nan, 'F'],
        'readmission_risk': ['Low', 'High', 'Low', 'High', 'Low']
    })
    
    # Preprocess data
    cleaned_data = preprocessor.handle_missing_values(
        sample_data, 
        numerical_columns=['age', 'blood_pressure'],
        categorical_columns=['gender']
    )
    
    encoded_data = preprocessor.encode_categorical_variables(
        cleaned_data,
        categorical_columns=['gender', 'readmission_risk'],
        target_column='readmission_risk'
    )
    
    print("Preprocessed Data:")
    print(encoded_data.head())