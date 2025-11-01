"""
Feature Engineering Module for Hospital Readmission Prediction
Creates meaningful features from raw healthcare data
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
import logging

class FeatureEngineer:
    """
    A class to create and select meaningful features for readmission prediction.
    
    This class handles feature creation, transformation, and selection
    to improve model performance and interpretability.
    """
    
    def __init__(self):
        self.feature_selector = None
        self.selected_features = []
        self.logger = logging.getLogger(__name__)
    
    def create_clinical_features(self, data):
        """
        Create clinically relevant features from raw medical data
        
        Args:
            data (pd.DataFrame): Raw medical data
            
        Returns:
            pd.DataFrame: Data with additional clinical features
        """
        self.logger.info("Creating clinical features...")
        data_with_features = data.copy()
        
        # Feature: Comorbidity score (sum of chronic conditions)
        chronic_conditions = ['diabetes', 'hypertension', 'heart_disease', 'kidney_disease']
        available_conditions = [col for col in chronic_conditions if col in data.columns]
        
        if available_conditions:
            data_with_features['comorbidity_score'] = data[available_conditions].sum(axis=1)
            self.logger.info(f"Created comorbidity_score from {available_conditions}")
        
        # Feature: Age categories for different risk profiles
        if 'age' in data.columns:
            data_with_features['age_group'] = pd.cut(
                data['age'],
                bins=[0, 30, 50, 65, 100],
                labels=['Young', 'Adult', 'Senior', 'Elderly']
            )
            self.logger.info("Created age_group feature")
        
        # Feature: Length of stay impact (if admission data available)
        if 'length_of_stay' in data.columns:
            data_with_features['prolonged_stay'] = (data['length_of_stay'] > 7).astype(int)
            data_with_features['log_length_of_stay'] = np.log1p(data['length_of_stay'])
            self.logger.info("Created length of stay features")
        
        # Feature: Medication complexity
        if 'medication_count' in data.columns:
            data_with_features['high_medication_burden'] = (
                data['medication_count'] > 5
            ).astype(int)
            self.logger.info("Created medication burden feature")
        
        return data_with_features
    
    def create_temporal_features(self, data):
        """
        Create time-based features from admission/discharge dates
        
        Args:
            data (pd.DataFrame): Input data with date columns
            
        Returns:
            pd.DataFrame: Data with temporal features
        """
        self.logger.info("Creating temporal features...")
        data_with_temporal = data.copy()
        
        # Convert date columns if they exist
        date_columns = ['admission_date', 'discharge_date', 'previous_discharge_date']
        
        for date_col in date_columns:
            if date_col in data.columns:
                data_with_temporal[date_col] = pd.to_datetime(
                    data_with_temporal[date_col], errors='coerce'
                )
        
        # Feature: Time since last admission
        if 'previous_discharge_date' in data.columns and 'admission_date' in data.columns:
            data_with_temporal['days_since_last_admission'] = (
                data_with_temporal['admission_date'] - 
                data_with_temporal['previous_discharge_date']
            ).dt.days
            self.logger.info("Created days_since_last_admission feature")
        
        # Feature: Seasonal admission
        if 'admission_date' in data.columns:
            data_with_temporal['admission_month'] = data_with_temporal['admission_date'].dt.month
            data_with_temporal['admission_season'] = (
                data_with_temporal['admission_month'] % 12 + 3) // 3
            self.logger.info("Created seasonal admission features")
        
        return data_with_temporal
    
    def select_features(self, X, y, k=20):
        """
        Select top k most important features using statistical tests
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of top features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        self.logger.info(f"Selecting top {k} features...")
        
        # Initialize feature selector
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        
        # Fit and transform features
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        self.selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        self.logger.info(f"Selected {len(self.selected_features)} features")
        self.logger.info(f"Selected features: {self.selected_features}")
        
        return pd.DataFrame(X_selected, columns=self.selected_features, index=X.index)
    
    def get_feature_importance(self):
        """
        Get feature importance scores from the selector
        
        Returns:
            dict: Feature names and their importance scores
        """
        if self.feature_selector is None:
            self.logger.warning("Feature selector not fitted yet")
            return {}
        
        feature_scores = dict(zip(
            self.selected_features, 
            self.feature_selector.scores_[self.feature_selector.get_support()]
        ))
        
        return dict(sorted(feature_scores.items(), key=lambda x: x[1], reverse=True))

# Example usage
if __name__ == "__main__":
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Sample data
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45],
        'length_of_stay': [3, 7, 10, 5, 8],
        'medication_count': [2, 5, 8, 3, 6],
        'diabetes': [0, 1, 0, 1, 1],
        'hypertension': [1, 0, 1, 1, 0],
        'readmission_risk': [0, 1, 1, 0, 1]
    })
    
    # Create features
    enhanced_data = feature_engineer.create_clinical_features(sample_data)
    print("Enhanced Data with Clinical Features:")
    print(enhanced_data.head())