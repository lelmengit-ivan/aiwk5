"""
Main Execution Script for Hospital Readmission Prediction
Orchestrates the entire workflow from data loading to model deployment
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_preprocessing import DataPreprocessor
from feature_engineering import FeatureEngineer
from model_training import ReadmissionPredictor
from evaluation import ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hospital_readmission.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class HospitalReadmissionPipeline:
    """
    End-to-end pipeline for hospital readmission prediction.
    
    This class orchestrates the complete workflow from data loading
    to model training and evaluation.
    """
    
    def __init__(self, data_path, target_column='readmission_risk'):
        """
        Initialize the pipeline with data path and target column.
        
        Args:
            data_path (str): Path to the input data file
            target_column (str): Name of the target variable column
        """
        self.data_path = data_path
        self.target_column = target_column
        self.data = None
        self.processed_data = None
        self.features = None
        self.target = None
        
        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.predictor = ReadmissionPredictor()
        self.evaluator = ModelEvaluator()
        
        logger.info("Hospital Readmission Pipeline initialized")
    
    def run_pipeline(self):
        """
        Execute the complete machine learning pipeline.
        
        Returns:
            dict: Pipeline results including trained model and metrics
        """
        logger.info("Starting Hospital Readmission Prediction Pipeline")
        
        try:
            # Step 1: Data Loading and Exploration
            logger.info("Step 1: Loading and exploring data...")
            self.load_and_explore_data()
            
            # Step 2: Data Preprocessing
            logger.info("Step 2: Preprocessing data...")
            self.preprocess_data()
            
            # Step 3: Feature Engineering
            logger.info("Step 3: Engineering features...")
            self.engineer_features()
            
            # Step 4: Model Training
            logger.info("Step 4: Training models...")
            model_results = self.train_models()
            
            # Step 5: Model Evaluation
            logger.info("Step 5: Evaluating models...")
            evaluation_results = self.evaluate_models(model_results)
            
            # Step 6: Save Best Model
            logger.info("Step 6: Saving best model...")
            self.save_best_model(model_results)
            
            logger.info("Pipeline completed successfully!")
            
            return {
                'model_results': model_results,
                'evaluation_results': evaluation_results,
                'best_model': self.predictor.best_model,
                'best_model_name': self.predictor.best_model_name
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {str(e)}")
            raise
    
    def load_and_explore_data(self):
        """Load data and perform initial exploration."""
        self.data = self.preprocessor.load_data(self.data_path)
        
        # Basic exploration
        logger.info(f"Data shape: {self.data.shape}")
        logger.info(f"Columns: {list(self.data.columns)}")
        logger.info(f"Target distribution:\n{self.data[self.target_column].value_counts()}")
        
        # Check for missing values
        missing_values = self.data.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Found {missing_values} missing values in the dataset")
    
    def preprocess_data(self):
        """Preprocess the data including handling missing values and encoding."""
        # Identify column types (in practice, this would be more sophisticated)
        numerical_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = self.data.select_dtypes(include=['object']).columns.tolist()
        
        # Remove target from feature columns
        if self.target_column in numerical_columns:
            numerical_columns.remove(self.target_column)
        if self.target_column in categorical_columns:
            categorical_columns.remove(self.target_column)
        
        # Handle missing values
        data_clean = self.preprocessor.handle_missing_values(
            self.data, numerical_columns, categorical_columns
        )
        
        # Encode categorical variables
        data_encoded = self.preprocessor.encode_categorical_variables(
            data_clean, categorical_columns, self.target_column
        )
        
        # Scale numerical features
        self.processed_data = self.preprocessor.scale_numerical_features(
            data_encoded, numerical_columns
        )
        
        logger.info(f"Processed data shape: {self.processed_data.shape}")
    
    def engineer_features(self):
        """Perform feature engineering and selection."""
        # Separate features and target
        feature_columns = [col for col in self.processed_data.columns 
                          if col != self.target_column]
        
        self.features = self.processed_data[feature_columns]
        self.target = self.processed_data[self.target_column]
        
        # Create additional features
        features_with_engineering = self.feature_engineer.create_clinical_features(
            self.features
        )
        
        # Select top features (optional - can be tuned)
        if len(feature_columns) > 20:
            self.features = self.feature_engineer.select_features(
                features_with_engineering, self.target, k=20
            )
        else:
            self.features = features_with_engineering
        
        logger.info(f"Final feature set shape: {self.features.shape}")
    
    def train_models(self):
        """Train multiple models and select the best one."""
        # Prepare data splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.predictor.prepare_data(
            self.features, self.target
        )
        
        # Store test set for final evaluation
        self.X_test = X_test
        self.y_test = y_test
        
        # Train models (using a subset for demonstration)
        models_to_train = ['logistic_regression', 'random_forest', 'xgboost']
        results = self.predictor.train_models(
            X_train, y_train, X_val, y_val, models_to_train
        )
        
        # Select best model
        best_name, best_model = self.predictor.select_best_model(results)
        
        return results
    
    def evaluate_models(self, model_results):
        """Comprehensively evaluate all trained models."""
        evaluation_results = {}
        
        for model_name, result in model_results.items():
            model = result['model']
            metrics = self.predictor.evaluate_model(model, self.X_test, self.y_test)
            evaluation_results[model_name] = metrics
        
        return evaluation_results
    
    def save_best_model(self, model_results):
        """Save the best model and feature engineering artifacts."""
        if self.predictor.best_model is not None:
            # Save model
            model_path = f"models/best_{self.predictor.best_model_name}.pkl"
            os.makedirs('models', exist_ok=True)
            self.predictor.save_model(self.predictor.best_model, model_path)
            
            # Save feature importance
            if hasattr(self.feature_engineer, 'get_feature_importance'):
                feature_importance = self.feature_engineer.get_feature_importance()
                importance_df = pd.DataFrame(
                    list(feature_importance.items()),
                    columns=['Feature', 'Importance_Score']
                )
                importance_path = "models/feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                logger.info(f"Feature importance saved to {importance_path}")

def main():
    """
    Main execution function for the hospital readmission prediction pipeline.
    """
    logger.info("Starting Hospital Readmission Prediction System")
    
    # Configuration
    DATA_PATH = "data/sample_data.csv"  # Update with actual data path
    TARGET_COLUMN = "readmission_risk"  # Update with actual target column name
    
    try:
        # Initialize and run pipeline
        pipeline = HospitalReadmissionPipeline(DATA_PATH, TARGET_COLUMN)
        results = pipeline.run_pipeline()
        
        # Print summary results
        logger.info("\n" + "="*50)
        logger.info("PIPELINE RESULTS SUMMARY")
        logger.info("="*50)
        
        if results['best_model_name']:
            logger.info(f"🏆 Best Model: {results['best_model_name']}")
            
            # Print evaluation metrics for best model
            best_model_metrics = results['evaluation_results'][results['best_model_name']]
            for metric, value in best_model_metrics.items():
                if value is not None:
                    logger.info(f"📊 {metric.capitalize()}: {value:.4f}")
        
        logger.info("Pipeline execution completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()