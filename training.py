"""
Model Training Module for Hospital Readmission Prediction
Handles model training, validation, and hyperparameter tuning
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import joblib
import logging

class ReadmissionPredictor:
    """
    A comprehensive class for training and evaluating readmission prediction models.
    
    This class supports multiple algorithms and includes hyperparameter tuning
    and cross-validation capabilities.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the predictor with multiple model options
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.logger = logging.getLogger(__name__)
        
        # Initialize models with default parameters
        self.model_configs = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state),
                'params': {
                    'C': [0.1, 1, 10],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            },
            'xgboost': {
                'model': xgb.XGBClassifier(random_state=random_state, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }
    
    def prepare_data(self, X, y, test_size=0.2, val_size=0.2):
        """
        Split data into training, validation, and test sets
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of test set
            val_size (float): Proportion of validation set from training data
            
        Returns:
            tuple: X_train, X_val, X_test, y_train, y_val, y_test
        """
        self.logger.info("Splitting data into train/validation/test sets...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        # Second split: separate validation set from temp
        val_size_adj = val_size / (1 - test_size)  # Adjust validation size
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adj, random_state=self.random_state, stratify=y_temp
        )
        
        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Validation set: {X_val.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val, y_val, models_to_train='all'):
        """
        Train multiple models and select the best performing one
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training target
            X_val (pd.DataFrame): Validation features
            y_val (pd.Series): Validation target
            models_to_train (list or str): Which models to train
            
        Returns:
            dict: Training results for all models
        """
        self.logger.info("Training multiple models...")
        
        if models_to_train == 'all':
            models_to_train = list(self.model_configs.keys())
        
        results = {}
        
        for model_name in models_to_train:
            if model_name not in self.model_configs:
                self.logger.warning(f"Model {model_name} not found in configurations")
                continue
            
            self.logger.info(f"Training {model_name}...")
            
            try:
                # Get model configuration
                config = self.model_configs[model_name]
                model = config['model']
                params = config['params']
                
                # Perform grid search with cross-validation
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params,
                    cv=5,
                    scoring='f1',
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit the model
                grid_search.fit(X_train, y_train)
                
                # Store the best model
                self.models[model_name] = grid_search.best_estimator_
                
                # Evaluate on validation set
                y_val_pred = grid_search.best_estimator_.predict(X_val)
                val_accuracy = accuracy_score(y_val, y_val_pred)
                val_precision = precision_score(y_val, y_val_pred)
                val_recall = recall_score(y_val, y_val_pred)
                val_f1 = f1_score(y_val, y_val_pred)
                
                # Store results
                results[model_name] = {
                    'model': grid_search.best_estimator_,
                    'best_params': grid_search.best_params_,
                    'val_accuracy': val_accuracy,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1,
                    'cv_score': grid_search.best_score_
                }
                
                self.logger.info(
                    f"{model_name} - Val F1: {val_f1:.4f}, "
                    f"Val Recall: {val_recall:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        return results
    
    def select_best_model(self, results, metric='val_f1'):
        """
        Select the best model based on specified metric
        
        Args:
            results (dict): Training results from train_models
            metric (str): Metric to use for model selection
            
        Returns:
            tuple: Best model name and the model object
        """
        self.logger.info(f"Selecting best model based on {metric}...")
        
        if not results:
            self.logger.error("No results available for model selection")
            return None, None
        
        # Find model with best performance on specified metric
        best_model_name = max(results.keys(), key=lambda x: results[x][metric])
        self.best_model = results[best_model_name]['model']
        self.best_model_name = best_model_name
        
        best_score = results[best_model_name][metric]
        self.logger.info(
            f"Best model: {best_model_name} with {metric}: {best_score:.4f}"
        )
        
        return best_model_name, self.best_model
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Comprehensive evaluation of the model on test set
        
        Args:
            model: Trained model object
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test target
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
        self.logger.info("Evaluating model on test set...")
        
        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': None,  # Would need to import roc_auc_score
        }
        
        # Log metrics
        for metric_name, value in metrics.items():
            if value is not None:
                self.logger.info(f"Test {metric_name}: {value:.4f}")
        
        return metrics
    
    def save_model(self, model, file_path):
        """
        Save trained model to file
        
        Args:
            model: Trained model object
            file_path (str): Path to save the model
        """
        try:
            joblib.dump(model, file_path)
            self.logger.info(f"Model saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
    
    def load_model(self, file_path):
        """
        Load trained model from file
        
        Args:
            file_path (str): Path to the saved model
            
        Returns:
            Loaded model object
        """
        try:
            model = joblib.load(file_path)
            self.logger.info(f"Model loaded from {file_path}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = ReadmissionPredictor()
    
    # Sample data
    X = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'feature3': np.random.randn(100)
    })
    y = np.random.randint(0, 2, 100)
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = predictor.prepare_data(X, y)
    
    # Train models
    results = predictor.train_models(X_train, y_train, X_val, y_val, models_to_train=['logistic_regression', 'random_forest'])
    
    # Select best model
    best_name, best_model = predictor.select_best_model(results)
    
    # Evaluate best model
    if best_model is not None:
        test_metrics = predictor.evaluate_model(best_model, X_test, y_test)
        print(f"Best Model Test F1: {test_metrics['f1']:.4f}")