import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor
import joblib
import json
import os
from datetime import datetime
from typing import Dict, Any, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

class MLHyperparameterOptimizer:
    def __init__(self, task_type: str = 'classification'):
        """
        Initialize ML Hyperparameter Optimizer
        
        Args:
            task_type: 'classification' or 'regression'
        """
        self.task_type = task_type
        self.best_params = {}
        self.best_scores = {}
        self.models = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42):
        """
        Prepare data for training and validation
        """
        from sklearn.model_selection import train_test_split
        
        # Handle categorical variables
        X_processed = X.copy()
        categorical_columns = X_processed.select_dtypes(include=['object']).columns
        
        label_encoders = {}
        for col in categorical_columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le
        
        # Handle target variable for classification
        y_processed = y.copy()
        target_encoder = None
        if self.task_type == 'classification' and y.dtype == 'object':
            target_encoder = LabelEncoder()
            y_processed = target_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_processed, test_size=test_size, 
            random_state=random_state, stratify=y_processed if self.task_type == 'classification' else None
        )
        
        self.data = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder
        }
        
        return X_train, X_test, y_train, y_test
    
    def xgboost_objective(self, trial, X_train, y_train):
        """
        Objective function for XGBoost hyperparameter optimization
        """
        params = {
            'n_estimators': trial.suggest_int('xgb_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
            'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('xgb_reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42
        }
        
        if self.task_type == 'classification':
            model = XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'f1_weighted'
        else:
            model = XGBRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores.mean()
    
    def random_forest_objective(self, trial, X_train, y_train):
        """
        Objective function for Random Forest hyperparameter optimization
        """
        params = {
            'n_estimators': trial.suggest_int('rf_n_estimators', 50, 500),
            'max_depth': trial.suggest_int('rf_max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('rf_min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('rf_min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('rf_max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('rf_bootstrap', [True, False]),
            'random_state': 42
        }
        
        if self.task_type == 'classification':
            model = RandomForestClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'f1_weighted'
        else:
            model = RandomForestRegressor(**params)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        return scores.mean()
    
    def optimize_xgboost(self, X_train, y_train, n_trials: int = 100, timeout: int = 1800):
        """
        Optimize XGBoost hyperparameters
        """
        print("Optimizing XGBoost hyperparameters...")
        
        def objective(trial):
            return self.xgboost_objective(trial, X_train, y_train)
        
        direction = 'maximize' if self.task_type == 'classification' else 'maximize'  # neg_mse -> maximize
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params['xgboost'] = study.best_params
        self.best_scores['xgboost'] = study.best_value
        
        return study
    
    def optimize_random_forest(self, X_train, y_train, n_trials: int = 100, timeout: int = 1800):
        """
        Optimize Random Forest hyperparameters
        """
        print("Optimizing Random Forest hyperparameters...")
        
        def objective(trial):
            return self.random_forest_objective(trial, X_train, y_train)
        
        direction = 'maximize' if self.task_type == 'classification' else 'maximize'  # neg_mse -> maximize
        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        self.best_params['random_forest'] = study.best_params
        self.best_scores['random_forest'] = study.best_value
        
        return study
    
    def train_optimized_models(self, X_train, y_train):
        """
        Train models with optimized hyperparameters
        """
        if 'xgboost' in self.best_params:
            xgb_params = {k.replace('xgb_', ''): v for k, v in self.best_params['xgboost'].items()}
            if self.task_type == 'classification':
                self.models['xgboost'] = XGBClassifier(**xgb_params)
            else:
                self.models['xgboost'] = XGBRegressor(**xgb_params)
            
            self.models['xgboost'].fit(X_train, y_train)
        
        if 'random_forest' in self.best_params:
            rf_params = {k.replace('rf_', ''): v for k, v in self.best_params['random_forest'].items()}
            if self.task_type == 'classification':
                self.models['random_forest'] = RandomForestClassifier(**rf_params)
            else:
                self.models['random_forest'] = RandomForestRegressor(**rf_params)
            
            self.models['random_forest'].fit(X_train, y_train)
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate optimized models on test set
        """
        results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            if self.task_type == 'classification':
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                results[model_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'predictions': y_pred.tolist()
                }
            else:
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                results[model_name] = {
                    'mse': mse,
                    'rmse': np.sqrt(mse),
                    'r2_score': r2,
                    'predictions': y_pred.tolist()
                }
        
        return results
    
    def save_results(self, evaluation_results: Dict):
        """
        Save optimization and evaluation results
        """
        results = {
            'task_type': self.task_type,
            'best_parameters': self.best_params,
            'cv_scores': self.best_scores,
            'test_evaluation': evaluation_results,
            'optimization_timestamp': str(datetime.now())
        }
        
        os.makedirs('hyperparameter_optimization/results', exist_ok=True)
        
        # Save results
        with open('hyperparameter_optimization/results/ml_optimization_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save models
        for model_name, model in self.models.items():
            joblib.dump(model, f'hyperparameter_optimization/results/{model_name}_optimized.pkl')
        
        print(f"Results saved to hyperparameter_optimization/results/")
        return results
    
    def run_full_optimization(self, X: pd.DataFrame, y: pd.Series, 
                            n_trials: int = 50, timeout_per_model: int = 900):
        """
        Run complete hyperparameter optimization pipeline
        """
        print(f"Starting {self.task_type} hyperparameter optimization...")
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Optimize models
        xgb_study = self.optimize_xgboost(X_train, y_train, n_trials, timeout_per_model)
        rf_study = self.optimize_random_forest(X_train, y_train, n_trials, timeout_per_model)
        
        # Train optimized models
        self.train_optimized_models(X_train, y_train)
        
        # Evaluate models
        evaluation_results = self.evaluate_models(X_test, y_test)
        
        # Save results
        final_results = self.save_results(evaluation_results)
        
        # Print summary
        print("\n=== Optimization Summary ===")
        for model_name in self.best_params.keys():
            print(f"\n{model_name.upper()}:")
            print(f"  Best CV Score: {self.best_scores[model_name]:.4f}")
            if model_name in evaluation_results:
                if self.task_type == 'classification':
                    print(f"  Test Accuracy: {evaluation_results[model_name]['accuracy']:.4f}")
                    print(f"  Test F1 Score: {evaluation_results[model_name]['f1_score']:.4f}")
                else:
                    print(f"  Test RMSE: {evaluation_results[model_name]['rmse']:.4f}")
                    print(f"  Test R² Score: {evaluation_results[model_name]['r2_score']:.4f}")
        
        return final_results

class DrillingDataOptimizer(MLHyperparameterOptimizer):
    """
    Specialized optimizer for drilling data analysis
    """
    
    def __init__(self, task_type: str = 'classification'):
        super().__init__(task_type)
    
    def load_drilling_data(self, data_path: str = None):
        """
        Load and preprocess drilling data
        """
        if data_path and os.path.exists(data_path):
            data = pd.read_csv(data_path)
        else:
            # Generate sample drilling data for demonstration
            np.random.seed(42)
            n_samples = 1000
            
            data = pd.DataFrame({
                'depth': np.random.uniform(1000, 5000, n_samples),
                'pressure': np.random.uniform(100, 500, n_samples),
                'temperature': np.random.uniform(50, 200, n_samples),
                'mud_weight': np.random.uniform(8, 15, n_samples),
                'flow_rate': np.random.uniform(200, 800, n_samples),
                'torque': np.random.uniform(5000, 25000, n_samples),
                'rpm': np.random.uniform(60, 180, n_samples),
                'formation_type': np.random.choice(['sandstone', 'shale', 'limestone', 'dolomite'], n_samples)
            })
            
            # Create target variable based on drilling conditions
            if self.task_type == 'classification':
                # Predict drilling difficulty: easy, medium, hard
                conditions = (
                    (data['depth'] > 3000) & (data['pressure'] > 300) |
                    (data['temperature'] > 150) & (data['mud_weight'] > 12)
                )
                data['target'] = np.where(
                    conditions, 'hard',
                    np.where(data['depth'] > 2000, 'medium', 'easy')
                )
            else:
                # Predict drilling rate (ROP)
                data['target'] = (
                    100 - (data['depth'] / 100) + 
                    (data['flow_rate'] / 10) - 
                    (data['pressure'] / 20) + 
                    np.random.normal(0, 5, n_samples)
                )
        
        return data
    
    def optimize_drilling_models(self, data_path: str = None, n_trials: int = 30):
        """
        Run optimization specifically for drilling data
        """
        # Load data
        data = self.load_drilling_data(data_path)
        
        # Separate features and target
        X = data.drop('target', axis=1)
        y = data['target']
        
        # Run optimization
        results = self.run_full_optimization(X, y, n_trials=n_trials)
        
        return results

if __name__ == "__main__":
    # Example usage for drilling data classification
    print("Running drilling data classification optimization...")
    classifier_optimizer = DrillingDataOptimizer(task_type='classification')
    classification_results = classifier_optimizer.optimize_drilling_models(n_trials=20)
    
    print("\n" + "="*50)
    print("Running drilling data regression optimization...")
    regressor_optimizer = DrillingDataOptimizer(task_type='regression')
    regression_results = regressor_optimizer.optimize_drilling_models(n_trials=20)
    
    print("\nOptimization completed! Check results in hyperparameter_optimization/results/")