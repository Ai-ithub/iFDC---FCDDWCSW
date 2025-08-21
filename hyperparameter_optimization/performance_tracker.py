import pandas as pd
import numpy as np
from sklearn.model_selection import (
    cross_val_score, cross_validate, StratifiedKFold, KFold,
    TimeSeriesSplit, GroupKFold, validation_curve, learning_curve
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import warnings
from dataclasses import dataclass, asdict
import pickle
from pathlib import Path

warnings.filterwarnings('ignore')

@dataclass
class PerformanceMetrics:
    """Data class to store performance metrics"""
    # Classification metrics
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    
    # Regression metrics
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2: Optional[float] = None
    mape: Optional[float] = None
    
    # Cross-validation metrics
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    cv_scores: Optional[List[float]] = None
    
    # Training metrics
    training_time: Optional[float] = None
    prediction_time: Optional[float] = None
    model_size_mb: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)

@dataclass
class ExperimentResult:
    """Data class to store complete experiment results"""
    experiment_id: str
    model_name: str
    task_type: str
    hyperparameters: Dict[str, Any]
    metrics: PerformanceMetrics
    timestamp: str
    dataset_info: Dict[str, Any]
    cv_strategy: str
    
    def to_dict(self):
        result = asdict(self)
        result['metrics'] = self.metrics.to_dict()
        return result

class CrossValidationTracker:
    """Advanced cross-validation and performance tracking system"""
    
    def __init__(self, results_dir: str = "hyperparameter_optimization/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.experiments = []
        self.current_experiment_id = None
        
    def generate_experiment_id(self) -> str:
        """Generate unique experiment ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}_{np.random.randint(1000, 9999)}"
    
    def setup_cv_strategy(self, cv_type: str, n_splits: int = 5, 
                         shuffle: bool = True, random_state: int = 42,
                         **kwargs) -> Any:
        """Setup cross-validation strategy"""
        cv_strategies = {
            'kfold': KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state),
            'stratified': StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state),
            'timeseries': TimeSeriesSplit(n_splits=n_splits),
            'group': GroupKFold(n_splits=n_splits)
        }
        
        if cv_type not in cv_strategies:
            raise ValueError(f"Unsupported CV type: {cv_type}. Choose from {list(cv_strategies.keys())}")
        
        return cv_strategies[cv_type]
    
    def calculate_classification_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        # Add ROC AUC for binary and multiclass
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception:
                metrics['roc_auc'] = None
        
        return metrics
    
    def calculate_regression_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Calculate comprehensive regression metrics"""
        mse = mean_squared_error(y_true, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Add MAPE if no zero values in y_true
        if not np.any(y_true == 0):
            metrics['mape'] = mean_absolute_percentage_error(y_true, y_pred)
        else:
            metrics['mape'] = None
        
        return metrics
    
    def perform_cross_validation(self, model, X, y, cv_strategy, 
                               scoring: str = 'accuracy', 
                               return_train_score: bool = True) -> Dict[str, Any]:
        """Perform comprehensive cross-validation"""
        import time
        
        start_time = time.time()
        
        # Perform cross-validation
        cv_results = cross_validate(
            model, X, y, cv=cv_strategy, scoring=scoring,
            return_train_score=return_train_score,
            return_estimator=True
        )
        
        end_time = time.time()
        
        # Calculate statistics
        test_scores = cv_results['test_score']
        train_scores = cv_results['train_score'] if return_train_score else None
        
        results = {
            'test_scores': test_scores.tolist(),
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'cv_time': end_time - start_time,
            'n_splits': len(test_scores)
        }
        
        if train_scores is not None:
            results.update({
                'train_scores': train_scores.tolist(),
                'train_mean': train_scores.mean(),
                'train_std': train_scores.std(),
                'overfitting_score': train_scores.mean() - test_scores.mean()
            })
        
        return results
    
    def evaluate_model_comprehensive(self, model, X_train, X_test, y_train, y_test,
                                   task_type: str = 'classification',
                                   cv_strategy=None, scoring: str = None) -> PerformanceMetrics:
        """Comprehensive model evaluation"""
        import time
        import sys
        
        # Training time
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Prediction time
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Model size estimation
        try:
            model_size_mb = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
        except Exception:
            model_size_mb = None
        
        # Initialize metrics
        metrics = PerformanceMetrics(
            training_time=training_time,
            prediction_time=prediction_time,
            model_size_mb=model_size_mb
        )
        
        # Task-specific metrics
        if task_type == 'classification':
            y_pred_proba = None
            if hasattr(model, 'predict_proba'):
                try:
                    y_pred_proba = model.predict_proba(X_test)
                except Exception:
                    pass
            
            class_metrics = self.calculate_classification_metrics(y_test, y_pred, y_pred_proba)
            metrics.accuracy = class_metrics['accuracy']
            metrics.precision = class_metrics['precision']
            metrics.recall = class_metrics['recall']
            metrics.f1_score = class_metrics['f1_score']
            metrics.roc_auc = class_metrics.get('roc_auc')
            
            if scoring is None:
                scoring = 'f1_weighted'
        
        elif task_type == 'regression':
            reg_metrics = self.calculate_regression_metrics(y_test, y_pred)
            metrics.mse = reg_metrics['mse']
            metrics.rmse = reg_metrics['rmse']
            metrics.mae = reg_metrics['mae']
            metrics.r2 = reg_metrics['r2']
            metrics.mape = reg_metrics.get('mape')
            
            if scoring is None:
                scoring = 'neg_mean_squared_error'
        
        # Cross-validation if strategy provided
        if cv_strategy is not None:
            cv_results = self.perform_cross_validation(
                model, X_train, y_train, cv_strategy, scoring
            )
            metrics.cv_mean = cv_results['test_mean']
            metrics.cv_std = cv_results['test_std']
            metrics.cv_scores = cv_results['test_scores']
        
        return metrics
    
    def track_experiment(self, model_name: str, model, hyperparameters: Dict[str, Any],
                        X_train, X_test, y_train, y_test, task_type: str = 'classification',
                        cv_strategy=None, dataset_info: Dict[str, Any] = None) -> str:
        """Track a complete experiment"""
        experiment_id = self.generate_experiment_id()
        self.current_experiment_id = experiment_id
        
        # Evaluate model
        metrics = self.evaluate_model_comprehensive(
            model, X_train, X_test, y_train, y_test, task_type, cv_strategy
        )
        
        # Create experiment result
        experiment = ExperimentResult(
            experiment_id=experiment_id,
            model_name=model_name,
            task_type=task_type,
            hyperparameters=hyperparameters,
            metrics=metrics,
            timestamp=datetime.now().isoformat(),
            dataset_info=dataset_info or {},
            cv_strategy=str(cv_strategy) if cv_strategy else 'none'
        )
        
        self.experiments.append(experiment)
        
        # Save experiment
        self.save_experiment(experiment)
        
        return experiment_id
    
    def save_experiment(self, experiment: ExperimentResult):
        """Save individual experiment results"""
        experiment_file = self.results_dir / f"{experiment.experiment_id}.json"
        
        with open(experiment_file, 'w') as f:
            json.dump(experiment.to_dict(), f, indent=2, default=str)
    
    def save_all_experiments(self):
        """Save all experiments to a summary file"""
        summary_file = self.results_dir / "experiments_summary.json"
        
        summary = {
            'total_experiments': len(self.experiments),
            'last_updated': datetime.now().isoformat(),
            'experiments': [exp.to_dict() for exp in self.experiments]
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def load_experiments(self) -> List[ExperimentResult]:
        """Load all saved experiments"""
        experiments = []
        
        for experiment_file in self.results_dir.glob("exp_*.json"):
            try:
                with open(experiment_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct experiment object
                metrics_data = data['metrics']
                metrics = PerformanceMetrics(**metrics_data)
                
                experiment = ExperimentResult(
                    experiment_id=data['experiment_id'],
                    model_name=data['model_name'],
                    task_type=data['task_type'],
                    hyperparameters=data['hyperparameters'],
                    metrics=metrics,
                    timestamp=data['timestamp'],
                    dataset_info=data['dataset_info'],
                    cv_strategy=data['cv_strategy']
                )
                
                experiments.append(experiment)
            
            except Exception as e:
                print(f"Error loading experiment {experiment_file}: {e}")
        
        self.experiments = experiments
        return experiments
    
    def get_best_experiments(self, metric: str = 'f1_score', top_k: int = 5) -> List[ExperimentResult]:
        """Get top performing experiments based on a metric"""
        if not self.experiments:
            self.load_experiments()
        
        # Filter experiments that have the requested metric
        valid_experiments = []
        for exp in self.experiments:
            metric_value = getattr(exp.metrics, metric, None)
            if metric_value is not None:
                valid_experiments.append((exp, metric_value))
        
        # Sort by metric (descending for most metrics, ascending for error metrics)
        error_metrics = ['mse', 'rmse', 'mae', 'mape']
        reverse = metric not in error_metrics
        
        valid_experiments.sort(key=lambda x: x[1], reverse=reverse)
        
        return [exp for exp, _ in valid_experiments[:top_k]]
    
    def compare_experiments(self, experiment_ids: List[str]) -> pd.DataFrame:
        """Compare multiple experiments"""
        if not self.experiments:
            self.load_experiments()
        
        # Find experiments by ID
        experiments_to_compare = []
        for exp in self.experiments:
            if exp.experiment_id in experiment_ids:
                experiments_to_compare.append(exp)
        
        # Create comparison DataFrame
        comparison_data = []
        for exp in experiments_to_compare:
            row = {
                'experiment_id': exp.experiment_id,
                'model_name': exp.model_name,
                'task_type': exp.task_type,
                'timestamp': exp.timestamp
            }
            
            # Add hyperparameters
            for param, value in exp.hyperparameters.items():
                row[f'param_{param}'] = value
            
            # Add metrics
            metrics_dict = exp.metrics.to_dict()
            for metric, value in metrics_dict.items():
                if value is not None:
                    row[f'metric_{metric}'] = value
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def generate_performance_report(self, output_file: str = None) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        if not self.experiments:
            self.load_experiments()
        
        report = {
            'summary': {
                'total_experiments': len(self.experiments),
                'unique_models': len(set(exp.model_name for exp in self.experiments)),
                'task_types': list(set(exp.task_type for exp in self.experiments)),
                'date_range': {
                    'earliest': min(exp.timestamp for exp in self.experiments) if self.experiments else None,
                    'latest': max(exp.timestamp for exp in self.experiments) if self.experiments else None
                }
            },
            'best_performers': {},
            'model_comparison': {},
            'hyperparameter_analysis': {}
        }
        
        # Best performers by metric
        metrics_to_check = ['accuracy', 'f1_score', 'r2', 'mse']
        for metric in metrics_to_check:
            best_exps = self.get_best_experiments(metric, top_k=3)
            if best_exps:
                report['best_performers'][metric] = [
                    {
                        'experiment_id': exp.experiment_id,
                        'model_name': exp.model_name,
                        'value': getattr(exp.metrics, metric)
                    }
                    for exp in best_exps
                ]
        
        # Model comparison
        model_stats = {}
        for exp in self.experiments:
            model_name = exp.model_name
            if model_name not in model_stats:
                model_stats[model_name] = {
                    'count': 0,
                    'avg_training_time': [],
                    'best_score': None,
                    'task_types': set()
                }
            
            model_stats[model_name]['count'] += 1
            model_stats[model_name]['task_types'].add(exp.task_type)
            
            if exp.metrics.training_time:
                model_stats[model_name]['avg_training_time'].append(exp.metrics.training_time)
        
        # Convert sets to lists for JSON serialization
        for model_name, stats in model_stats.items():
            stats['task_types'] = list(stats['task_types'])
            if stats['avg_training_time']:
                stats['avg_training_time'] = np.mean(stats['avg_training_time'])
            else:
                stats['avg_training_time'] = None
        
        report['model_comparison'] = model_stats
        
        # Save report if output file specified
        if output_file:
            output_path = self.results_dir / output_file
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
        
        return report

if __name__ == "__main__":
    # Example usage
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    
    # Create sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize tracker
    tracker = CrossValidationTracker()
    
    # Setup CV strategy
    cv_strategy = tracker.setup_cv_strategy('stratified', n_splits=5)
    
    # Track experiment
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    hyperparams = {'n_estimators': 100, 'max_depth': None, 'random_state': 42}
    dataset_info = {'n_samples': len(X), 'n_features': X.shape[1], 'n_classes': len(np.unique(y))}
    
    experiment_id = tracker.track_experiment(
        model_name='RandomForest',
        model=model,
        hyperparameters=hyperparams,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        task_type='classification',
        cv_strategy=cv_strategy,
        dataset_info=dataset_info
    )
    
    print(f"Experiment tracked with ID: {experiment_id}")
    
    # Generate report
    report = tracker.generate_performance_report('performance_report.json')
    print("Performance report generated!")