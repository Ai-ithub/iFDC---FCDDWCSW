#!/usr/bin/env python3
"""
Machine Learning Hyperparameter Optimization Example

This example demonstrates how to use the ML hyperparameter optimization framework
for XGBoost, Random Forest, and drilling data analysis.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from ml_hyperopt import XGBoostOptimizer, RandomForestOptimizer, DrillingDataOptimizer
from performance_tracker import CrossValidationTracker
from visualizer import HyperparameterVisualizationSuite

def create_sample_classification_data(n_samples=1000, n_features=20, n_classes=3):
    """
    Create sample classification dataset
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
    
    Returns:
        X, y: Features and target
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.7),
        n_redundant=int(n_features * 0.2),
        n_classes=n_classes,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # Convert to DataFrame for easier handling
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

def create_sample_regression_data(n_samples=1000, n_features=15):
    """
    Create sample regression dataset
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
    
    Returns:
        X, y: Features and target
    """
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=int(n_features * 0.8),
        noise=0.1,
        random_state=42
    )
    
    # Convert to DataFrame
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    return X_df, y_series

def create_drilling_sample_data(n_samples=800):
    """
    Create sample drilling data with realistic features
    
    Args:
        n_samples: Number of samples
    
    Returns:
        DataFrame with drilling features and target
    """
    np.random.seed(42)
    
    # Drilling parameters
    depth = np.random.uniform(1000, 5000, n_samples)
    pressure = np.random.uniform(2000, 8000, n_samples) + depth * 0.5
    temperature = np.random.uniform(60, 200, n_samples) + depth * 0.02
    mud_weight = np.random.uniform(8.5, 12.0, n_samples)
    flow_rate = np.random.uniform(200, 800, n_samples)
    rpm = np.random.uniform(80, 200, n_samples)
    torque = np.random.uniform(5000, 25000, n_samples)
    
    # Formation properties
    porosity = np.random.uniform(0.05, 0.35, n_samples)
    permeability = np.random.lognormal(0, 2, n_samples)
    
    # Well log data
    gamma_ray = np.random.uniform(20, 150, n_samples)
    resistivity = np.random.lognormal(1, 1.5, n_samples)
    
    # Create target based on formation type (classification)
    # 0: Sandstone, 1: Shale, 2: Limestone
    formation_type = np.zeros(n_samples)
    
    # Sandstone: high porosity, low gamma ray
    sandstone_mask = (porosity > 0.15) & (gamma_ray < 80)
    formation_type[sandstone_mask] = 0
    
    # Shale: low porosity, high gamma ray
    shale_mask = (porosity < 0.15) & (gamma_ray > 100)
    formation_type[shale_mask] = 1
    
    # Limestone: medium porosity, medium gamma ray
    limestone_mask = ~(sandstone_mask | shale_mask)
    formation_type[limestone_mask] = 2
    
    # Create DataFrame
    data = pd.DataFrame({
        'depth': depth,
        'pressure': pressure,
        'temperature': temperature,
        'mud_weight': mud_weight,
        'flow_rate': flow_rate,
        'rpm': rpm,
        'torque': torque,
        'porosity': porosity,
        'permeability': permeability,
        'gamma_ray': gamma_ray,
        'resistivity': resistivity,
        'formation_type': formation_type.astype(int)
    })
    
    return data

def run_xgboost_classification_example(results_dir):
    """
    Run XGBoost classification optimization example
    """
    print("\n" + "="*50)
    print("XGBoost Classification Optimization")
    print("="*50)
    
    # Create dataset
    print("\n1. Creating classification dataset...")
    X, y = create_sample_classification_data(n_samples=1000, n_features=20, n_classes=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Classes: {sorted(y.unique())}")
    
    # Initialize optimizer
    print("\n2. Initializing XGBoost optimizer...")
    xgb_optimizer = XGBoostOptimizer(
        config_path="config/ml_config.yaml",
        results_dir=str(results_dir / "xgboost_classification")
    )
    
    # Run optimization
    print("\n3. Running hyperparameter optimization...")
    study = xgb_optimizer.optimize(
        X_train, y_train,
        task_type='classification',
        n_trials=30,
        timeout=600  # 10 minutes
    )
    
    print(f"\n4. Optimization Results:")
    print(f"   Best value: {study.best_value:.6f}")
    print("   Best parameters:")
    for param, value in study.best_params.items():
        print(f"     {param}: {value}")
    
    # Create and evaluate best model
    print("\n5. Evaluating best model...")
    best_model = xgb_optimizer.create_optimized_model(study.best_params, task_type='classification')
    best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    accuracy = best_model.score(X_test, y_test)
    
    print(f"   Test accuracy: {accuracy:.4f}")
    print("\n   Classification Report:")
    print(classification_report(y_test, y_pred, target_names=[f'Class_{i}' for i in sorted(y.unique())]))
    
    return study, best_model

def run_random_forest_regression_example(results_dir):
    """
    Run Random Forest regression optimization example
    """
    print("\n" + "="*50)
    print("Random Forest Regression Optimization")
    print("="*50)
    
    # Create dataset
    print("\n1. Creating regression dataset...")
    X, y = create_sample_regression_data(n_samples=1000, n_features=15)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Target range: [{y.min():.2f}, {y.max():.2f}]")
    
    # Initialize optimizer
    print("\n2. Initializing Random Forest optimizer...")
    rf_optimizer = RandomForestOptimizer(
        config_path="config/ml_config.yaml",
        results_dir=str(results_dir / "random_forest_regression")
    )
    
    # Run optimization
    print("\n3. Running hyperparameter optimization...")
    study = rf_optimizer.optimize(
        X_train, y_train,
        task_type='regression',
        n_trials=25,
        timeout=600
    )
    
    print(f"\n4. Optimization Results:")
    print(f"   Best value: {study.best_value:.6f}")
    print("   Best parameters:")
    for param, value in study.best_params.items():
        print(f"     {param}: {value}")
    
    # Create and evaluate best model
    print("\n5. Evaluating best model...")
    best_model = rf_optimizer.create_optimized_model(study.best_params, task_type='regression')
    best_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   Test MSE: {mse:.4f}")
    print(f"   Test R²: {r2:.4f}")
    
    return study, best_model

def run_drilling_data_example(results_dir):
    """
    Run drilling data optimization example
    """
    print("\n" + "="*50)
    print("Drilling Data Optimization")
    print("="*50)
    
    # Create drilling dataset
    print("\n1. Creating drilling dataset...")
    drilling_data = create_drilling_sample_data(n_samples=800)
    
    print(f"   Dataset shape: {drilling_data.shape}")
    print(f"   Formation types: {sorted(drilling_data['formation_type'].unique())}")
    print("   Features:", list(drilling_data.columns[:-1]))
    
    # Save sample data
    data_path = results_dir / "sample_drilling_data.csv"
    drilling_data.to_csv(data_path, index=False)
    print(f"   Sample data saved to: {data_path}")
    
    # Initialize drilling optimizer
    print("\n2. Initializing drilling data optimizer...")
    drilling_optimizer = DrillingDataOptimizer(
        config_path="config/ml_config.yaml",
        results_dir=str(results_dir / "drilling_optimization")
    )
    
    # Load and preprocess data
    print("\n3. Loading and preprocessing data...")
    X_processed, y_processed = drilling_optimizer.load_and_preprocess_data(
        str(data_path),
        target_column="formation_type"
    )
    
    print(f"   Processed features shape: {X_processed.shape}")
    print(f"   Target shape: {y_processed.shape}")
    
    # Run optimization
    print("\n4. Running drilling-specific optimization...")
    study = drilling_optimizer.optimize(
        X_processed, y_processed,
        model_type='xgboost',
        task_type='classification',
        n_trials=20,
        timeout=600
    )
    
    print(f"\n5. Optimization Results:")
    print(f"   Best value: {study.best_value:.6f}")
    print("   Best parameters:")
    for param, value in study.best_params.items():
        print(f"     {param}: {value}")
    
    return study, drilling_data

def create_comprehensive_visualizations(studies_dict, results_dir):
    """
    Create comprehensive visualizations for all optimization studies
    """
    print("\n" + "="*50)
    print("Creating Comprehensive Visualizations")
    print("="*50)
    
    visualizer = HyperparameterVisualizationSuite(str(results_dir / "visualizations"))
    
    for study_name, study in studies_dict.items():
        print(f"\n   Creating visualizations for {study_name}...")
        
        # Individual plots
        hist_fig = visualizer.plot_optimization_history(
            study, 
            save_path=f'{study_name}_optimization_history.html'
        )
        
        imp_fig = visualizer.plot_parameter_importance(
            study, 
            save_path=f'{study_name}_parameter_importance.html'
        )
        
        par_fig = visualizer.plot_parallel_coordinate(
            study, 
            save_path=f'{study_name}_parallel_coordinates.html'
        )
        
        # Dashboard
        dashboard_path = visualizer.create_optimization_dashboard(
            study, 
            save_path=f'{study_name}_dashboard.html'
        )
        
        print(f"     ✓ Dashboard created: {dashboard_path}")
    
    print("\n   All visualizations created successfully!")

def run_performance_tracking_example(studies_dict, results_dir):
    """
    Demonstrate performance tracking and experiment comparison
    """
    print("\n" + "="*50)
    print("Performance Tracking and Comparison")
    print("="*50)
    
    # Initialize performance tracker
    tracker = CrossValidationTracker(str(results_dir / "experiments"))
    
    # Create sample experiments for comparison
    experiments = []
    
    for study_name, study in studies_dict.items():
        # Create mock experiment result
        from performance_tracker import ExperimentResult, PerformanceMetrics
        
        # Mock metrics based on study results
        if 'classification' in study_name:
            metrics = PerformanceMetrics(
                accuracy=abs(study.best_value),
                precision=abs(study.best_value) * 0.95,
                recall=abs(study.best_value) * 0.98,
                f1_score=abs(study.best_value),
                roc_auc=abs(study.best_value) * 1.02 if abs(study.best_value) < 0.98 else 0.98
            )
        else:
            metrics = PerformanceMetrics(
                mse=abs(study.best_value),
                rmse=np.sqrt(abs(study.best_value)),
                mae=abs(study.best_value) * 0.8,
                r2_score=1 - abs(study.best_value) / 100  # Mock R² calculation
            )
        
        experiment = ExperimentResult(
            model_name=study_name,
            hyperparameters=study.best_params,
            metrics=metrics,
            cv_scores=[abs(study.best_value) + np.random.normal(0, 0.01) for _ in range(5)],
            training_time=np.random.uniform(10, 300),  # Mock training time
            model_size=np.random.randint(1000, 50000)  # Mock model size
        )
        
        experiments.append(experiment)
        tracker.save_experiment(experiment)
    
    print(f"\n   Saved {len(experiments)} experiments")
    
    # Load and compare experiments
    loaded_experiments = tracker.load_experiments()
    print(f"   Loaded {len(loaded_experiments)} experiments")
    
    # Get best experiments
    if any('classification' in exp.model_name for exp in loaded_experiments):
        best_classification = tracker.get_best_experiments(
            metric="f1_score", 
            n_best=3,
            experiments=[exp for exp in loaded_experiments if 'classification' in exp.model_name]
        )
        print(f"\n   Best classification models:")
        for i, exp in enumerate(best_classification, 1):
            print(f"     {i}. {exp.model_name}: F1={exp.metrics.f1_score:.4f}")
    
    # Generate comparison report
    comparison = tracker.compare_experiments(
        loaded_experiments,
        metrics=["accuracy", "f1_score", "precision", "recall"] if any('classification' in exp.model_name for exp in loaded_experiments) else ["mse", "rmse", "r2_score"]
    )
    
    print(f"\n   Experiment comparison:")
    print(comparison)
    
    # Generate performance report
    report = tracker.generate_performance_report(loaded_experiments)
    print(f"\n   Performance report generated")
    
    return loaded_experiments

def main():
    """
    Main function demonstrating ML hyperparameter optimization
    """
    print("=" * 60)
    print("Machine Learning Hyperparameter Optimization Example")
    print("=" * 60)
    
    # Set random seeds
    np.random.seed(42)
    
    # Create results directory
    results_dir = Path("results/ml_example")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all studies
    studies = {}
    
    # Run different optimization examples
    try:
        # XGBoost Classification
        xgb_study, xgb_model = run_xgboost_classification_example(results_dir)
        studies['xgboost_classification'] = xgb_study
        
        # Random Forest Regression
        rf_study, rf_model = run_random_forest_regression_example(results_dir)
        studies['random_forest_regression'] = rf_study
        
        # Drilling Data
        drilling_study, drilling_data = run_drilling_data_example(results_dir)
        studies['drilling_optimization'] = drilling_study
        
    except Exception as e:
        print(f"Error during optimization: {e}")
        print("Continuing with available results...")
    
    # Create visualizations
    if studies:
        create_comprehensive_visualizations(studies, results_dir)
        
        # Performance tracking
        experiments = run_performance_tracking_example(studies, results_dir)
    
    # Final summary
    print("\n" + "="*60)
    print("OPTIMIZATION EXAMPLES COMPLETE!")
    print("="*60)
    print(f"Results directory: {results_dir}")
    print(f"Completed optimizations: {len(studies)}")
    
    if studies:
        print("\nBest results summary:")
        for name, study in studies.items():
            print(f"  {name}: {study.best_value:.6f}")
    
    print("\nGenerated files:")
    print("  - Individual optimization results in subdirectories")
    print("  - Comprehensive visualizations in visualizations/")
    print("  - Experiment tracking data in experiments/")
    print("  - Sample drilling data: sample_drilling_data.csv")
    
    print("\nNext steps:")
    print("  1. Review the optimization dashboards")
    print("  2. Analyze parameter importance plots")
    print("  3. Use best parameters for production models")
    print("  4. Experiment with different search spaces")

if __name__ == "__main__":
    main()