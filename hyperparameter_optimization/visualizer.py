import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import optuna
from typing import Dict, List, Any, Optional, Tuple
import json
from pathlib import Path
import warnings
from performance_tracker import CrossValidationTracker, ExperimentResult

warnings.filterwarnings('ignore')

class HyperparameterVisualizationSuite:
    """Comprehensive visualization suite for hyperparameter optimization results"""
    
    def __init__(self, results_dir: str = "hyperparameter_optimization/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.tracker = CrossValidationTracker(results_dir)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_optimization_history(self, study: optuna.Study, save_path: str = None) -> go.Figure:
        """Plot optimization history showing how the objective value improves over trials"""
        trials_df = study.trials_dataframe()
        
        fig = go.Figure()
        
        # Plot all trial values
        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=trials_df['value'],
            mode='markers',
            name='Trial Values',
            marker=dict(color='lightblue', size=6),
            hovertemplate='Trial: %{x}<br>Value: %{y:.4f}<extra></extra>'
        ))
        
        # Plot best values so far
        best_values = []
        current_best = float('inf') if study.direction == optuna.study.StudyDirection.MINIMIZE else float('-inf')
        
        for trial in study.trials:
            if study.direction == optuna.study.StudyDirection.MINIMIZE:
                current_best = min(current_best, trial.value)
            else:
                current_best = max(current_best, trial.value)
            best_values.append(current_best)
        
        fig.add_trace(go.Scatter(
            x=trials_df.index,
            y=best_values,
            mode='lines',
            name='Best Value So Far',
            line=dict(color='red', width=2),
            hovertemplate='Trial: %{x}<br>Best Value: %{y:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Optimization History',
            xaxis_title='Trial Number',
            yaxis_title='Objective Value',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(self.results_dir / save_path)
        
        return fig
    
    def plot_parameter_importance(self, study: optuna.Study, save_path: str = None) -> go.Figure:
        """Plot parameter importance based on fANOVA"""
        try:
            importance = optuna.importance.get_param_importances(study)
            
            params = list(importance.keys())
            values = list(importance.values())
            
            fig = go.Figure(data=[
                go.Bar(
                    x=values,
                    y=params,
                    orientation='h',
                    marker_color='skyblue',
                    text=[f'{v:.3f}' for v in values],
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title='Hyperparameter Importance',
                xaxis_title='Importance',
                yaxis_title='Parameters',
                template='plotly_white',
                height=max(400, len(params) * 30)
            )
            
            if save_path:
                fig.write_html(self.results_dir / save_path)
            
            return fig
        
        except Exception as e:
            print(f"Could not calculate parameter importance: {e}")
            return None
    
    def plot_parallel_coordinate(self, study: optuna.Study, save_path: str = None) -> go.Figure:
        """Plot parallel coordinate plot for hyperparameter relationships"""
        trials_df = study.trials_dataframe()
        
        # Get parameter columns
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        
        if not param_cols:
            print("No parameter columns found")
            return None
        
        # Prepare data for parallel coordinates
        dimensions = []
        
        for col in param_cols:
            param_name = col.replace('params_', '')
            values = trials_df[col].dropna()
            
            if values.dtype in ['object', 'category']:
                # Categorical parameter
                unique_vals = values.unique()
                label_map = {val: i for i, val in enumerate(unique_vals)}
                numeric_values = values.map(label_map)
                
                dimensions.append(dict(
                    label=param_name,
                    values=numeric_values,
                    tickvals=list(range(len(unique_vals))),
                    ticktext=unique_vals
                ))
            else:
                # Numeric parameter
                dimensions.append(dict(
                    label=param_name,
                    values=values
                ))
        
        # Add objective value
        dimensions.append(dict(
            label='Objective Value',
            values=trials_df['value']
        ))
        
        fig = go.Figure(data=
            go.Parcoords(
                line=dict(
                    color=trials_df['value'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Objective Value")
                ),
                dimensions=dimensions
            )
        )
        
        fig.update_layout(
            title='Hyperparameter Parallel Coordinates',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(self.results_dir / save_path)
        
        return fig
    
    def plot_parameter_slice(self, study: optuna.Study, param_name: str, save_path: str = None) -> go.Figure:
        """Plot slice plot for a specific parameter"""
        trials_df = study.trials_dataframe()
        param_col = f'params_{param_name}'
        
        if param_col not in trials_df.columns:
            print(f"Parameter {param_name} not found in study")
            return None
        
        fig = go.Figure()
        
        # Scatter plot of parameter vs objective value
        fig.add_trace(go.Scatter(
            x=trials_df[param_col],
            y=trials_df['value'],
            mode='markers',
            marker=dict(
                color=trials_df['value'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Objective Value")
            ),
            hovertemplate=f'{param_name}: %{{x}}<br>Objective: %{{y:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Parameter Slice: {param_name}',
            xaxis_title=param_name,
            yaxis_title='Objective Value',
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(self.results_dir / save_path)
        
        return fig
    
    def plot_cross_validation_results(self, cv_scores: List[float], model_name: str = "", 
                                    save_path: str = None) -> plt.Figure:
        """Plot cross-validation results"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        ax1.boxplot(cv_scores)
        ax1.set_title(f'CV Scores Distribution\n{model_name}')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # Individual scores
        ax2.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', markersize=8)
        ax2.axhline(y=np.mean(cv_scores), color='r', linestyle='--', 
                   label=f'Mean: {np.mean(cv_scores):.4f}')
        ax2.fill_between(range(1, len(cv_scores) + 1), 
                        np.mean(cv_scores) - np.std(cv_scores),
                        np.mean(cv_scores) + np.std(cv_scores),
                        alpha=0.2, color='red')
        ax2.set_title(f'CV Scores by Fold\n{model_name}')
        ax2.set_xlabel('Fold')
        ax2.set_ylabel('Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.results_dir / save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_model_comparison(self, experiments: List[ExperimentResult], 
                            metric: str = 'f1_score', save_path: str = None) -> go.Figure:
        """Compare multiple models on a specific metric"""
        # Prepare data
        model_data = {}
        for exp in experiments:
            model_name = exp.model_name
            metric_value = getattr(exp.metrics, metric, None)
            
            if metric_value is not None:
                if model_name not in model_data:
                    model_data[model_name] = []
                model_data[model_name].append(metric_value)
        
        if not model_data:
            print(f"No data found for metric: {metric}")
            return None
        
        fig = go.Figure()
        
        for model_name, values in model_data.items():
            fig.add_trace(go.Box(
                y=values,
                name=model_name,
                boxpoints='all',
                jitter=0.3,
                pointpos=-1.8
            ))
        
        fig.update_layout(
            title=f'Model Comparison: {metric.replace("_", " ").title()}',
            yaxis_title=metric.replace("_", " ").title(),
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(self.results_dir / save_path)
        
        return fig
    
    def plot_hyperparameter_heatmap(self, study: optuna.Study, param1: str, param2: str, 
                                   save_path: str = None) -> go.Figure:
        """Create heatmap showing interaction between two hyperparameters"""
        trials_df = study.trials_dataframe()
        
        param1_col = f'params_{param1}'
        param2_col = f'params_{param2}'
        
        if param1_col not in trials_df.columns or param2_col not in trials_df.columns:
            print(f"Parameters {param1} or {param2} not found in study")
            return None
        
        # Create pivot table
        pivot_data = trials_df.pivot_table(
            values='value',
            index=param2_col,
            columns=param1_col,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale='Viridis',
            hovertemplate=f'{param1}: %{{x}}<br>{param2}: %{{y}}<br>Objective: %{{z:.4f}}<extra></extra>'
        ))
        
        fig.update_layout(
            title=f'Hyperparameter Interaction: {param1} vs {param2}',
            xaxis_title=param1,
            yaxis_title=param2,
            template='plotly_white'
        )
        
        if save_path:
            fig.write_html(self.results_dir / save_path)
        
        return fig
    
    def plot_learning_curves(self, model, X, y, cv_folds: int = 5, 
                           train_sizes: np.ndarray = None, save_path: str = None) -> plt.Figure:
        """Plot learning curves to analyze model performance vs training set size"""
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X, y, cv=cv_folds, train_sizes=train_sizes,
            scoring='f1_weighted' if hasattr(model, 'predict_proba') else 'neg_mean_squared_error',
            n_jobs=-1
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(train_sizes_abs, train_mean, 'o-', color='blue', label='Training Score')
        ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, 
                       alpha=0.2, color='blue')
        
        ax.plot(train_sizes_abs, val_mean, 'o-', color='red', label='Validation Score')
        ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, 
                       alpha=0.2, color='red')
        
        ax.set_xlabel('Training Set Size')
        ax.set_ylabel('Score')
        ax.set_title('Learning Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(self.results_dir / save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_optimization_dashboard(self, study: optuna.Study, 
                                    experiments: List[ExperimentResult] = None,
                                    save_path: str = "optimization_dashboard.html") -> str:
        """Create comprehensive optimization dashboard"""
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Optimization History', 'Parameter Importance',
                'Parallel Coordinates', 'Best Trials Comparison',
                'Parameter Distribution', 'Model Performance'
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"colspan": 2}, None],
                [{"secondary_y": False}, {"secondary_y": False}]
            ]
        )
        
        # 1. Optimization History
        trials_df = study.trials_dataframe()
        best_values = []
        current_best = float('inf') if study.direction == optuna.study.StudyDirection.MINIMIZE else float('-inf')
        
        for trial in study.trials:
            if study.direction == optuna.study.StudyDirection.MINIMIZE:
                current_best = min(current_best, trial.value)
            else:
                current_best = max(current_best, trial.value)
            best_values.append(current_best)
        
        fig.add_trace(
            go.Scatter(x=trials_df.index, y=trials_df['value'], mode='markers', 
                      name='Trial Values', marker=dict(color='lightblue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=trials_df.index, y=best_values, mode='lines', 
                      name='Best So Far', line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. Parameter Importance
        try:
            importance = optuna.importance.get_param_importances(study)
            params = list(importance.keys())
            values = list(importance.values())
            
            fig.add_trace(
                go.Bar(x=values, y=params, orientation='h', name='Importance'),
                row=1, col=2
            )
        except Exception:
            pass
        
        # 3. Parameter Distribution (example with first numeric parameter)
        param_cols = [col for col in trials_df.columns if col.startswith('params_')]
        if param_cols:
            first_param = param_cols[0]
            param_values = trials_df[first_param].dropna()
            
            if param_values.dtype in ['int64', 'float64']:
                fig.add_trace(
                    go.Histogram(x=param_values, name=first_param.replace('params_', '')),
                    row=3, col=1
                )
        
        # 4. Model Performance (if experiments provided)
        if experiments:
            model_names = [exp.model_name for exp in experiments]
            f1_scores = [getattr(exp.metrics, 'f1_score', 0) for exp in experiments]
            
            fig.add_trace(
                go.Bar(x=model_names, y=f1_scores, name='F1 Score'),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="Hyperparameter Optimization Dashboard",
            showlegend=False
        )
        
        # Save dashboard
        dashboard_path = self.results_dir / save_path
        fig.write_html(dashboard_path)
        
        return str(dashboard_path)
    
    def generate_optimization_report(self, study: optuna.Study, 
                                   experiments: List[ExperimentResult] = None) -> Dict[str, Any]:
        """Generate comprehensive optimization report with visualizations"""
        report = {
            'study_summary': {
                'n_trials': len(study.trials),
                'best_value': study.best_value,
                'best_params': study.best_params,
                'direction': str(study.direction)
            },
            'visualizations': {}
        }
        
        # Generate and save visualizations
        try:
            # Optimization history
            hist_fig = self.plot_optimization_history(study, 'optimization_history.html')
            report['visualizations']['optimization_history'] = 'optimization_history.html'
            
            # Parameter importance
            imp_fig = self.plot_parameter_importance(study, 'parameter_importance.html')
            if imp_fig:
                report['visualizations']['parameter_importance'] = 'parameter_importance.html'
            
            # Parallel coordinates
            par_fig = self.plot_parallel_coordinate(study, 'parallel_coordinates.html')
            if par_fig:
                report['visualizations']['parallel_coordinates'] = 'parallel_coordinates.html'
            
            # Dashboard
            dashboard_path = self.create_optimization_dashboard(study, experiments)
            report['visualizations']['dashboard'] = 'optimization_dashboard.html'
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
        
        # Save report
        report_path = self.results_dir / 'optimization_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report

if __name__ == "__main__":
    # Example usage
    import optuna
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    # Create sample optimization study
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 10, 100)
        max_depth = trial.suggest_int('max_depth', 1, 10)
        
        X, y = make_classification(n_samples=500, n_features=10, random_state=42)
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        return scores.mean()
    
    # Run optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    # Create visualizations
    visualizer = HyperparameterVisualizationSuite()
    
    # Generate individual plots
    hist_fig = visualizer.plot_optimization_history(study)
    imp_fig = visualizer.plot_parameter_importance(study)
    par_fig = visualizer.plot_parallel_coordinate(study)
    
    # Generate comprehensive report
    report = visualizer.generate_optimization_report(study)
    
    print("Visualization suite demonstration completed!")
    print(f"Report saved to: {visualizer.results_dir / 'optimization_report.json'}")