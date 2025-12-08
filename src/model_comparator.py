"""
Model comparison and evaluation utilities.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    model_type: str  # 'neuromorphic' or 'traditional'
    
    # Classification metrics
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    roc_auc: float = 0.0
    
    # Confusion matrix
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    # Performance metrics
    training_time: float = 0.0
    inference_time: float = 0.0
    memory_usage: float = 0.0  # MB
    
    # Model characteristics
    n_parameters: int = 0
    model_size: float = 0.0  # MB
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'metrics': {
                'accuracy': self.accuracy,
                'precision': self.precision,
                'recall': self.recall,
                'f1_score': self.f1_score,
                'roc_auc': self.roc_auc
            },
            'confusion_matrix': {
                'TP': self.true_positives,
                'TN': self.true_negatives,
                'FP': self.false_positives,
                'FN': self.false_negatives
            },
            'performance': {
                'training_time': self.training_time,
                'inference_time': self.inference_time,
                'memory_usage': self.memory_usage
            },
            'characteristics': {
                'n_parameters': self.n_parameters,
                'model_size': self.model_size
            }
        }


class ModelComparator:
    """
    Compare neuromorphic and traditional ML models.
    
    Provides side-by-side comparison of performance, efficiency, and
    resource requirements.
    """
    
    def __init__(self):
        """Initialize model comparator."""
        self.results: List[ModelPerformance] = []
        
    def add_neuromorphic_model(self, model, model_name: str,
                              X_train: np.ndarray, y_train: np.ndarray,
                              X_test: np.ndarray, y_test: np.ndarray) -> ModelPerformance:
        """
        Evaluate and add neuromorphic model.
        
        Args:
            model: Neuromorphic model instance
            model_name: Name of the model
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            ModelPerformance object
        """
        print(f"Evaluating {model_name}...")
        
        perf = ModelPerformance(
            model_name=model_name,
            model_type='neuromorphic'
        )
        
        # Training
        start_time = time.time()
        model.train(X_train, y_train)
        perf.training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        perf.inference_time = time.time() - start_time
        
        # Metrics
        perf.accuracy = accuracy_score(y_test, y_pred)
        perf.precision = precision_score(y_test, y_pred, zero_division=0)
        perf.recall = recall_score(y_test, y_pred, zero_division=0)
        perf.f1_score = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            perf.roc_auc = roc_auc_score(y_test, y_pred)
        except:
            perf.roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            perf.true_negatives = int(cm[0, 0])
            perf.false_positives = int(cm[0, 1])
            perf.false_negatives = int(cm[1, 0])
            perf.true_positives = int(cm[1, 1])
        
        self.results.append(perf)
        return perf
        
    def add_traditional_model(self, model_class, model_name: str,
                            X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            **model_params) -> ModelPerformance:
        """
        Evaluate and add traditional ML model.
        
        Args:
            model_class: sklearn model class
            model_name: Name of the model
            X_train, y_train: Training data
            X_test, y_test: Test data
            **model_params: Parameters for model initialization
            
        Returns:
            ModelPerformance object
        """
        print(f"Evaluating {model_name}...")
        
        perf = ModelPerformance(
            model_name=model_name,
            model_type='traditional'
        )
        
        # Initialize model
        model = model_class(**model_params)
        
        # Training
        start_time = time.time()
        model.fit(X_train, y_train)
        perf.training_time = time.time() - start_time
        
        # Inference
        start_time = time.time()
        y_pred = model.predict(X_test)
        perf.inference_time = time.time() - start_time
        
        # Metrics
        perf.accuracy = accuracy_score(y_test, y_pred)
        perf.precision = precision_score(y_test, y_pred, zero_division=0)
        perf.recall = recall_score(y_test, y_pred, zero_division=0)
        perf.f1_score = f1_score(y_test, y_pred, zero_division=0)
        
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)[:, 1]
                perf.roc_auc = roc_auc_score(y_test, y_proba)
            else:
                perf.roc_auc = roc_auc_score(y_test, y_pred)
        except:
            perf.roc_auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2, 2):
            perf.true_negatives = int(cm[0, 0])
            perf.false_positives = int(cm[0, 1])
            perf.false_negatives = int(cm[1, 0])
            perf.true_positives = int(cm[1, 1])
        
        # Model size estimation
        try:
            if hasattr(model, 'tree_'):
                perf.n_parameters = model.tree_.node_count
            elif hasattr(model, 'coefs_'):
                perf.n_parameters = sum(c.size for c in model.coefs_)
        except:
            pass
        
        self.results.append(perf)
        return perf
        
    def compare_all(self) -> Dict[str, Any]:
        """
        Generate comprehensive comparison of all models.
        
        Returns:
            Dictionary with comparison results
        """
        if not self.results:
            return {}
        
        comparison = {
            'neuromorphic_models': [],
            'traditional_models': [],
            'summary': {}
        }
        
        # Separate by type
        for result in self.results:
            if result.model_type == 'neuromorphic':
                comparison['neuromorphic_models'].append(result.to_dict())
            else:
                comparison['traditional_models'].append(result.to_dict())
        
        # Best models by metric
        comparison['summary']['best_accuracy'] = self._find_best('accuracy')
        comparison['summary']['best_precision'] = self._find_best('precision')
        comparison['summary']['best_recall'] = self._find_best('recall')
        comparison['summary']['best_f1'] = self._find_best('f1_score')
        comparison['summary']['fastest_training'] = self._find_fastest('training_time')
        comparison['summary']['fastest_inference'] = self._find_fastest('inference_time')
        
        # Average metrics by type
        neuro_results = [r for r in self.results if r.model_type == 'neuromorphic']
        trad_results = [r for r in self.results if r.model_type == 'traditional']
        
        if neuro_results:
            comparison['summary']['neuromorphic_avg'] = {
                'accuracy': np.mean([r.accuracy for r in neuro_results]),
                'f1_score': np.mean([r.f1_score for r in neuro_results]),
                'training_time': np.mean([r.training_time for r in neuro_results]),
                'inference_time': np.mean([r.inference_time for r in neuro_results])
            }
        
        if trad_results:
            comparison['summary']['traditional_avg'] = {
                'accuracy': np.mean([r.accuracy for r in trad_results]),
                'f1_score': np.mean([r.f1_score for r in trad_results]),
                'training_time': np.mean([r.training_time for r in trad_results]),
                'inference_time': np.mean([r.inference_time for r in trad_results])
            }
        
        return comparison
    
    def _find_best(self, metric: str) -> Dict:
        """Find model with best performance on given metric."""
        best = max(self.results, key=lambda r: getattr(r, metric))
        return {
            'model_name': best.model_name,
            'model_type': best.model_type,
            'value': getattr(best, metric)
        }
    
    def _find_fastest(self, metric: str) -> Dict:
        """Find fastest model on given metric."""
        fastest = min(self.results, key=lambda r: getattr(r, metric))
        return {
            'model_name': fastest.model_name,
            'model_type': fastest.model_type,
            'value': getattr(fastest, metric)
        }
    
    def print_comparison_table(self):
        """Print formatted comparison table."""
        print("\n" + "="*100)
        print("MODEL COMPARISON RESULTS")
        print("="*100)
        
        # Header
        print(f"\n{'Model Name':<25} {'Type':<15} {'Acc':<8} {'Prec':<8} "
              f"{'Recall':<8} {'F1':<8} {'Train(s)':<10} {'Infer(s)':<10}")
        print("-"*100)
        
        # Results
        for result in self.results:
            print(f"{result.model_name:<25} {result.model_type:<15} "
                  f"{result.accuracy:<8.4f} {result.precision:<8.4f} "
                  f"{result.recall:<8.4f} {result.f1_score:<8.4f} "
                  f"{result.training_time:<10.4f} {result.inference_time:<10.6f}")
        
        print("-"*100)
        
        # Summary
        comparison = self.compare_all()
        
        if 'summary' in comparison:
            print("\n" + "="*100)
            print("SUMMARY")
            print("="*100)
            
            summary = comparison['summary']
            
            print(f"\n🏆 Best Accuracy:  {summary['best_accuracy']['model_name']} "
                  f"({summary['best_accuracy']['value']:.4f})")
            print(f"🏆 Best F1 Score:  {summary['best_f1']['model_name']} "
                  f"({summary['best_f1']['value']:.4f})")
            print(f"⚡ Fastest Train:  {summary['fastest_training']['model_name']} "
                  f"({summary['fastest_training']['value']:.4f}s)")
            print(f"⚡ Fastest Infer:  {summary['fastest_inference']['model_name']} "
                  f"({summary['fastest_inference']['value']:.6f}s)")
            
            if 'neuromorphic_avg' in summary:
                print(f"\n📊 Neuromorphic Average:")
                print(f"   Accuracy: {summary['neuromorphic_avg']['accuracy']:.4f}")
                print(f"   F1 Score: {summary['neuromorphic_avg']['f1_score']:.4f}")
                print(f"   Training Time: {summary['neuromorphic_avg']['training_time']:.4f}s")
                
            if 'traditional_avg' in summary:
                print(f"\n📊 Traditional Average:")
                print(f"   Accuracy: {summary['traditional_avg']['accuracy']:.4f}")
                print(f"   F1 Score: {summary['traditional_avg']['f1_score']:.4f}")
                print(f"   Training Time: {summary['traditional_avg']['training_time']:.4f}s")
        
        print("\n" + "="*100)
    
    def save_comparison(self, filepath: str):
        """Save comparison results to JSON."""
        comparison = self.compare_all()
        with open(filepath, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\nComparison saved to {filepath}")


class TraditionalModelBenchmark:
    """
    Benchmark suite for traditional ML models.
    
    Provides pre-configured traditional models for comparison.
    """
    
    @staticmethod
    def get_models() -> Dict[str, Tuple]:
        """
        Get dictionary of traditional models with parameters.
        
        Returns:
            Dictionary mapping model name to (class, params)
        """
        return {
            'Logistic Regression': (LogisticRegression, {
                'max_iter': 1000,
                'random_state': 42
            }),
            'Random Forest': (RandomForestClassifier, {
                'n_estimators': 100,
                'max_depth': 10,
                'random_state': 42
            }),
            'Gradient Boosting': (GradientBoostingClassifier, {
                'n_estimators': 100,
                'max_depth': 5,
                'random_state': 42
            }),
            'MLP Neural Network': (MLPClassifier, {
                'hidden_layer_sizes': (100, 50),
                'max_iter': 500,
                'random_state': 42
            }),
            'SVM': (SVC, {
                'kernel': 'rbf',
                'probability': True,
                'random_state': 42
            })
        }
    
    @staticmethod
    def benchmark_all(comparator: ModelComparator,
                     X_train: np.ndarray, y_train: np.ndarray,
                     X_test: np.ndarray, y_test: np.ndarray,
                     models_to_test: Optional[List[str]] = None):
        """
        Benchmark all traditional models.
        
        Args:
            comparator: ModelComparator instance
            X_train, y_train: Training data
            X_test, y_test: Test data
            models_to_test: List of model names to test (None = all)
        """
        models = TraditionalModelBenchmark.get_models()
        
        if models_to_test is not None:
            models = {k: v for k, v in models.items() if k in models_to_test}
        
        for model_name, (model_class, params) in models.items():
            try:
                comparator.add_traditional_model(
                    model_class, model_name,
                    X_train, y_train, X_test, y_test,
                    **params
                )
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")


# Example usage
if __name__ == "__main__":
    print("Model Comparison Framework")
    print("="*60)
    
    # Generate synthetic data
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.9, 0.1],  # Imbalanced like fraud detection
        random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"\nDataset: {len(X_train)} train, {len(X_test)} test samples")
    print(f"Fraud rate: {y_train.mean()*100:.2f}%")
    
    # Create comparator
    comparator = ModelComparator()
    
    # Benchmark traditional models
    print("\nBenchmarking traditional ML models...")
    TraditionalModelBenchmark.benchmark_all(
        comparator,
        X_train, y_train,
        X_test, y_test,
        models_to_test=['Logistic Regression', 'Random Forest', 'MLP Neural Network']
    )
    
    # Print comparison
    comparator.print_comparison_table()
    
    # Save results
    comparator.save_comparison('model_comparison.json')
