"""
**Descrição:** Métodos de prevenção de overfitting e regularização.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Track training metrics"""
    train_loss: List[float]
    val_loss: List[float]
    train_acc: List[float]
    val_acc: List[float]
    epoch: List[int]


class DataAugmenter:
    """
    Data augmentation for fraud detection
    
    Técnicas:
    1. Gaussian noise injection
    2. Feature scaling perturbation
    3. SMOTE (Synthetic Minority Over-sampling)
    4. Mixup (linear interpolation)
    
    Benefit: 10x virtual dataset size without real data collection
    """
    
    def __init__(self, noise_level: float = 0.01):
        self.noise_level = noise_level
    
    def add_gaussian_noise(
        self,
        X: torch.Tensor,
        noise_factor: float = 0.01
    ) -> torch.Tensor:
        """
        Add Gaussian noise to features
        
        Example: amount=100 → amount=100±1
        """
        noise = torch.randn_like(X) * noise_factor
        return X + noise
    
    def random_scaling(
        self,
        X: torch.Tensor,
        scale_range: Tuple[float, float] = (0.95, 1.05)
    ) -> torch.Tensor:
        """
        Random feature scaling
        
        Simulates: Natural variation in transaction values
        """
        scale = torch.FloatTensor(X.shape[1]).uniform_(*scale_range)
        return X * scale
    
    def smote(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        k_neighbors: int = 5,
        sampling_ratio: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SMOTE: Synthetic Minority Over-sampling Technique with progress tracking
        
        Method:
        1. Find k nearest neighbors of minority class
        2. Interpolate between sample and neighbor
        3. Create synthetic samples
        
        Best for: Imbalanced datasets (fraud: 3.5%)
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Separate classes
        fraud_mask = (y == 1)
        X_fraud = X[fraud_mask]
        y_fraud = y[fraud_mask]
        
        if len(X_fraud) == 0:
            return X, y
        
        # Find nearest neighbors
        knn = NearestNeighbors(n_neighbors=k_neighbors + 1)
        knn.fit(X_fraud.numpy())
        
        # Generate synthetic samples
        n_synthetic = int(len(X_fraud) * sampling_ratio)
        X_synthetic = []
        
        # Progress bar for synthetic sample generation
        for _ in tqdm(range(n_synthetic), 
                     desc="🧬 Gerando amostras sintéticas (SMOTE)", 
                     unit="sample",
                     leave=False):
            # Random sample
            idx = np.random.randint(0, len(X_fraud))
            sample = X_fraud[idx]
            
            # Find neighbors
            distances, indices = knn.kneighbors(sample.unsqueeze(0).numpy())
            neighbor_idx = np.random.choice(indices[0][1:])  # Exclude self
            neighbor = X_fraud[neighbor_idx]
            
            # Interpolate
            alpha = np.random.uniform(0, 1)
            synthetic = sample + alpha * (neighbor - sample)
            X_synthetic.append(synthetic)
        
        # Combine
        X_synthetic = torch.stack(X_synthetic)
        y_synthetic = torch.ones(len(X_synthetic), dtype=torch.long)
        
        X_augmented = torch.cat([X, X_synthetic], dim=0)
        y_augmented = torch.cat([y, y_synthetic], dim=0)
        
        logger.info(f"SMOTE: {len(X)} → {len(X_augmented)} samples (+{len(X_synthetic)} synthetic)")
        
        return X_augmented, y_augmented
    
    def mixup(
        self,
        X1: torch.Tensor,
        X2: torch.Tensor,
        y1: torch.Tensor,
        y2: torch.Tensor,
        alpha: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mixup: Linear interpolation between samples
        
        Formula:
            X_mixed = alpha * X1 + (1 - alpha) * X2
            y_mixed = alpha * y1 + (1 - alpha) * y2
        """
        X_mixed = alpha * X1 + (1 - alpha) * X2
        y_mixed = alpha * y1.float() + (1 - alpha) * y2.float()
        
        return X_mixed, y_mixed.long()


class RegularizedSNN(nn.Module):
    """
    SNN with L1/L2 regularization and dropout
    
    Regularization:
    - L1 (Lasso): Sparse weights (feature selection)
    - L2 (Ridge): Small weights (generalization)
    - Dropout: Random neuron deactivation
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        dropout_rate: float = 0.3,
        l1_lambda: float = 0.001,
        l2_lambda: float = 0.01
    ):
        super().__init__()
        
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        
        # Build network
        layers = []
        layer_sizes = [input_size] + hidden_sizes + [output_size]
        
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            
            # Add dropout (except last layer)
            if i < len(layer_sizes) - 2:
                layers.append(nn.Dropout(dropout_rate))
            
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
    
    def regularization_loss(self) -> torch.Tensor:
        """
        Calculate L1 + L2 regularization loss
        
        Formula:
            L_reg = λ1 * Σ|w| + λ2 * Σw²
        """
        l1_loss = 0.0
        l2_loss = 0.0
        
        for param in self.parameters():
            l1_loss += torch.sum(torch.abs(param))
            l2_loss += torch.sum(param ** 2)
        
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss
    
    def total_loss(self, pred_loss: torch.Tensor) -> torch.Tensor:
        """
        Total loss = Prediction loss + Regularization loss
        """
        return pred_loss + self.regularization_loss()


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    
    Strategy:
    - Monitor validation loss
    - Stop if no improvement for N epochs (patience)
    - Restore best model weights
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.best_loss = float('inf') if mode == 'min' else float('-inf')
        self.best_weights = None
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if should stop training
        
        Returns:
            True if should stop, False otherwise
        """
        if self.mode == 'min':
            improved = val_loss < (self.best_loss - self.min_delta)
        else:
            improved = val_loss > (self.best_loss + self.min_delta)
        
        if improved:
            # Improvement
            self.best_loss = val_loss
            self.best_weights = model.state_dict().copy()
            self.counter = 0
            logger.info(f"Validation loss improved to {val_loss:.4f}")
        else:
            # No improvement
            self.counter += 1
            logger.info(f"No improvement ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                logger.info("Early stopping triggered!")
                
                # Restore best weights
                if self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    logger.info("Restored best model weights")
        
        return self.early_stop


class CrossValidator:
    """
    K-Fold Cross-Validation
    
    Benefits:
    - More reliable performance estimates
    - Detect overfitting
    - Better use of limited data
    """
    
    def __init__(self, n_folds: int = 5):
        self.n_folds = n_folds
    
    def split(
        self,
        X: torch.Tensor,
        y: torch.Tensor
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Create K-fold splits
        
        Returns:
            List of (X_train, X_val, y_train, y_val) tuples
        """
        from sklearn.model_selection import KFold
        
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        splits = []
        for train_idx, val_idx in kf.split(X):
            X_train = X[train_idx]
            X_val = X[val_idx]
            y_train = y[train_idx]
            y_val = y[val_idx]
            
            splits.append((X_train, X_val, y_train, y_val))
        
        return splits
    
    def evaluate(
        self,
        model_fn,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 50
    ) -> Dict[str, float]:
        """
        Perform K-fold cross-validation with progress tracking
        
        Returns:
            {
                'mean_val_acc': float,
                'std_val_acc': float,
                'fold_scores': List[float]
            }
        """
        splits = self.split(X, y)
        fold_scores = []
        
        # Progress bar for folds
        for fold_idx, (X_train, X_val, y_train, y_val) in tqdm(enumerate(splits),
                                                                 total=len(splits),
                                                                 desc="📊 Cross-validation",
                                                                 unit="fold"):
            logger.info(f"Fold {fold_idx + 1}/{self.n_folds}")
            
            # Create fresh model
            model = model_fn()
            
            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # Progress bar for epochs
            for epoch in tqdm(range(epochs), 
                            desc=f"  Treinando fold {fold_idx + 1}", 
                            unit="epoch",
                            leave=False):
                model.train()
                optimizer.zero_grad()
                
                output = model(X_train)
                loss = criterion(output, y_train)
                
                if hasattr(model, 'regularization_loss'):
                    loss = model.total_loss(loss)
                
                loss.backward()
                optimizer.step()
            
            # Evaluate
            model.eval()
            with torch.no_grad():
                output = model(X_val)
                predictions = torch.argmax(output, dim=1)
                accuracy = (predictions == y_val).float().mean().item()
            
            fold_scores.append(accuracy)
            logger.info(f"Fold {fold_idx + 1} accuracy: {accuracy:.4f}")
        
        return {
            'mean_val_acc': np.mean(fold_scores),
            'std_val_acc': np.std(fold_scores),
            'fold_scores': fold_scores
        }


class OverfittingDetector:
    """
    Detect overfitting from training curves
    """
    
    @staticmethod
    def detect(history: TrainingHistory) -> Dict[str, any]:
        """
        Analyze training history for overfitting
        
        Indicators:
        - Train loss ↓, Val loss ↑
        - Train acc ↑, Val acc stagnant
        - Large gap between train/val metrics
        """
        # Last 5 epochs
        recent_train_loss = np.mean(history.train_loss[-5:])
        recent_val_loss = np.mean(history.val_loss[-5:])
        recent_train_acc = np.mean(history.train_acc[-5:])
        recent_val_acc = np.mean(history.val_acc[-5:])
        
        # Gap analysis
        loss_gap = recent_val_loss - recent_train_loss
        acc_gap = recent_train_acc - recent_val_acc
        
        # Overfitting score
        overfitting_score = (loss_gap / recent_train_loss + acc_gap) / 2
        
        is_overfitting = overfitting_score > 0.15
        
        return {
            'is_overfitting': is_overfitting,
            'overfitting_score': overfitting_score,
            'loss_gap': loss_gap,
            'acc_gap': acc_gap,
            'recommendations': [
                "Add more data (Kaggle dataset)" if is_overfitting else None,
                "Increase L1/L2 regularization" if loss_gap > 0.2 else None,
                "Add dropout layers" if acc_gap > 0.1 else None,
                "Reduce model complexity" if overfitting_score > 0.3 else None
            ]
        }
    
    @staticmethod
    def plot(history: TrainingHistory, save_path: Optional[str] = None):
        """
        Plot training curves
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax = axes[0]
        ax.plot(history.epoch, history.train_loss, label='Train Loss', marker='o')
        ax.plot(history.epoch, history.val_loss, label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training vs Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy
        ax = axes[1]
        ax.plot(history.epoch, history.train_acc, label='Train Acc', marker='o')
        ax.plot(history.epoch, history.val_acc, label='Val Acc', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training vs Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Demo
    print("Overfitting Prevention Module")
    print("-" * 60)
    
    # 1. Data Augmentation
    print("\n1. Data Augmentation (SMOTE)")
    augmenter = DataAugmenter()
    
    X = torch.randn(100, 64)
    y = torch.cat([torch.zeros(90), torch.ones(10)])  # 10% fraud
    
    X_aug, y_aug = augmenter.smote(X, y, sampling_ratio=2.0)
    print(f"Original: {len(X)} samples")
    print(f"Augmented: {len(X_aug)} samples")
    print(f"Fraud ratio: {(y_aug == 1).sum().item() / len(y_aug) * 100:.1f}%")
    
    # 2. Regularized Model
    print("\n2. Regularized SNN")
    model = RegularizedSNN(
        input_size=64,
        hidden_sizes=[32, 16],
        output_size=2,
        dropout_rate=0.3,
        l1_lambda=0.001,
        l2_lambda=0.01
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Regularization loss: {model.regularization_loss():.4f}")
    
    # 3. Early Stopping
    print("\n3. Early Stopping")
    early_stopping = EarlyStopping(patience=5)
    
    print("Simulating training...")
    for epoch in range(20):
        val_loss = 0.5 - 0.02 * epoch + np.random.randn() * 0.01
        should_stop = early_stopping(val_loss, model)
        
        if should_stop:
            print(f"Stopped at epoch {epoch}")
            break
