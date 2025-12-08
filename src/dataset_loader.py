"""
Dataset loading and preprocessing utilities.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
Date: December 2025
License: MIT
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.request
import zipfile


class CreditCardDatasetLoader:
    """
    Loader for Credit Card Fraud Detection Dataset.
    
    This class handles downloading and preprocessing of the popular
    Kaggle Credit Card Fraud Detection dataset (or similar datasets).
    
    The dataset contains transactions made by credit cards in September 2013
    by European cardholders. It contains 284,807 transactions with 492 frauds.
    """
    
    def __init__(self, data_dir: str = "data/"):
        """
        Initialize the dataset loader.
        
        Args:
            data_dir: Directory to store downloaded dataset
        """
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.dataset_path = os.path.join(data_dir, "creditcard.csv")
        
    def download_dataset(self, url: Optional[str] = None) -> str:
        """
        Download the credit card fraud dataset.
        
        Args:
            url: URL to download dataset from. If None, uses default URL.
            
        Returns:
            Path to the downloaded dataset
            
        Note:
            Default dataset can be obtained from:
            https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
            
            For automated download, you'll need Kaggle API credentials.
        """
        if os.path.exists(self.dataset_path):
            print(f"Dataset already exists at {self.dataset_path}")
            return self.dataset_path
            
        print("Dataset not found. Please download manually from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print(f"Place the creditcard.csv file in: {self.data_dir}")
        print("\nAlternatively, use Kaggle API:")
        print("  kaggle datasets download -d mlg-ulb/creditcardfraud")
        
        return self.dataset_path
        
    def load_dataset(self, sample_size: Optional[int] = None,
                    balance_classes: bool = False) -> pd.DataFrame:
        """
        Load the credit card fraud dataset.
        
        Args:
            sample_size: Number of samples to load (None = all)
            balance_classes: Whether to balance fraud/legitimate classes
            
        Returns:
            DataFrame with transaction data
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.dataset_path}. "
                f"Please run download_dataset() first."
            )
            
        # Load dataset
        df = pd.read_csv(self.dataset_path)
        
        print(f"Loaded dataset: {len(df)} transactions")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
        
        # Balance classes if requested
        if balance_classes:
            df = self._balance_classes(df)
            print(f"After balancing: {len(df)} transactions")
            print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
            
        # Sample if requested
        if sample_size is not None and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            print(f"Sampled {sample_size} transactions")
            
        return df
        
    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance the dataset by undersampling the majority class.
        
        Args:
            df: Imbalanced DataFrame
            
        Returns:
            Balanced DataFrame
        """
        fraud_df = df[df['Class'] == 1]
        legit_df = df[df['Class'] == 0]
        
        # Undersample legitimate transactions to match fraud count
        legit_df_sampled = legit_df.sample(n=len(fraud_df), random_state=42)
        
        # Combine and shuffle
        balanced_df = pd.concat([fraud_df, legit_df_sampled])
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return balanced_df
        
    def prepare_for_snn(self, df: pd.DataFrame, 
                       test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, 
                                                         np.ndarray, np.ndarray]:
        """
        Prepare dataset for SNN training.
        
        Args:
            df: DataFrame with transaction data
            test_size: Fraction of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Original dataset has V1-V28 (PCA features), Amount, and Time
        feature_columns = [col for col in df.columns if col.startswith('V')]
        feature_columns.extend(['Amount', 'Time'])
        
        X = df[feature_columns].values
        y = df['Class'].values
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Normalize features (important for spike encoding)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
        
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive dataset statistics.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Dictionary with statistics
        """
        stats = {
            'total_transactions': len(df),
            'fraud_count': df['Class'].sum(),
            'fraud_percentage': df['Class'].mean() * 100,
            'legitimate_count': (df['Class'] == 0).sum(),
            'time_range_hours': (df['Time'].max() - df['Time'].min()) / 3600,
            'amount_stats': {
                'mean': df['Amount'].mean(),
                'median': df['Amount'].median(),
                'std': df['Amount'].std(),
                'min': df['Amount'].min(),
                'max': df['Amount'].max()
            },
            'fraud_amount_stats': {
                'mean': df[df['Class'] == 1]['Amount'].mean(),
                'median': df[df['Class'] == 1]['Amount'].median()
            },
            'legit_amount_stats': {
                'mean': df[df['Class'] == 0]['Amount'].mean(),
                'median': df[df['Class'] == 0]['Amount'].median()
            }
        }
        
        return stats
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional temporal features from Time column.
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            DataFrame with additional temporal features
        """
        df = df.copy()
        
        # Convert seconds to hours
        df['Hour'] = (df['Time'] / 3600) % 24
        
        # Time of day categories
        df['TimeOfDay'] = pd.cut(df['Hour'], 
                                 bins=[0, 6, 12, 18, 24],
                                 labels=['Night', 'Morning', 'Afternoon', 'Evening'])
        
        # Time since start (normalized)
        df['TimeNorm'] = df['Time'] / df['Time'].max()
        
        return df


class SyntheticDataGenerator:
    """
    Generator for synthetic fraud data with realistic patterns.
    
    This class creates synthetic transaction data that mimics real-world
    fraud patterns, useful for testing and development.
    """
    
    def __init__(self, n_samples: int = 10000, fraud_ratio: float = 0.02):
        """
        Initialize the synthetic data generator.
        
        Args:
            n_samples: Number of transactions to generate
            fraud_ratio: Proportion of fraudulent transactions
        """
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        
    def generate_transactions(self) -> pd.DataFrame:
        """
        Generate synthetic transaction dataset.
        
        Returns:
            DataFrame with synthetic transactions
        """
        np.random.seed(42)
        
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_legit = self.n_samples - n_fraud
        
        # Generate legitimate transactions
        legit_data = self._generate_legitimate(n_legit)
        
        # Generate fraudulent transactions
        fraud_data = self._generate_fraudulent(n_fraud)
        
        # Combine and shuffle
        df = pd.concat([legit_data, fraud_data], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
        
    def _generate_legitimate(self, n: int) -> pd.DataFrame:
        """Generate legitimate transaction patterns."""
        # Normal spending patterns
        amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n)
        amounts = np.clip(amounts, 1, 5000)
        
        # Time distribution (more during daytime)
        hours = np.random.beta(a=5, b=3, size=n) * 24
        times = hours * 3600
        
        # Geographic consistency (limited locations)
        n_locations = 10
        locations = np.random.choice(n_locations, size=n, 
                                    p=np.ones(n_locations)/n_locations)
        
        # Transaction velocity (low frequency)
        velocity = np.random.exponential(scale=2, size=n)
        
        df = pd.DataFrame({
            'Amount': amounts,
            'Time': times,
            'Location': locations,
            'Velocity': velocity,
            'Class': 0
        })
        
        return df
        
    def _generate_fraudulent(self, n: int) -> pd.DataFrame:
        """Generate fraudulent transaction patterns."""
        # Unusual amounts (higher values, round numbers)
        amounts = np.random.choice(
            [500, 1000, 1500, 2000, 2500, 3000],
            size=n,
            p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]
        )
        amounts += np.random.uniform(-50, 50, size=n)
        
        # Unusual times (late night/early morning)
        hours = np.concatenate([
            np.random.uniform(0, 5, size=n//2),
            np.random.uniform(22, 24, size=n-n//2)
        ])
        np.random.shuffle(hours)
        times = hours * 3600
        
        # Geographic inconsistency (random locations)
        n_locations = 50
        locations = np.random.choice(n_locations, size=n)
        
        # High transaction velocity (rapid succession)
        velocity = np.random.exponential(scale=10, size=n)
        
        df = pd.DataFrame({
            'Amount': amounts,
            'Time': times,
            'Location': locations,
            'Velocity': velocity,
            'Class': 1
        })
        
        return df


# Example usage
if __name__ == "__main__":
    # Test dataset loader
    loader = CreditCardDatasetLoader()
    
    # Check if real dataset is available
    try:
        df = loader.load_dataset(sample_size=1000)
        stats = loader.get_dataset_statistics(df)
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
            
        # Prepare for SNN
        X_train, X_test, y_train, y_test = loader.prepare_for_snn(df)
        print(f"\nPrepared {len(X_train)} training samples")
        
    except FileNotFoundError:
        print("\nReal dataset not available. Using synthetic data...")
        
        # Use synthetic data generator
        generator = SyntheticDataGenerator(n_samples=1000)
        df = generator.generate_transactions()
        
        print(f"\nGenerated {len(df)} synthetic transactions")
        print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
