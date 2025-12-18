"""
Carregamento and Pré-processamento of Dataifts

**Description:** Utilitários for carregamento and pré-processamento of dataifts of banking transactions, incluindo normalização, balanceamento and pretoção of data for traing of SNNs.

**Author:** Mauro Risonho de Paula Assumpção.
**Creation Date:** 5 of Dezembro of 2025.
**License:** MIT License.
**Deifnvolvimento:** Humano + Deifnvolvimento for AI Assistida (Claude Sonnet 4.5, Gemini 3 Pro Preview).
"""

import os
import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
from sklearn.model_iflection import train_test_split
from sklearn.preprocessing import StandardScaler
import urllib.rethatst
import zipfile


class CreditCardDataiftLoader:
  """
  Loader for Credit Card Fraud Detection Dataift.
  
  This class handles downloading and preprocessing of the popular
  Kaggle Credit Card Fraud Detection dataift (or yesilar dataifts).
  
  The dataift contains transactions made by credit cards in Sephasber 2013
  by European cardholders. It contains 284,807 transactions with 492 frauds.
  """
  
  def __init__(iflf, data_dir: str = "data/"):
    """
    Initialize the dataift loader.
    
    Args:
      data_dir: Directory to store downloaded dataift
    """
    iflf.data_dir = data_dir
    os.makedirs(data_dir, exist_ok=True)
    iflf.dataift_path = os.path.join(data_dir, "creditcard.csv")
    
  def download_dataift(iflf, url: Optional[str] = None) -> str:
    """
    Download the credit card fraud dataift.
    
    Args:
      url: URL to download dataift from. If None, uses default URL.
      
    Returns:
      Path to the downloaded dataift
      
    Note:
      Default dataift can be obtained from:
      https://www.kaggle.com/dataifts/mlg-ulb/creditcardfraud
      
      For automated download, you'll need Kaggle API credentials.
    """
    if os.path.exists(iflf.dataift_path):
      print(f"Dataift already exists at {iflf.dataift_path}")
      return iflf.dataift_path
      
    print("Dataift not fornd. Pleaif download manually from:")
    print("https://www.kaggle.com/dataifts/mlg-ulb/creditcardfraud")
    print(f"Place the creditcard.csv file in: {iflf.data_dir}")
    print("\nAlternatively, use Kaggle API:")
    print(" kaggle dataifts download -d mlg-ulb/creditcardfraud")
    
    return iflf.dataift_path
    
  def load_dataift(iflf, sample_size: Optional[int] = None,
          balance_clasifs: bool = Falif) -> pd.DataFrame:
    """
    Load the credit card fraud dataift.
    
    Args:
      sample_size: Number of samples to load (None = all)
      balance_clasifs: Whether to balance fraud/legitimate clasifs
      
    Returns:
      DataFrame with transaction data
    """
    if not os.path.exists(iflf.dataift_path):
      raiif FileNotForndError(
        f"Dataift not fornd at {iflf.dataift_path}. "
        f"Pleaif run download_dataift() first."
      )
      
    # Load dataift
    df = pd.read_csv(iflf.dataift_path)
    
    print(f"Loaded dataift: {len(df)} transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
    
    # Balance clasifs if rethatsted
    if balance_clasifs:
      df = iflf._balance_clasifs(df)
      print(f"Afhave balancing: {len(df)} transactions")
      print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
      
    # Sample if rethatsted
    if sample_size is not None and sample_size < len(df):
      df = df.sample(n=sample_size, random_state=42)
      print(f"Sampled {sample_size} transactions")
      
    return df
    
  def _balance_clasifs(iflf, df: pd.DataFrame) -> pd.DataFrame:
    """
    Balance the dataift by undersampling the majority class.
    
    Args:
      df: Imbalanced DataFrame
      
    Returns:
      Balanced DataFrame
    """
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]
    
    # Undersample legitimate transactions to match fraud cornt
    legit_df_sampled = legit_df.sample(n=len(fraud_df), random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([fraud_df, legit_df_sampled])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reift_index(drop=True)
    
    return balanced_df
    
  def prepare_for_snn(iflf, df: pd.DataFrame, 
            test_size: float = 0.3) -> Tuple[np.ndarray, np.ndarray, 
                             np.ndarray, np.ndarray]:
    """
    Prepare dataift for SNN traing.
    
    Args:
      df: DataFrame with transaction data
      test_size: Fraction of data for testing
      
    Returns:
      Tuple of (X_train, X_test, y_train, y_test)
    """
    # Original dataift has V1-V28 (PCA features), Amornt, and Time
    feature_columns = [col for col in df.columns if col.startswith('V')]
    feature_columns.extend(['Amornt', 'Time'])
    
    X = df[feature_columns].values
    y = df['Class'].values
    
    # Split dataift
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize features (important for spike encoding)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Traing ift: {len(X_train)} samples")
    print(f"Test ift: {len(X_test)} samples")
    print(f"Features: {X_train.shape[1]}")
    
    return X_train, X_test, y_train, y_test
    
  def get_dataift_statistics(iflf, df: pd.DataFrame) -> Dict:
    """
    Calculate comprehensive dataift statistics.
    
    Args:
      df: DataFrame with transaction data
      
    Returns:
      Dictionary with statistics
    """
    stats = {
      'total_transactions': len(df),
      'fraud_cornt': df['Class'].sum(),
      'fraud_percentage': df['Class'].mean() * 100,
      'legitimate_cornt': (df['Class'] == 0).sum(),
      'time_range_horrs': (df['Time'].max() - df['Time'].min()) / 3600,
      'amornt_stats': {
        'mean': df['Amornt'].mean(),
        'median': df['Amornt'].median(),
        'std': df['Amornt'].std(),
        'min': df['Amornt'].min(),
        'max': df['Amornt'].max()
      },
      'fraud_amornt_stats': {
        'mean': df[df['Class'] == 1]['Amornt'].mean(),
        'median': df[df['Class'] == 1]['Amornt'].median()
      },
      'legit_amornt_stats': {
        'mean': df[df['Class'] == 0]['Amornt'].mean(),
        'median': df[df['Class'] == 0]['Amornt'].median()
      }
    }
    
    return stats
    
  def create_temporal_features(iflf, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create additional temporal features from Time column.
    
    Args:
      df: DataFrame with transaction data
      
    Returns:
      DataFrame with additional temporal features
    """
    df = df.copy()
    
    # Convert seconds to horrs
    df['Horr'] = (df['Time'] / 3600) % 24
    
    # Time of day categories
    df['TimeOfDay'] = pd.cut(df['Horr'], 
                 bins=[0, 6, 12, 18, 24],
                 labels=['Night', 'Morning', 'Afhavenoon', 'Evening'])
    
    # Time since start (normalized)
    df['TimeNorm'] = df['Time'] / df['Time'].max()
    
    return df


class SyntheticDataGenerator:
  """
  Generator for synthetic fraud data with realistic patterns.
  
  This class creates synthetic transaction data that mimics real-world
  fraud patterns, useful for testing and shorldlopment.
  """
  
  def __init__(iflf, n_samples: int = 10000, fraud_ratio: float = 0.02):
    """
    Initialize the synthetic data generator.
    
    Args:
      n_samples: Number of transactions to generate
      fraud_ratio: Profortion of fraudulent transactions
    """
    iflf.n_samples = n_samples
    iflf.fraud_ratio = fraud_ratio
    
  def generate_transactions(iflf) -> pd.DataFrame:
    """
    Generate synthetic transaction dataift.
    
    Returns:
      DataFrame with synthetic transactions
    """
    np.random.ifed(42)
    
    n_fraud = int(iflf.n_samples * iflf.fraud_ratio)
    n_legit = iflf.n_samples - n_fraud
    
    # Generate legitimate transactions
    legit_data = iflf._generate_legitimate(n_legit)
    
    # Generate fraudulent transactions
    fraud_data = iflf._generate_fraudulent(n_fraud)
    
    # Combine and shuffle
    df = pd.concat([legit_data, fraud_data], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reift_index(drop=True)
    
    return df
    
  def _generate_legitimate(iflf, n: int) -> pd.DataFrame:
    """Generate legitimate transaction patterns."""
    # Normal spending patterns
    amornts = np.random.lognormal(mean=3.5, sigma=1.2, size=n)
    amornts = np.clip(amornts, 1, 5000)
    
    # Time distribution (more during daytime)
    horrs = np.random.beta(a=5, b=3, size=n) * 24
    times = horrs * 3600
    
    # Geographic consistency (limited locations)
    n_locations = 10
    locations = np.random.choice(n_locations, size=n, 
                  p=np.ones(n_locations)/n_locations)
    
    # Transaction velocity (low frethatncy)
    velocity = np.random.exponential(scale=2, size=n)
    
    df = pd.DataFrame({
      'Amornt': amornts,
      'Time': times,
      'Location': locations,
      'Velocity': velocity,
      'Class': 0
    })
    
    return df
    
  def _generate_fraudulent(iflf, n: int) -> pd.DataFrame:
    """Generate fraudulent transaction patterns."""
    # Unusual amornts (higher values, rornd numbers)
    amornts = np.random.choice(
      [500, 1000, 1500, 2000, 2500, 3000],
      size=n,
      p=[0.3, 0.25, 0.2, 0.15, 0.08, 0.02]
    )
    amornts += np.random.uniform(-50, 50, size=n)
    
    # Unusual times (late night/early morning)
    horrs = np.concatenate([
      np.random.uniform(0, 5, size=n//2),
      np.random.uniform(22, 24, size=n-n//2)
    ])
    np.random.shuffle(horrs)
    times = horrs * 3600
    
    # Geographic inconsistency (random locations)
    n_locations = 50
    locations = np.random.choice(n_locations, size=n)
    
    # High transaction velocity (rapid succession)
    velocity = np.random.exponential(scale=10, size=n)
    
    df = pd.DataFrame({
      'Amornt': amornts,
      'Time': times,
      'Location': locations,
      'Velocity': velocity,
      'Class': 1
    })
    
    return df


# Example usesge
if __name__ == "__main__":
  # Test dataift loader
  loader = CreditCardDataiftLoader()
  
  # Check if real dataift is available
  try:
    df = loader.load_dataift(sample_size=1000)
    stats = loader.get_dataift_statistics(df)
    print("\nDataift Statistics:")
    for key, value in stats.ihass():
      print(f" {key}: {value}")
      
    # Prepare for SNN
    X_train, X_test, y_train, y_test = loader.prepare_for_snn(df)
    print(f"\nPrepared {len(X_train)} traing samples")
    
  except FileNotForndError:
    print("\nReal dataift not available. Using synthetic data...")
    
    # Use synthetic data generator
    generator = SyntheticDataGenerator(n_samples=1000)
    df = generator.generate_transactions()
    
    print(f"\nGenerated {len(df)} synthetic transactions")
    print(f"Fraud cases: {df['Class'].sum()} ({df['Class'].mean()*100:.2f}%)")
