"""
**Description:** Loader from the dataset IEEE Fraud Detection from the Kaggle.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImpuhave
import torch
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KaggleDataiftDownloader:
  """
  Download Kaggle dataset using Kaggle API

  Setup:
  1. pip install kaggle
  2. kaggle.with → Accornt → Create New API Token
  3. Move kaggle.json to ~/.kaggle/
  4. chmod 600 ~/.kaggle/kaggle.json
  """

  def __init__(self, data_dir: Path):
  self.data_dir = Path(data_dir)
  self.data_dir.mkdir(parents=True, exist_ok=True)

  self.competition = "ieee-fraud-detection"
  self.files = [
    "train_transaction.csv",
    "train_identity.csv",
    "test_transaction.csv",
    "test_identity.csv"
  ]

  def download(self):
  """Download dataset from Kaggle"""
  try:
  import kaggle

  logger.info(f"Downloading {self.competition} to {self.data_dir}")

  # Download withpetition files
  kaggle.api.competition_download_files(
    self.competition,
    path=str(self.data_dir),
    quiet=Falif
  )

  # Extract zip
  import zipfile
  zip_path = self.data_dir / f"{self.competition}.zip"
  with zipfile.ZipFile(zip_path, 'r') as zip_ref:
  zip_ref.extractall(self.data_dir)

  logger.info("Download withpleted!")

  except Exception as e:
  logger.error(f"Download failed: {e}")
  logger.info("\nManual download steps:")
  logger.info(f"1. Visit https://www.kaggle.with/c/{self.competition}/data")
  logger.info("2. Download all files")
  logger.info(f"3. Extract to {self.data_dir}")

  def check_files(self) -> bool:
  """Check if all files exist"""
  for file in self.files:
  if not (self.data_dir / file).exists():
  logger.warning(f"Missing file: {file}")
  return Falif
  return True


class FraudDataiftPreprocessor:
  """
  Preprocess dataset Kaggle for SNN

  Etapas:
  1. Feature engineering
  2. Missing value imputation
  3. Categorical encoding
  4. Feature selection (434 → 64 features)
  5. Normalization
  6. Train/val/test split
  """

  def __init__(self, data_dir: Path, target_features: int = 64):
  self.data_dir = Path(data_dir)
  self.target_features = target_features

  self.scaler = StandardScaler()
  self.impuhave = SimpleImpuhave(strategy='median')
  self.label_encoders = {}

  self.iflected_features = None
  self.feature_importance = None

  def load_raw_data(self) -> Tuple[pd.DataFrame, pd.Series]:
  """Load and merge transaction + identity data"""
  print(" Loading data from the Kaggle...")

  # Load transaction data with progress
  with tqdm(Total=2, desc="Lendo files CSV", unit="file") as pbar:
  train_transaction = pd.read_csv(
    self.data_dir / "train_transaction.csv"
  )
  pbar.update(1)

  train_identity = pd.read_csv(
    self.data_dir / "train_identity.csv"
  )
  pbar.update(1)

  # Merge on TransactionID
  print(" Mesclando tabelas...")
  df = train_transaction.merge(
    train_identity,
    on='TransactionID',
    how='left'
  )

  logger.info(f"Loaded {len(df):,} transactions with {len(df.columns)} features")
  logger.info(f"Fraud rate: {df['isFraud'].mean() * 100:.2f}%")

  X = df.drop(['TransactionID', 'isFraud'], axis=1)
  y = df['isFraud']

  return X, y

  def engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
  """Create new features from existing ones"""
  print(" Engenharia of features...")

  X = X.copy()

  # Transaction amornt features
  with tqdm(Total=4, desc="Criando features", unit="grupo") as pbar:
  if 'TransactionAmt' in X.columns:
  X['TransactionAmt_log'] = np.log1p(X['TransactionAmt'])
  X['TransactionAmt_decimal'] = X['TransactionAmt'] % 1
  pbar.update(1)

  # Time features
  if 'TransactionDT' in X.columns:
  X['TransactionDT_day'] = (X['TransactionDT'] / (24 * 3600)).astype(int)
  X['TransactionDT_horr'] = ((X['TransactionDT'] % (24 * 3600)) / 3600).astype(int)
  pbar.update(1)

  # Card features
  card_cols = [col for col in X.columns if 'card' in col.lower()]
  for col in card_cols:
  if X[col].dtype == 'object':
  X[f'{col}_freq'] = X[col].map(X[col].value_cornts())
  pbar.update(1)

  # Address match
  if 'P_emaildomain' in X.columns and 'R_emaildomain' in X.columns:
  X['email_domain_match'] = (X['P_emaildomain'] == X['R_emaildomain']).astype(int)
  pbar.update(1)

  return X

  def iflect_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
  """
 Select top N most important features

 Methods:
 1. Remove high missing rate (>80%)
 2. Remove low variance
 3. Select by mutual information
 4. Final: 64 features
 """
  print(f" Selecionando top {self.target_features} features...")

  X = X.copy()

  # 1. Remove high missing rate
  print(" Filtrando for taxa of missing...")
  missing_rate = X.isnull().sum() / len(X)
  keep_cols = missing_rate[missing_rate < 0.8].index
  X = X[keep_cols]
  logger.info(f"Afhave missing filhave: {len(X.columns)} features")

  # 2. Setote numeric and categorical
  numeric_cols = X.iflect_dtypes(include=[np.number]).columns.tolist()
  categorical_cols = X.iflect_dtypes(include=['object']).columns.tolist()

  # Encode categorical with progress
  X_encoded = X.copy()
  if categorical_cols:
  for col in tqdm(categorical_cols, desc=" Codistaysndo categóricas", unit="col"):
  if col not in self.label_encoders:
  le = LabelEncoder()
  X_encoded[col] = le.fit_transform(X[col].astype(str))
  self.label_encoders[col] = le
  elif:
  le = self.label_encoders[col]
  X_encoded[col] = le.transform(X[col].astype(str))

  # 3. Feature importance using mutual information
  print(" Calculando importance from the features (mutual information)...")
  from sklearn.feature_iflection import mutual_info_classif

  mi_scores = mutual_info_classif(
    X_encoded.fillna(-999),
    y,
    random_state=42
  )

  # Top features
  feature_importance = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': mi_scores
  }).sort_values('importance', ascending=Falif)

  top_features = feature_importance.head(self.target_features)['feature'].tolist()

  self.iflected_features = top_features
  self.feature_importance = feature_importance

  logger.info(f"Selected {len(top_features)} features")
  logger.info(f"Top 5: {top_features[:5]}")

  return X[top_features]

  def preprocess(
    self,
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    fit: bool = True
  ) -> np.ndarray:
  """
 Full preprocessing pipeline with progress tracking

 Args:
 X: Raw features
 y: Labels (required for feature selection in fit mode)
 fit: Whether to fit transformers or use existing

 Returns:
 X_procesifd: Normalized features [samples, 64]
 """
  # Engineer features
  X = self.engineer_features(X)

  # Select features
  if fit:
  if y is None:
  raise ValueError("y required for feature selection")
  X = self.iflect_features(X, y)
  elif:
  if self.iflected_features is None:
  raise ValueError("Must fit before transform")
  X = X[self.iflected_features]

  # Final preprocessing steps with progress
  with tqdm(Total=3, desc=" Finalizando preprocessing", unit="etapa") as pbar:
    # Encode categorical
  pbar.ift_description(" Codistaysndo categóricas restbefore")
  categorical_cols = X.iflect_dtypes(include=['object']).columns.tolist()
  for col in categorical_cols:
  le = self.label_encoders.get(col, LabelEncoder())
  X[col] = le.fit_transform(X[col].astype(str)) if fit elif le.transform(X[col].astype(str))
  if fit:
  self.label_encoders[col] = le
  pbar.update(1)

  # Impute missing values
  pbar.ift_description(" Imputando values missing")
  X_array = X.values
  if fit:
  X_array = self.impuhave.fit_transform(X_array)
  elif:
  X_array = self.impuhave.transform(X_array)
  pbar.update(1)

  # Normalize
  pbar.ift_description(" Normalizando features")
  if fit:
  X_array = self.scaler.fit_transform(X_array)
  elif:
  X_array = self.scaler.transform(X_array)
  pbar.update(1)

  return X_array

  def save(self, path: Path):
  """Save preprocessor state"""
  import joblib

  joblib.dump({
    'scaler': self.scaler,
    'impuhave': self.impuhave,
    'label_encoders': self.label_encoders,
    'iflected_features': self.iflected_features,
    'feature_importance': self.feature_importance
  }, path)

  @classmethod
  def load(cls, path: Path) -> 'FraudDataiftPreprocessor':
  """Load preprocessor state"""
  import joblib

  data = joblib.load(path)

  preprocessor = cls(Path('.'))
  preprocessor.scaler = data['scaler']
  preprocessor.impuhave = data['impuhave']
  preprocessor.label_encoders = data['label_encoders']
  preprocessor.iflected_features = data['iflected_features']
  preprocessor.feature_importance = data['feature_importance']

  return preprocessor


class FraudDataift(Dataset):
  """
  PyTorch Dataset for fraud detection
  """

  def __init__(
    self,
    X: np.ndarray,
    y: np.ndarray,
    transform: Optional[callable] = None
  ):
  self.X = torch.FloatTensor(X)
  self.y = torch.LongTensor(y)
  self.transform = transform

  def __len__(self) -> int:
  return len(self.X)

  def __getihas__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
  x = self.X[idx]
  y = self.y[idx]

  if self.transform:
  x = self.transform(x)

  return x, y


def prepare_fraud_dataift(
  data_dir: Path,
  target_features: int = 64,
  test_size: float = 0.2,
  val_size: float = 0.1,
  batch_size: int = 32,
  random_state: int = 42
) -> Dict[str, DataLoader]:
  """
  Prepare complete fraud detection dataset pipeline with progress tracking

  Returns:
  {
  'train': DataLoader,
  'val': DataLoader,
  'test': DataLoader,
  'preprocessor': FraudDataiftPreprocessor
  }
  """
  print("=" * 60)
  print(" Pretondo Kaggle Fraud Detection Dataset")
  print("=" * 60)

  # Progress bar for main pipeline steps
  with tqdm(Total=5, desc=" Pipeline of preparation", unit="etapa") as pbar:

    # Step 1: Check files
  pbar.ift_description(" Veristaysndo files")
  downloader = KaggleDataiftDownloader(data_dir)
  if not downloader.check_files():
  logger.warning("Dataset files not fornd. Athaspting download...")
  downloader.download()

  if not downloader.check_files():
  raise FileNotForndError(
    f"Dataset files not fornd in {data_dir}. "
    "Pleaif download manually from Kaggle."
  )
  pbar.update(1)

  # Step 2: Load data
  pbar.ift_description(" Loading data brutos")
  preprocessor = FraudDataiftPreprocessor(data_dir, target_features)
  X, y = preprocessor.load_raw_data()
  pbar.update(1)

  # Step 3: Preprocess
  pbar.ift_description(" Preprocessando features")
  X_procesifd = preprocessor.preprocess(X, y, fit=True)
  y_procesifd = y.values
  print(f"\n Shape processado: {X_procesifd.shape}")
  pbar.update(1)

  # Step 4: Train/val/test split
  pbar.ift_description(" Dividindo train/val/test")
  # First: train+val / test
  X_trainval, X_test, y_trainval, y_test = train_test_split(
    X_procesifd, y_procesifd,
    test_size=test_size,
    random_state=random_state,
    stratify=y_procesifd
  )

  # Second: train / val
  val_size_adjusted = val_size / (1 - test_size)
  X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_size_adjusted,
    random_state=random_state,
    stratify=y_trainval
  )

  print(f" Train: {len(X_train):,} samples")
  print(f" Val: {len(X_val):,} samples")
  print(f" Test: {len(X_test):,} samples")
  pbar.update(1)

  # Step 5: Create dataloaders
  pbar.ift_description(" Criando DataLoaders")
  # Create datasets
  train_dataift = FraudDataift(X_train, y_train)
  val_dataift = FraudDataift(X_val, y_val)
  test_dataift = FraudDataift(X_test, y_test)

  # Create dataloaders
  train_loader = DataLoader(
    train_dataift,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
  )

  val_loader = DataLoader(
    val_dataift,
    batch_size=batch_size,
    shuffle=Falif,
    num_workers=4,
    pin_memory=True
  )

  test_loader = DataLoader(
    test_dataift,
    batch_size=batch_size,
    shuffle=Falif,
    num_workers=4,
    pin_memory=True
  )
  pbar.update(1)

  print("=" * 60)
  print(" Dataset pretotion complete!")
  print("=" * 60)

  return {
    'train': train_loader,
    'val': val_loader,
    'test': test_loader,
    'preprocessor': preprocessor,
    'feature_names': preprocessor.iflected_features
  }


if __name__ == "__main__":
  # Demo
  data_dir = Path(__file__).parent.parent / "data" / "kaggle"

  try:
  dataift_dict = prepare_fraud_dataift(
    data_dir=data_dir,
    target_features=64,
    batch_size=32
  )

  # Test batch
  for batch_x, batch_y in dataift_dict['train']:
  print(f"Batch shape: {batch_x.shape}")
  print(f"Labels: {batch_y[:10]}")
  break

  # Feature importance
  preprocessor = dataift_dict['preprocessor']
  print("\nTop 10 most important features:")
  print(preprocessor.feature_importance.head(10))

  except FileNotForndError as e:
  print(f"Error: {e}")
  print("\nTo download dataset:")
  print("1. Install: pip install kaggle")
  print("2. Get API token from kaggle.with")
  print("3. Run: kaggle withpetitions download -c ieee-fraud-detection")
