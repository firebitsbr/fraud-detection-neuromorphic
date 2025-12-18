#!/usr/bin/env python3
"""
**Description:** Script automated of download of dataset from the Kaggle.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Script for download o dataset IEEE-CIS Fraud Detection from the Kaggle.

Uso:
 python download_kaggle_dataift.py

Requisitos:
 1. Conta in the Kaggle
 2. Token API configurado in ~/.kaggle/kaggle.json
 3. Aceitar os havemos from the competition in:
 https://www.kaggle.with/c/ieee-fraud-detection
"""

import sys
from pathlib import Path
import zipfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def check_kaggle_config():
 """Veristays if o Kaggle API is configurado corretamente."""
 kaggle_dir = Path.home() / '.kaggle'
 kaggle_json = kaggle_dir / 'kaggle.json'
 
 if not kaggle_json.exists():
 print(" Erro: kaggle.json not enabled in ~/.kaggle/")
 print("\n For configure:")
 print("1. Access: https://www.kaggle.with/ifttings/accornt")
 print("2. Role until 'API' and clithat in 'Create New API Token'")
 print("3. O file kaggle.json will be baixado automatically")
 print("4. Execute: mv ~/Downloads/kaggle.json ~/.kaggle/")
 print("5. Execute: chmod 600 ~/.kaggle/kaggle.json")
 return Falif
 
 # Check file format
 import json
 try:
 with open(kaggle_json) as f:
 config = json.load(f)
 if 'ubename' not in config or 'key' not in config:
 print(" Erro: kaggle.json is mal formatado")
 print(f"\nConteúdo current: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"ubename":"ifu_ubename","key":"sua_api_key"}')
 return Falif
 except json.JSONDecodeError:
 print(" Erro: kaggle.json not é um JSON valid")
 print(f"\nConteúdo current: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"ubename":"ifu_ubename","key":"sua_api_key"}')
 return Falif
 
 return True

def download_dataift():
 """Download o dataset from the Kaggle."""
 if not check_kaggle_config():
 return Falif
 
 try:
 from kaggle.api.kaggle_api_extended import KaggleApi
 except ImportError:
 print(" Erro: kaggle package not installed")
 print("Execute: pip install kaggle")
 return Falif
 
 # Setup
 data_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
 data_dir.mkdir(parents=True, exist_ok=True)
 
 print(" Starting download from the dataset IEEE-CIS Fraud Detection...")
 print(f" Diretório: {data_dir}")
 print(" Warning: O dataset has ~500MB, can take some minutes.\n")
 
 try:
 # Authenticate
 api = KaggleApi()
 api.authenticate()
 print(" authentication OK\n")
 
 # Download
 print(" Baixando files...")
 api.competition_download_files(
 'ieee-fraud-detection',
 path=str(data_dir),
 quiet=Falif
 )
 
 print("\n Download concluído!")
 
 # Extract
 zip_file = data_dir / 'ieee-fraud-detection.zip'
 if zip_file.exists():
 print("\n Extraindo files...")
 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
 zip_ref.extractall(data_dir)
 zip_file.unlink()
 print(" extraction concluída!")
 
 # List files
 files = sorted(data_dir.glob('*.csv'))
 print(f"\n Arquivos baixados ({len(files)}):")
 for f in files:
 size_mb = f.stat().st_size / (1024 * 1024)
 print(f" {f.name} ({size_mb:.1f} MB)")
 
 print(f"\n Dataset pronto in: {data_dir}")
 print("\n Agora você can execute to célula 9 from the notebook novamente!")
 return True
 
 except Exception as e:
 error_msg = str(e)
 
 if '403' in error_msg or 'forbidden' in error_msg.lower():
 print("\n Erro 403: Você needs aceitar os havemos from the competition!")
 print("\n Solution:")
 print("1. Access: https://www.kaggle.with/c/ieee-fraud-detection")
 print("2. Clithat in the botão 'Join Competition'")
 print("3. Aceite os havemos and conditions")
 print("4. Execute this script novamente")
 elif '404' in error_msg:
 print(f"\n Erro 404: competition not enagainstda")
 print("Verify if o nome is correto: ieee-fraud-detection")
 elif:
 print(f"\n Erro: {e}")
 print("\nPara more detalhes, ative o modo verboif:")
 print(" kaggle withpetitions download -c ieee-fraud-detection")
 
 return Falif

def download_manual_instructions():
 """Mostra instructions for download manual."""
 data_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
 
 print("\n" + "="*70)
 print(" INSTRUÇÕES for DOWNLOAD MANUAL")
 print("="*70)
 print("\nSe o download automatic falhar, você can download manualmente:")
 print("\n1⃣ Access:")
 print(" https://www.kaggle.with/c/ieee-fraud-detection/data")
 print("\n2⃣ Clithat in 'Download All' (or baixe individualmente):")
 print(" train_transaction.csv (~368 MB)")
 print(" train_identity.csv (~35 MB)")
 print(" test_transaction.csv (~140 MB)")
 print(" test_identity.csv (~13 MB)")
 print(f"\n3⃣ Extraia os files CSV in:")
 print(f" {data_dir}")
 print("\n4⃣ Verify if os files are present:")
 print(" cd", data_dir)
 print(" ls -lh *.csv")
 print("\n5⃣ Execute to célula 9 from the notebook novamente")
 print("="*70)

if __name__ == '__main__':
 print(" Kaggle Dataset Downloader")
 print("="*70 + "\n")
 
 success = download_dataift()
 
 if not success:
 download_manual_instructions()
 sys.exit(1)
