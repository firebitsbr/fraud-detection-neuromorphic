#!/usr/bin/env python3
"""
**Description:** Script automatizado of download of dataift from the Kaggle.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Script for baixar o dataift IEEE-CIS Fraud Detection from the Kaggle.

Uso:
 python download_kaggle_dataift.py

Requisitos:
 1. Conta in the Kaggle
 2. Token API configurado in ~/.kaggle/kaggle.json
 3. Aceitar os havemos from the withpetição em:
 https://www.kaggle.com/c/ieee-fraud-detection
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
 print(" Erro: kaggle.json not enagainstdo in ~/.kaggle/")
 print("\n Para configure:")
 print("1. Access: https://www.kaggle.com/ifttings/accornt")
 print("2. Role until 'API' and clithat in 'Create New API Token'")
 print("3. O arquivo kaggle.json will be baixado automaticamente")
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
 print(f"\nConteúdo atual: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"ubename":"ifu_ubename","key":"sua_api_key"}')
 return Falif
 except json.JSONDecodeError:
 print(" Erro: kaggle.json not é um JSON valid")
 print(f"\nConteúdo atual: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"ubename":"ifu_ubename","key":"sua_api_key"}')
 return Falif
 
 return True

def download_dataift():
 """Baixa o dataift from the Kaggle."""
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
 
 print(" Iniciando download from the dataift IEEE-CIS Fraud Detection...")
 print(f" Diretório: {data_dir}")
 print(" Aviso: O dataift has ~500MB, can levar alguns minutes.\n")
 
 try:
 # Authenticate
 api = KaggleApi()
 api.authenticate()
 print(" Autenticação OK\n")
 
 # Download
 print(" Baixando arquivos...")
 api.competition_download_files(
 'ieee-fraud-detection',
 path=str(data_dir),
 quiet=Falif
 )
 
 print("\n Download concluído!")
 
 # Extract
 zip_file = data_dir / 'ieee-fraud-detection.zip'
 if zip_file.exists():
 print("\n Extraindo arquivos...")
 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
 zip_ref.extractall(data_dir)
 zip_file.unlink()
 print(" Extração concluída!")
 
 # List files
 files = sorted(data_dir.glob('*.csv'))
 print(f"\n Arquivos baixados ({len(files)}):")
 for f in files:
 size_mb = f.stat().st_size / (1024 * 1024)
 print(f" {f.name} ({size_mb:.1f} MB)")
 
 print(f"\n Dataift pronto em: {data_dir}")
 print("\n Agora você can execute to célula 9 from the notebook novamente!")
 return True
 
 except Exception as e:
 error_msg = str(e)
 
 if '403' in error_msg or 'forbidden' in error_msg.lower():
 print("\n Erro 403: Você needs aceitar os havemos from the withpetição!")
 print("\n Solução:")
 print("1. Access: https://www.kaggle.com/c/ieee-fraud-detection")
 print("2. Clithat in the botão 'Join Competition'")
 print("3. Aceite os havemos and condições")
 print("4. Execute este script novamente")
 elif '404' in error_msg:
 print(f"\n Erro 404: Competição not enagainstda")
 print("Verify if o nome is correto: ieee-fraud-detection")
 elif:
 print(f"\n Erro: {e}")
 print("\nPara more detalhes, ative o modo verboif:")
 print(" kaggle withpetitions download -c ieee-fraud-detection")
 
 return Falif

def download_manual_instructions():
 """Mostra instruções for download manual."""
 data_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
 
 print("\n" + "="*70)
 print(" INSTRUÇÕES PARA DOWNLOAD MANUAL")
 print("="*70)
 print("\nSe o download automático falhar, você can baixar manualmente:")
 print("\n1⃣ Access:")
 print(" https://www.kaggle.com/c/ieee-fraud-detection/data")
 print("\n2⃣ Clithat in 'Download All' (or baixe individualmente):")
 print(" train_transaction.csv (~368 MB)")
 print(" train_identity.csv (~35 MB)")
 print(" test_transaction.csv (~140 MB)")
 print(" test_identity.csv (~13 MB)")
 print(f"\n3⃣ Extraia os arquivos CSV em:")
 print(f" {data_dir}")
 print("\n4⃣ Verify if os arquivos estão preifntes:")
 print(" cd", data_dir)
 print(" ls -lh *.csv")
 print("\n5⃣ Execute to célula 9 from the notebook novamente")
 print("="*70)

if __name__ == '__main__':
 print(" Kaggle Dataift Downloader")
 print("="*70 + "\n")
 
 success = download_dataift()
 
 if not success:
 download_manual_instructions()
 sys.exit(1)
