#!/usr/bin/env python3
"""
**Descrição:** Script automatizado de download de dataset do Kaggle.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Script para baixar o dataset IEEE-CIS Fraud Detection do Kaggle.

Uso:
 python download_kaggle_dataset.py

Requisitos:
 1. Conta no Kaggle
 2. Token API configurado em ~/.kaggle/kaggle.json
 3. Aceitar os termos da competição em:
 https://www.kaggle.com/c/ieee-fraud-detection
"""

import sys
from pathlib import Path
import zipfile

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

def check_kaggle_config():
 """Verifica se o Kaggle API está configurado corretamente."""
 kaggle_dir = Path.home() / '.kaggle'
 kaggle_json = kaggle_dir / 'kaggle.json'
 
 if not kaggle_json.exists():
 print(" Erro: kaggle.json não encontrado em ~/.kaggle/")
 print("\n Para configurar:")
 print("1. Acesse: https://www.kaggle.com/settings/account")
 print("2. Role até 'API' e clique em 'Create New API Token'")
 print("3. O arquivo kaggle.json será baixado automaticamente")
 print("4. Execute: mv ~/Downloads/kaggle.json ~/.kaggle/")
 print("5. Execute: chmod 600 ~/.kaggle/kaggle.json")
 return False
 
 # Check file format
 import json
 try:
 with open(kaggle_json) as f:
 config = json.load(f)
 if 'username' not in config or 'key' not in config:
 print(" Erro: kaggle.json está mal formatado")
 print(f"\nConteúdo atual: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"username":"seu_username","key":"sua_api_key"}')
 return False
 except json.JSONDecodeError:
 print(" Erro: kaggle.json não é um JSON válido")
 print(f"\nConteúdo atual: {kaggle_json.read_text()}")
 print("\n Formato correto:")
 print('{"username":"seu_username","key":"sua_api_key"}')
 return False
 
 return True

def download_dataset():
 """Baixa o dataset do Kaggle."""
 if not check_kaggle_config():
 return False
 
 try:
 from kaggle.api.kaggle_api_extended import KaggleApi
 except ImportError:
 print(" Erro: kaggle package não instalado")
 print("Execute: pip install kaggle")
 return False
 
 # Setup
 data_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
 data_dir.mkdir(parents=True, exist_ok=True)
 
 print(" Iniciando download do dataset IEEE-CIS Fraud Detection...")
 print(f" Diretório: {data_dir}")
 print(" Aviso: O dataset tem ~500MB, pode levar alguns minutos.\n")
 
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
 quiet=False
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
 
 print(f"\n Dataset pronto em: {data_dir}")
 print("\n Agora você pode executar a célula 9 do notebook novamente!")
 return True
 
 except Exception as e:
 error_msg = str(e)
 
 if '403' in error_msg or 'forbidden' in error_msg.lower():
 print("\n Erro 403: Você precisa aceitar os termos da competição!")
 print("\n Solução:")
 print("1. Acesse: https://www.kaggle.com/c/ieee-fraud-detection")
 print("2. Clique no botão 'Join Competition'")
 print("3. Aceite os termos e condições")
 print("4. Execute este script novamente")
 elif '404' in error_msg:
 print(f"\n Erro 404: Competição não encontrada")
 print("Verifique se o nome está correto: ieee-fraud-detection")
 else:
 print(f"\n Erro: {e}")
 print("\nPara mais detalhes, ative o modo verbose:")
 print(" kaggle competitions download -c ieee-fraud-detection")
 
 return False

def download_manual_instructions():
 """Mostra instruções para download manual."""
 data_dir = Path(__file__).parent.parent / 'data' / 'kaggle'
 
 print("\n" + "="*70)
 print(" INSTRUÇÕES PARA DOWNLOAD MANUAL")
 print("="*70)
 print("\nSe o download automático falhar, você pode baixar manualmente:")
 print("\n1⃣ Acesse:")
 print(" https://www.kaggle.com/c/ieee-fraud-detection/data")
 print("\n2⃣ Clique em 'Download All' (ou baixe individualmente):")
 print(" train_transaction.csv (~368 MB)")
 print(" train_identity.csv (~35 MB)")
 print(" test_transaction.csv (~140 MB)")
 print(" test_identity.csv (~13 MB)")
 print(f"\n3⃣ Extraia os arquivos CSV em:")
 print(f" {data_dir}")
 print("\n4⃣ Verifique se os arquivos estão presentes:")
 print(" cd", data_dir)
 print(" ls -lh *.csv")
 print("\n5⃣ Execute a célula 9 do notebook novamente")
 print("="*70)

if __name__ == '__main__':
 print(" Kaggle Dataset Downloader")
 print("="*70 + "\n")
 
 success = download_dataset()
 
 if not success:
 download_manual_instructions()
 sys.exit(1)
