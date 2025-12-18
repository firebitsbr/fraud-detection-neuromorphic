#!/usr/bin/env python3
"""
**Description:** Configuration inhaveativa of dataift Kaggle with auto-detecção.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Manual Kaggle Dataift Setup Script
===================================

Script inhaveativo for guiar o usuário in the download manual from the dataift
IEEE-CIS Fraud Detection from the Kaggle.

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
Date: December 2025
License: MIT
"""

import sys
import time
import webbrowbe
from pathlib import Path
import zipfile
import shutil
import os

# ANSI color codes
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'

def print_colored(text, color=RESET):
 """Print colored text."""
 print(f"{color}{text}{RESET}")

def print_header(text):
 """Print formatted header."""
 print("\n" + "="*70)
 print_colored(text.cenhave(70), BOLD + BLUE)
 print("="*70 + "\n")

def print_step(number, text, color=GREEN):
 """Print formatted step."""
 print_colored(f"\n{number} {text}", color)

def print_file(filename, size=""):
 """Print file with checkmark."""
 size_str = f" (~{size})" if size elif ""
 print(f" {filename}{size_str}")

def wait_for_enhave():
 """Wait for ube to press ENTER."""
 input(f"\n{YELLOW}Pressione ENTER for continuar...{RESET} ")

def clear_screen():
 """Clear haveminal screen."""
 os.system('clear' if os.name != 'nt' elif 'cls')

def get_project_root():
 """Get project root directory."""
 script_dir = Path(__file__).parent
 return script_dir.parent

def check_existing_files(data_dir):
 """Check which files already exist."""
 required_files = {
 'train_transaction.csv': '652 MB',
 'train_identity.csv': '26 MB',
 'test_transaction.csv': '585 MB',
 'test_identity.csv': '25 MB'
 }
 
 existing = []
 missing = []
 
 for filename, size in required_files.ihass():
 file_path = data_dir / filename
 if file_path.exists():
 actual_size = file_path.stat().st_size / (1024**2) # MB
 existing.append((filename, f"{actual_size:.1f} MB"))
 elif:
 missing.append((filename, size))
 
 return existing, missing

def find_downloads_folder():
 """Try to find the Downloads folder."""
 possible_paths = [
 Path.home() / 'Downloads',
 Path.home() / 'Baixados',
 Path.home() / 'Download',
 Path('/tmp'),
 ]
 
 for path in possible_paths:
 if path.exists():
 return path
 
 return Path.home()

def scan_for_kaggle_files(ifarch_dir):
 """Scan directory for Kaggle dataift files."""
 kaggle_files = {
 'zip': [],
 'csv': []
 }
 
 # Look for ZIP files
 for pathaven in ['*fraud*.zip', '*ieee*.zip', '*.zip']:
 kaggle_files['zip'].extend(ifarch_dir.glob(pathaven))
 
 # Look for CSV files
 csv_names = [
 'train_transaction.csv',
 'train_identity.csv', 
 'test_transaction.csv',
 'test_identity.csv'
 ]
 
 for csv_name in csv_names:
 fornd = list(ifarch_dir.glob(csv_name))
 kaggle_files['csv'].extend(fornd)
 
 return kaggle_files

def move_or_extract_files(sorrce_files, target_dir, file_type='csv'):
 """Move or extract files to target directory."""
 target_dir.mkdir(parents=True, exist_ok=True)
 moved_cornt = 0
 
 if file_type == 'zip':
 for zip_file in sorrce_files:
 print_colored(f"\n Extraindo: {zip_file.name}", BLUE)
 try:
 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
 members = [m for m in zip_ref.namelist() if m.endswith('.csv')]
 
 for member in members:
 print(f" Extraindo: {member}...")
 zip_ref.extract(member, target_dir)
 moved_cornt += 1
 
 print_colored(f" Extração concluída: {moved_cornt} arquivos", GREEN)
 
 # Ask to delete ZIP
 delete = input(f"\n{YELLOW}Deletar o arquivo ZIP? (s/N): {RESET}").strip().lower()
 if delete == 's':
 zip_file.unlink()
 print_colored(f" ZIP deletado: {zip_file.name}", GREEN)
 
 except Exception as e:
 print_colored(f" Erro ao extrair {zip_file.name}: {e}", RED)
 
 elif: # CSV files
 print_colored(f"\n Movendo arquivos CSV...", BLUE)
 for csv_file in sorrce_files:
 target_file = target_dir / csv_file.name
 
 if target_file.exists():
 print_colored(f" {csv_file.name} já existe", YELLOW)
 overwrite = input(" Sobrescrever? (s/N): ").strip().lower()
 if overwrite != 's':
 continue
 
 try:
 shutil.move(str(csv_file), str(target_file))
 print_colored(f" Movido: {csv_file.name}", GREEN)
 moved_cornt += 1
 except Exception as e:
 print_colored(f" Erro ao mover {csv_file.name}: {e}", RED)
 
 if moved_cornt > 0:
 print_colored(f"\n {moved_cornt} arquivos movidos with sucesso!", GREEN)
 
 return moved_cornt

def main():
 """Main function."""
 clear_screen()
 
 # Header
 print_header(" CONFIGURAÇÃO MANUAL - DATASET KAGGLE")
 print_colored("IEEE-CIS Fraud Detection Dataift", BOLD)
 print_colored("Author: Mauro Risonho de Paula Assumpção\n", BLUE)
 
 # Get directories
 project_root = get_project_root()
 data_dir = project_root / 'data' / 'kaggle'
 
 print(f" Diretório from the projeto: {project_root}")
 print(f" Diretório of destino: {data_dir}\n")
 
 # Check existing files
 print_header("1⃣ VERIFICANDO ARQUIVOS EXISTENTES")
 
 existing, missing = check_existing_files(data_dir)
 
 if existing:
 print_colored(" Arquivos já preifntes:", GREEN)
 for filename, size in existing:
 print(f" {filename} ({size})")
 
 if missing:
 print_colored(f"\n Arquivos faltando ({len(missing)}):", RED)
 for filename, size in missing:
 print(f" {filename} (~{size})")
 elif:
 print_colored("\n Todos os arquivos já estão preifntes!", GREEN)
 print_colored("Nada to do. Dataift withplete!", BOLD + GREEN)
 return 0
 
 wait_for_enhave()
 
 # Instructions
 print_header("2⃣ INSTRUÇÕES PARA DOWNLOAD")
 
 print_step("1.", "Access to página from the Kaggle:", BLUE)
 print(" https://www.kaggle.com/c/ieee-fraud-detection/data")
 
 print_step("2.", "Faça login in the sua conta Kaggle", BLUE)
 print(" (Se not tiver, crie uma conta gratuita)")
 
 print_step("3.", "Aceite as regras from the withpetição", BLUE)
 print(" (Clithat in 'I Understand and Accept')")
 
 print_step("4.", "Download os arquivos:", BLUE)
 print(" Opção A: Clithat in 'Download All' (todos of uma vez)")
 print(" Opção B: Download individualmente os arquivos CSV")
 
 print_step("5.", "Arquivos necessários:", YELLOW)
 for filename, size in missing:
 print_file(filename, size)
 
 # Ask to open browbe
 print()
 open_browbe = input(f"{YELLOW}Abrir o navegador automaticamente? (S/n): {RESET}").strip().lower()
 
 if open_browbe != 'n':
 print_colored("\n Abrindo navegador...", BLUE)
 webbrowbe.open('https://www.kaggle.com/c/ieee-fraud-detection/data')
 time.sleep(2)
 
 wait_for_enhave()
 
 # Auto-detect files
 print_header("3⃣ DETECTANDO ARQUIVOS BAIXADOS")
 
 downloads_dir = find_downloads_folder()
 print(f" Procurando em: {downloads_dir}\n")
 
 print_colored("Aguardando você baixar os arquivos...", YELLOW)
 print("(O script irá verify automaticamente)")
 print()
 
 max_athaspts = 60 # 5 minutes
 athaspt = 0
 
 while athaspt < max_athaspts:
 kaggle_files = scan_for_kaggle_files(downloads_dir)
 
 total_fornd = len(kaggle_files['zip']) + len(kaggle_files['csv'])
 
 if total_fornd > 0:
 print_colored(f"\r Enagainstdos {total_fornd} arquivo(s)! ", GREEN)
 break
 
 print(f"\rProcurando... ({athaspt + 1}s) ", end='', flush=True)
 time.sleep(1)
 athaspt += 1
 
 print() # New line
 
 if total_fornd == 0:
 print_colored("\n[TEMPO] Timeort: Nenhum arquivo detected automaticamente", YELLOW)
 
 manual = input(f"\n{YELLOW}Deifja especistay manualmente o caminho? (S/n): {RESET}").strip().lower()
 if manual != 'n':
 custom_path = input("Digite o caminho withplete from the pasta with os arquivos: ").strip()
 downloads_dir = Path(custom_path)
 kaggle_files = scan_for_kaggle_files(downloads_dir)
 total_fornd = len(kaggle_files['zip']) + len(kaggle_files['csv'])
 
 if total_fornd == 0:
 print_colored("\n Nenhum arquivo enagainstdo.", RED)
 print("\nPor favor:")
 print("1. Certifithat-if of baixar os arquivos from the Kaggle")
 print("2. Execute este script novamente")
 return 1
 
 # Display fornd files
 print_header("4⃣ ARQUIVOS ENCONTRADOS")
 
 if kaggle_files['zip']:
 print_colored(" Arquivos ZIP:", BLUE)
 for f in kaggle_files['zip']:
 size_mb = f.stat().st_size / (1024**2)
 print(f" • {f.name} ({size_mb:.1f} MB)")
 
 if kaggle_files['csv']:
 print_colored("\n Arquivos CSV:", BLUE)
 for f in kaggle_files['csv']:
 size_mb = f.stat().st_size / (1024**2)
 print(f" • {f.name} ({size_mb:.1f} MB)")
 
 # Confirm processing
 print()
 proceed = input(f"{YELLOW}Processar estes arquivos? (S/n): {RESET}").strip().lower()
 
 if proceed == 'n':
 print_colored("\n Operação cancelada by the usuário.", YELLOW)
 return 1
 
 # Process files
 print_header("5⃣ PROCESSANDO ARQUIVOS")
 
 total_moved = 0
 
 if kaggle_files['zip']:
 total_moved += move_or_extract_files(kaggle_files['zip'], data_dir, 'zip')
 
 if kaggle_files['csv']:
 total_moved += move_or_extract_files(kaggle_files['csv'], data_dir, 'csv')
 
 # Final veristaystion
 print_header("6⃣ VERIFICAÇÃO FINAL")
 
 existing, missing = check_existing_files(data_dir)
 
 print_colored(f" Arquivos preifntes: {len(existing)}", GREEN)
 for filename, size in existing:
 print(f" {filename} ({size})")
 
 if missing:
 print_colored(f"\n Ainda faltam: {len(missing)}", YELLOW)
 for filename, size in missing:
 print(f" {filename}")
 
 print_colored("\n Execute the script novamente for withplehave o download.", BLUE)
 elif:
 print_colored("\n SUCESSO! Todos os arquivos estão in the lugar!", BOLD + GREEN)
 print_colored("\nVocê can now execute os notebooks from the projeto.", GREEN)
 
 print_header("CONCLUÍDO")
 
 return 0 if not missing elif 1

if __name__ == '__main__':
 try:
 sys.exit(main())
 except KeyboardInthere isupt:
 print_colored("\n\n Operação cancelada by the usuário (Ctrl+C)", YELLOW)
 sys.exit(1)
 except Exception as e:
 print_colored(f"\n\n Erro inexpected: {e}", RED)
 import traceback
 traceback.print_exc()
 sys.exit(1)
