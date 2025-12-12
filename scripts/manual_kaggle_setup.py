#!/usr/bin/env python3
"""
**Descrição:** Configuração interativa de dataset Kaggle com auto-detecção.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Manual Kaggle Dataset Setup Script
===================================

Script interativo para guiar o usuário no download manual do dataset
IEEE-CIS Fraud Detection do Kaggle.

Autor: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
Data: Dezembro 2025
Licença: MIT
"""

import sys
import time
import webbrowser
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
 print_colored(text.center(70), BOLD + BLUE)
 print("="*70 + "\n")

def print_step(number, text, color=GREEN):
 """Print formatted step."""
 print_colored(f"\n{number} {text}", color)

def print_file(filename, size=""):
 """Print file with checkmark."""
 size_str = f" (~{size})" if size else ""
 print(f" {filename}{size_str}")

def wait_for_enter():
 """Wait for user to press ENTER."""
 input(f"\n{YELLOW}Pressione ENTER para continuar...{RESET} ")

def clear_screen():
 """Clear terminal screen."""
 os.system('clear' if os.name != 'nt' else 'cls')

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
 
 for filename, size in required_files.items():
 file_path = data_dir / filename
 if file_path.exists():
 actual_size = file_path.stat().st_size / (1024**2) # MB
 existing.append((filename, f"{actual_size:.1f} MB"))
 else:
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

def scan_for_kaggle_files(search_dir):
 """Scan directory for Kaggle dataset files."""
 kaggle_files = {
 'zip': [],
 'csv': []
 }
 
 # Look for ZIP files
 for pattern in ['*fraud*.zip', '*ieee*.zip', '*.zip']:
 kaggle_files['zip'].extend(search_dir.glob(pattern))
 
 # Look for CSV files
 csv_names = [
 'train_transaction.csv',
 'train_identity.csv', 
 'test_transaction.csv',
 'test_identity.csv'
 ]
 
 for csv_name in csv_names:
 found = list(search_dir.glob(csv_name))
 kaggle_files['csv'].extend(found)
 
 return kaggle_files

def move_or_extract_files(source_files, target_dir, file_type='csv'):
 """Move or extract files to target directory."""
 target_dir.mkdir(parents=True, exist_ok=True)
 moved_count = 0
 
 if file_type == 'zip':
 for zip_file in source_files:
 print_colored(f"\n Extraindo: {zip_file.name}", BLUE)
 try:
 with zipfile.ZipFile(zip_file, 'r') as zip_ref:
 members = [m for m in zip_ref.namelist() if m.endswith('.csv')]
 
 for member in members:
 print(f" Extraindo: {member}...")
 zip_ref.extract(member, target_dir)
 moved_count += 1
 
 print_colored(f" Extração concluída: {moved_count} arquivos", GREEN)
 
 # Ask to delete ZIP
 delete = input(f"\n{YELLOW}Deletar o arquivo ZIP? (s/N): {RESET}").strip().lower()
 if delete == 's':
 zip_file.unlink()
 print_colored(f" ZIP deletado: {zip_file.name}", GREEN)
 
 except Exception as e:
 print_colored(f" Erro ao extrair {zip_file.name}: {e}", RED)
 
 else: # CSV files
 print_colored(f"\n Movendo arquivos CSV...", BLUE)
 for csv_file in source_files:
 target_file = target_dir / csv_file.name
 
 if target_file.exists():
 print_colored(f" {csv_file.name} já existe", YELLOW)
 overwrite = input(" Sobrescrever? (s/N): ").strip().lower()
 if overwrite != 's':
 continue
 
 try:
 shutil.move(str(csv_file), str(target_file))
 print_colored(f" Movido: {csv_file.name}", GREEN)
 moved_count += 1
 except Exception as e:
 print_colored(f" Erro ao mover {csv_file.name}: {e}", RED)
 
 if moved_count > 0:
 print_colored(f"\n {moved_count} arquivos movidos com sucesso!", GREEN)
 
 return moved_count

def main():
 """Main function."""
 clear_screen()
 
 # Header
 print_header(" CONFIGURAÇÃO MANUAL - DATASET KAGGLE")
 print_colored("IEEE-CIS Fraud Detection Dataset", BOLD)
 print_colored("Autor: Mauro Risonho de Paula Assumpção\n", BLUE)
 
 # Get directories
 project_root = get_project_root()
 data_dir = project_root / 'data' / 'kaggle'
 
 print(f" Diretório do projeto: {project_root}")
 print(f" Diretório de destino: {data_dir}\n")
 
 # Check existing files
 print_header("1⃣ VERIFICANDO ARQUIVOS EXISTENTES")
 
 existing, missing = check_existing_files(data_dir)
 
 if existing:
 print_colored(" Arquivos já presentes:", GREEN)
 for filename, size in existing:
 print(f" {filename} ({size})")
 
 if missing:
 print_colored(f"\n Arquivos faltando ({len(missing)}):", RED)
 for filename, size in missing:
 print(f" {filename} (~{size})")
 else:
 print_colored("\n Todos os arquivos já estão presentes!", GREEN)
 print_colored("Nada a fazer. Dataset completo!", BOLD + GREEN)
 return 0
 
 wait_for_enter()
 
 # Instructions
 print_header("2⃣ INSTRUÇÕES PARA DOWNLOAD")
 
 print_step("1.", "Acesse a página do Kaggle:", BLUE)
 print(" https://www.kaggle.com/c/ieee-fraud-detection/data")
 
 print_step("2.", "Faça login na sua conta Kaggle", BLUE)
 print(" (Se não tiver, crie uma conta gratuita)")
 
 print_step("3.", "Aceite as regras da competição", BLUE)
 print(" (Clique em 'I Understand and Accept')")
 
 print_step("4.", "Baixe os arquivos:", BLUE)
 print(" Opção A: Clique em 'Download All' (todos de uma vez)")
 print(" Opção B: Baixe individualmente os arquivos CSV")
 
 print_step("5.", "Arquivos necessários:", YELLOW)
 for filename, size in missing:
 print_file(filename, size)
 
 # Ask to open browser
 print()
 open_browser = input(f"{YELLOW}Abrir o navegador automaticamente? (S/n): {RESET}").strip().lower()
 
 if open_browser != 'n':
 print_colored("\n Abrindo navegador...", BLUE)
 webbrowser.open('https://www.kaggle.com/c/ieee-fraud-detection/data')
 time.sleep(2)
 
 wait_for_enter()
 
 # Auto-detect files
 print_header("3⃣ DETECTANDO ARQUIVOS BAIXADOS")
 
 downloads_dir = find_downloads_folder()
 print(f" Procurando em: {downloads_dir}\n")
 
 print_colored("Aguardando você baixar os arquivos...", YELLOW)
 print("(O script irá verificar automaticamente)")
 print()
 
 max_attempts = 60 # 5 minutes
 attempt = 0
 
 while attempt < max_attempts:
 kaggle_files = scan_for_kaggle_files(downloads_dir)
 
 total_found = len(kaggle_files['zip']) + len(kaggle_files['csv'])
 
 if total_found > 0:
 print_colored(f"\r Encontrados {total_found} arquivo(s)! ", GREEN)
 break
 
 print(f"\rProcurando... ({attempt + 1}s) ", end='', flush=True)
 time.sleep(1)
 attempt += 1
 
 print() # New line
 
 if total_found == 0:
 print_colored("\n[TEMPO] Timeout: Nenhum arquivo detectado automaticamente", YELLOW)
 
 manual = input(f"\n{YELLOW}Deseja especificar manualmente o caminho? (S/n): {RESET}").strip().lower()
 if manual != 'n':
 custom_path = input("Digite o caminho completo da pasta com os arquivos: ").strip()
 downloads_dir = Path(custom_path)
 kaggle_files = scan_for_kaggle_files(downloads_dir)
 total_found = len(kaggle_files['zip']) + len(kaggle_files['csv'])
 
 if total_found == 0:
 print_colored("\n Nenhum arquivo encontrado.", RED)
 print("\nPor favor:")
 print("1. Certifique-se de baixar os arquivos do Kaggle")
 print("2. Execute este script novamente")
 return 1
 
 # Display found files
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
 print_colored("\n Operação cancelada pelo usuário.", YELLOW)
 return 1
 
 # Process files
 print_header("5⃣ PROCESSANDO ARQUIVOS")
 
 total_moved = 0
 
 if kaggle_files['zip']:
 total_moved += move_or_extract_files(kaggle_files['zip'], data_dir, 'zip')
 
 if kaggle_files['csv']:
 total_moved += move_or_extract_files(kaggle_files['csv'], data_dir, 'csv')
 
 # Final verification
 print_header("6⃣ VERIFICAÇÃO FINAL")
 
 existing, missing = check_existing_files(data_dir)
 
 print_colored(f" Arquivos presentes: {len(existing)}", GREEN)
 for filename, size in existing:
 print(f" {filename} ({size})")
 
 if missing:
 print_colored(f"\n Ainda faltam: {len(missing)}", YELLOW)
 for filename, size in missing:
 print(f" {filename}")
 
 print_colored("\n Execute o script novamente para completar o download.", BLUE)
 else:
 print_colored("\n SUCESSO! Todos os arquivos estão no lugar!", BOLD + GREEN)
 print_colored("\nVocê pode agora executar os notebooks do projeto.", GREEN)
 
 print_header("CONCLUÍDO")
 
 return 0 if not missing else 1

if __name__ == '__main__':
 try:
 sys.exit(main())
 except KeyboardInterrupt:
 print_colored("\n\n Operação cancelada pelo usuário (Ctrl+C)", YELLOW)
 sys.exit(1)
 except Exception as e:
 print_colored(f"\n\n Erro inesperado: {e}", RED)
 import traceback
 traceback.print_exc()
 sys.exit(1)
