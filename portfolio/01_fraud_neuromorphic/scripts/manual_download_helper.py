#!/usr/bin/env python3
"""
Helper script para download manual do dataset IEEE-CIS Fraud Detection.

Este script:
1. Abre o navegador na página de download do Kaggle
2. Aguarda você baixar os arquivos
3. Detecta automaticamente quando os arquivos aparecem em ~/Downloads
4. Move e organiza os arquivos para o diretório correto
5. Valida a integridade dos dados

Uso:
    python manual_download_helper.py
"""

import sys
import time
import webbrowser
from pathlib import Path
import zipfile
import shutil

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")

def print_step(number, text):
    """Print formatted step."""
    print(f"{number}️⃣  {text}")

def wait_for_user(prompt="Pressione ENTER quando terminar..."):
    """Wait for user confirmation."""
    input(f"\n{prompt} ")

def check_downloads_folder():
    """Check if Downloads folder exists."""
    downloads = Path.home() / 'Downloads'
    if not downloads.exists():
        print(f"⚠️  Pasta Downloads não encontrada em: {downloads}")
        downloads = Path(input("Digite o caminho completo da pasta de downloads: ").strip())
    return downloads

def find_kaggle_files(downloads_dir):
    """Find Kaggle dataset files in Downloads folder."""
    # Look for zip file
    zip_files = list(downloads_dir.glob('*fraud*.zip'))
    
    # Look for CSV files
    csv_files = list(downloads_dir.glob('train_transaction.csv'))
    csv_files.extend(downloads_dir.glob('train_identity.csv'))
    csv_files.extend(downloads_dir.glob('test_transaction.csv'))
    csv_files.extend(downloads_dir.glob('test_identity.csv'))
    
    return zip_files, csv_files

def extract_and_move_files(source_files, target_dir, is_zip=False):
    """Extract zip or move CSV files to target directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if is_zip:
        zip_file = source_files[0]
        print(f"\n📦 Extraindo: {zip_file.name}")
        print(f"   Para: {target_dir}")
        
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            members = zip_ref.namelist()
            print(f"   Arquivos no ZIP: {len(members)}")
            
            for member in members:
                if member.endswith('.csv'):
                    print(f"   ✓ Extraindo: {member}")
                    zip_ref.extract(member, target_dir)
        
        print(f"\n✅ Extração concluída!")
        
        # Ask if user wants to delete the zip
        delete = input("\n🗑️  Deletar o arquivo ZIP? (s/N): ").strip().lower()
        if delete == 's':
            zip_file.unlink()
            print(f"✅ ZIP deletado: {zip_file.name}")
    else:
        print(f"\n📁 Movendo arquivos CSV para: {target_dir}")
        for csv_file in source_files:
            target_file = target_dir / csv_file.name
            if target_file.exists():
                print(f"   ⚠️  Arquivo já existe: {csv_file.name}")
                overwrite = input("   Sobrescrever? (s/N): ").strip().lower()
                if overwrite != 's':
                    continue
            
            shutil.move(str(csv_file), str(target_file))
            print(f"   ✓ Movido: {csv_file.name}")
        
        print(f"\n✅ Arquivos movidos!")

def validate_dataset(data_dir):
    """Validate that all required files are present."""
    required_files = [
        'train_transaction.csv',
        'train_identity.csv',
        'test_transaction.csv',
        'test_identity.csv'
    ]
    
    print("\n🔍 Validando arquivos...\n")
    
    all_present = True
    total_size = 0
    
    for filename in required_files:
        filepath = data_dir / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            total_size += size_mb
            print(f"   ✓ {filename:25s} ({size_mb:>6.1f} MB)")
        else:
            print(f"   ✗ {filename:25s} (FALTANDO)")
            all_present = False
    
    print(f"\n   Total: {total_size:.1f} MB")
    
    return all_present

def main():
    print_header("🤖 KAGGLE DATASET - DOWNLOAD MANUAL HELPER")
    
    # Setup paths
    project_dir = Path(__file__).parent.parent
    data_dir = project_dir / 'data' / 'kaggle'
    downloads_dir = check_downloads_folder()
    
    print(f"📂 Pasta de Downloads: {downloads_dir}")
    print(f"📂 Destino final: {data_dir}")
    
    # Step 1: Open browser
    print_header("PASSO 1: ABRIR PÁGINA DE DOWNLOAD")
    print_step(1, "Vou abrir o navegador na página do Kaggle")
    print("   URL: https://www.kaggle.com/c/ieee-fraud-detection/data")
    
    open_browser = input("\n🌐 Abrir navegador agora? (S/n): ").strip().lower()
    if open_browser != 'n':
        webbrowser.open('https://www.kaggle.com/c/ieee-fraud-detection/data')
        print("✅ Navegador aberto!")
    
    # Step 2: Instructions
    print_header("PASSO 2: BAIXAR OS ARQUIVOS")
    print("No navegador:")
    print_step(1, "Faça login na sua conta Kaggle")
    print_step(2, "Clique no botão 'Download All' (ou baixe individualmente)")
    print_step(3, "Aguarde o download completar (~500 MB)")
    print("\nOPÇÕES DE DOWNLOAD:")
    print("   A) Download All → arquivo ZIP único")
    print("   B) Download individual → 4 arquivos CSV separados")
    
    wait_for_user("\n⏸️  Pressione ENTER quando o download TERMINAR...")
    
    # Step 3: Detect files
    print_header("PASSO 3: DETECTAR ARQUIVOS BAIXADOS")
    print("🔍 Procurando arquivos em:", downloads_dir)
    
    zip_files, csv_files = find_kaggle_files(downloads_dir)
    
    if not zip_files and not csv_files:
        print("\n⚠️  Nenhum arquivo encontrado!")
        print("\nArquivos esperados:")
        print("   - ieee-fraud-detection.zip (ou similar)")
        print("   OU")
        print("   - train_transaction.csv")
        print("   - train_identity.csv")
        print("   - test_transaction.csv")
        print("   - test_identity.csv")
        
        manual_path = input("\n📁 Digite o caminho completo do arquivo/pasta: ").strip()
        if manual_path:
            manual_path = Path(manual_path)
            if manual_path.is_file() and manual_path.suffix == '.zip':
                zip_files = [manual_path]
            elif manual_path.is_dir():
                csv_files = list(manual_path.glob('*.csv'))
        
        if not zip_files and not csv_files:
            print("\n❌ Nenhum arquivo válido encontrado. Saindo...")
            return 1
    
    # Show found files
    if zip_files:
        print(f"\n✅ Encontrado arquivo ZIP:")
        for f in zip_files:
            print(f"   📦 {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    
    if csv_files:
        print(f"\n✅ Encontrados {len(csv_files)} arquivo(s) CSV:")
        for f in csv_files:
            print(f"   📄 {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")
    
    # Step 4: Move/Extract
    print_header("PASSO 4: ORGANIZAR ARQUIVOS")
    
    confirm = input("📁 Mover arquivos para o diretório do projeto? (S/n): ").strip().lower()
    if confirm == 'n':
        print("❌ Operação cancelada pelo usuário.")
        return 0
    
    try:
        if zip_files:
            extract_and_move_files(zip_files, data_dir, is_zip=True)
        else:
            extract_and_move_files(csv_files, data_dir, is_zip=False)
    except Exception as e:
        print(f"\n❌ Erro ao processar arquivos: {e}")
        return 1
    
    # Step 5: Validate
    print_header("PASSO 5: VALIDAR DATASET")
    
    if validate_dataset(data_dir):
        print("\n" + "="*70)
        print("🎉 SUCESSO! Dataset configurado corretamente!")
        print("="*70)
        print(f"\n📂 Localização: {data_dir}")
        print("\n✅ Próximo passo:")
        print("   Execute a célula 9 do notebook: 05_production_solutions.ipynb")
        print("   O dataset será carregado automaticamente!")
        print("\n" + "="*70)
        return 0
    else:
        print("\n⚠️  Alguns arquivos estão faltando!")
        print("\n🔧 Solução:")
        print("   1. Verifique se baixou TODOS os 4 arquivos CSV")
        print("   2. Execute este script novamente")
        print("   3. Ou baixe os arquivos faltantes manualmente")
        return 1

if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n❌ Operação cancelada pelo usuário (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
