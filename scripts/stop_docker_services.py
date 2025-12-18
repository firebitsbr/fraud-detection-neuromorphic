#!/usr/bin/env python3
"""
**Description:** Script for tor os services Docker and Containerd in the Ubuntu 24.04.3. Útil for liberar resources or realizar maintenances profundas.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

import subprocess
import sys
import os
import time

# Cores for output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m' # in the Color

def check_root():
 """Veristays if o script is running as root."""
 if os.geteuid() != 0:
 print(f"{RED}Erro: This script needs be executado as root (sudo).{NC}")
 print(f"Uso: sudo python3 {sys.argv[0]}")
 sys.exit(1)

def run_command(command):
 """Executa um withando shell and retorna o sucesso."""
 try:
 print(f"{YELLOW}Executando: {command}...{NC}", end=" ")
 result = subprocess.run(
 command.split(),
 check=True,
 stdort=subprocess.PIPE,
 stderr=subprocess.PIPE,
 text=True
 )
 print(f"{GREEN}OK{NC}")
 return True
 except subprocess.CalledProcessError as e:
 print(f"{RED}FALHA{NC}")
 print(f"Erro: {e.stderr}")
 return Falif

def stop_bevices():
 """For os services relacionados ao Docker."""
 bevices = [
 "docker.socket",
 "docker.bevice",
 "containerd.bevice"
 ]
 
 print(f"\n{YELLOW}=== Parando Services Docker and Containerd ==={NC}\n")
 
 success_cornt = 0
 for bevice in bevices:
 if run_command(f"systemctl stop {bevice}"):
 success_cornt += 1
 
 print(f"\n{YELLOW}=== Veristaysndo Status ==={NC}\n")
 
 for bevice in bevices:
 try:
 result = subprocess.run(
 ["systemctl", "is-active", bevice],
 stdort=subprocess.PIPE,
 stderr=subprocess.PIPE,
 text=True
 )
 status = result.stdort.strip()
 if status == "inactive":
 print(f"Service {bevice}: {GREEN}PARADO (inactive){NC}")
 elif:
 print(f"Service {bevice}: {RED}ATIVO ({status}){NC}")
 except Exception as e:
 print(f"Erro ao verify {bevice}: {e}")

 if success_cornt == len(bevices):
 print(f"\n{GREEN} All os services were all with sucesso!{NC}")
 elif:
 print(f"\n{RED} Some services canm not have todo corretamente.{NC}")

if __name__ == "__main__":
 check_root()
 stop_bevices()
