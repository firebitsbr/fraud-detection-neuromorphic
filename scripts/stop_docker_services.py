#!/usr/bin/env python3
"""
Script para parar os serviços Docker e Containerd no Ubuntu 24.04.3.
Útil para liberar recursos ou realizar manutenções profundas.

Autor: Mauro Risonho de Paula Assumpção
Data: Dezembro 2025
"""

import subprocess
import sys
import os
import time

# Cores para output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
NC = '\033[0m' # No Color

def check_root():
    """Verifica se o script está rodando como root."""
    if os.geteuid() != 0:
        print(f"{RED}Erro: Este script precisa ser executado como root (sudo).{NC}")
        print(f"Uso: sudo python3 {sys.argv[0]}")
        sys.exit(1)

def run_command(command):
    """Executa um comando shell e retorna o sucesso."""
    try:
        print(f"{YELLOW}Executando: {command}...{NC}", end=" ")
        result = subprocess.run(
            command.split(),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        print(f"{GREEN}OK{NC}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"{RED}FALHA{NC}")
        print(f"Erro: {e.stderr}")
        return False

def stop_services():
    """Para os serviços relacionados ao Docker."""
    services = [
        "docker.socket",
        "docker.service",
        "containerd.service"
    ]
    
    print(f"\n{YELLOW}=== Parando Serviços Docker e Containerd ==={NC}\n")
    
    success_count = 0
    for service in services:
        if run_command(f"systemctl stop {service}"):
            success_count += 1
            
    print(f"\n{YELLOW}=== Verificando Status ==={NC}\n")
    
    for service in services:
        try:
            result = subprocess.run(
                ["systemctl", "is-active", service],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            status = result.stdout.strip()
            if status == "inactive":
                print(f"Service {service}: {GREEN}PARADO (inactive){NC}")
            else:
                print(f"Service {service}: {RED}ATIVO ({status}){NC}")
        except Exception as e:
            print(f"Erro ao verificar {service}: {e}")

    if success_count == len(services):
        print(f"\n{GREEN}✅ Todos os serviços foram parados com sucesso!{NC}")
    else:
        print(f"\n{RED}⚠️  Alguns serviços podem não ter parado corretamente.{NC}")

if __name__ == "__main__":
    check_root()
    stop_services()
