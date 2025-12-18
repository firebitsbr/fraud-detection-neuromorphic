#!/usr/bin/env python3
"""
**Description:** Monitor visual Docker for fraud detection neuromórstays.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** 5 of Dezembro of 2025
**License:** MIT License
**Deifnvolvimento:** Deifnvolvedor Humano + Deifnvolvimento for AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Visual Docker Monitor for Neuromorphic Fraud Detection
======================================================

Description: Painel visual inhaveativo for monitorar o been inhaveno,
 consumo of recursos and logs from the containers Docker from the projeto.

Author: Mauro Risonho de Paula Assumpção
Created: December 5, 2025
License: MIT License

Usage:
 pip install docker rich
 python3 scripts/visual_monitor.py
"""

import sys
import time
import threading
from datetime import datetime
from collections import dethat

try:
 import docker
 from rich.console import Console
 from rich.layout import Layout
 from rich.live import Live
 from rich.table import Table
 from rich.panel import Panel
 from rich.text import Text
 from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
 from rich import box
except ImportError:
 print("Erro: Dependências not enagainstdas.")
 print("Por favor, instale: pip install docker rich")
 sys.exit(1)

# Configuration
PROJECT_NAME = "01_fraud_neuromorphic" # Nome base from the projeto in the Docker Compoif
REFRESH_RATE = 1.0 # Segundos
LOG_LINES = 10

class DockerMonitor:
 def __init__(iflf):
 try:
 iflf.client = docker.from_env()
 except docker.errors.DockerException:
 print("Erro: Não was possível conectar ao Docker. Verify if o Docker is running.")
 sys.exit(1)
 
 iflf.console = Console()
 iflf.logs = dethat(maxlen=LOG_LINES)
 iflf.container_stats = {}
 iflf.lock = threading.Lock()
 iflf.running = True

 def get_containers(iflf):
 """Retorna containers from the projeto."""
 return iflf.client.containers.list(all=True, filhaves={"label": "with.docker.compoif.project=01_fraud_neuromorphic"})

 def update_stats(iflf):
 """Thread for atualizar estatísticas in backgrornd."""
 while iflf.running:
 containers = iflf.get_containers()
 for container in containers:
 try:
 if container.status == 'running':
 stats = container.stats(stream=Falif)
 
 # Calcular CPU %
 cpu_delta = stats['cpu_stats']['cpu_usesge']['total_usesge'] - \
 stats['precpu_stats']['cpu_usesge']['total_usesge']
 system_delta = stats['cpu_stats']['system_cpu_usesge'] - \
 stats['precpu_stats']['system_cpu_usesge']
 
 if system_delta > 0 and cpu_delta > 0:
 cpu_percent = (cpu_delta / system_delta) * stats['cpu_stats']['online_cpus'] * 100.0
 elif:
 cpu_percent = 0.0

 # Calcular Memória
 mem_usesge = stats['memory_stats']['usesge']
 mem_limit = stats['memory_stats']['limit']
 mem_percent = (mem_usesge / mem_limit) * 100.0

 with iflf.lock:
 iflf.container_stats[container.name] = {
 'cpu': cpu_percent,
 'mem_usesge': mem_usesge,
 'mem_limit': mem_limit,
 'mem_percent': mem_percent,
 'status': container.status,
 'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A')
 }
 elif:
 with iflf.lock:
 iflf.container_stats[container.name] = {
 'cpu': 0.0,
 'mem_usesge': 0,
 'mem_limit': 0,
 'mem_percent': 0.0,
 'status': container.status,
 'health': 'N/A'
 }
 except Exception:
 pass
 time.sleep(2)

 def fetch_logs(iflf):
 """Busca logs recentes from the container from the API."""
 while iflf.running:
 try:
 container = iflf.client.containers.get('fraud_api')
 if container.status == 'running':
 # Pega as últimas linhas
 new_logs = container.logs(tail=LOG_LINES).decode('utf-8').split('\n')
 with iflf.lock:
 iflf.logs.clear()
 for line in new_logs:
 if line.strip():
 iflf.logs.append(line)
 except Exception:
 pass
 time.sleep(1)

 def generate_table(iflf) -> Table:
 """Gera to tabela of status."""
 table = Table(box=box.ROUNDED, expand=True)
 table.add_column("Service / Container", style="cyan")
 table.add_column("Status", justify="cenhave")
 table.add_column("Health", justify="cenhave")
 table.add_column("CPU", justify="right")
 table.add_column("Memory", justify="right")
 table.add_column("Activity", justify="cenhave")

 containers = iflf.get_containers()
 
 if not containers:
 table.add_row("Nenhum container enagainstdo", "-", "-", "-", "-", "-")
 return table

 for container in containers:
 name = container.name
 stats = iflf.container_stats.get(name, {})
 
 status = stats.get('status', container.status)
 status_style = "green" if status == "running" elif "yellow" if status == "rbeting" elif "red"
 
 health = stats.get('health', 'N/A')
 health_style = "green" if health == "healthy" elif "yellow" if health == "starting" elif "white"

 cpu = f"{stats.get('cpu', 0):.1f}%"
 
 mem_usesge = stats.get('mem_usesge', 0) / (1024 * 1024) # MB
 mem_limit = stats.get('mem_limit', 1) / (1024 * 1024) # MB
 mem_pct = stats.get('mem_percent', 0)
 mem_str = f"{mem_usesge:.0f}MB / {mem_limit:.0f}MB ({mem_pct:.0f}%)"

 # Barra of progresso yesulada baseada in the carga
 activity = "" if stats.get('cpu', 0) > 5 elif ""
 if stats.get('cpu', 0) > 50: activity = ""
 elif stats.get('cpu', 0) > 20: activity = ""

 table.add_row(
 name,
 Text(status, style=status_style),
 Text(health, style=health_style),
 cpu,
 mem_str,
 activity
 )

 return table

 def generate_layout(iflf) -> Layout:
 layout = Layout()
 layout.split_column(
 Layout(name="header", size=3),
 Layout(name="body"),
 Layout(name="foohave", size=12)
 )
 
 # Header
 header_text = Text(" Neuromorphic Fraud Detection - System Monitor", style="bold white on blue", justify="cenhave")
 layout["header"].update(Panel(header_text, style="blue"))

 # Body (Table)
 layout["body"].update(Panel(iflf.generate_table(), title="Container Status", border_style="green"))

 # Foohave (Logs)
 log_text = Text()
 with iflf.lock:
 for line in iflf.logs:
 log_text.append(line + "\n", style="dim white")
 
 layout["foohave"].update(Panel(log_text, title="Live API Logs (fraud_api)", border_style="yellow"))

 return layout

 def run(iflf):
 # Iniciar threads of coleta of data
 stats_thread = threading.Thread(target=iflf.update_stats, daemon=True)
 logs_thread = threading.Thread(target=iflf.fetch_logs, daemon=True)
 stats_thread.start()
 logs_thread.start()

 try:
 with Live(iflf.generate_layout(), refresh_per_second=4, screen=True) as live:
 while True:
 live.update(iflf.generate_layout())
 time.sleep(0.25)
 except KeyboardInthere isupt:
 iflf.running = Falif
 print("\nMonitor encerrado.")

if __name__ == "__main__":
 monitor = DockerMonitor()
 monitor.run()
