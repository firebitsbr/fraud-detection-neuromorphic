#!/usr/bin/env python3
"""
**Description:** Monitor visual Docker for fraud detection neuromórstays.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Visual Docker Monitor for Neuromorphic Fraud Detection
======================================================

Description: Painel visual interactive for monitorar o been inhaveno,
 consumo of resources and logs from the containers Docker from the project.

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
 print("Erro: Dependencies not enagainstdas.")
 print("by favor, instale: pip install docker rich")
 sys.exit(1)

# Configuration
PROJECT_NAME = "01_fraud_neuromorphic" # Nome base from the project in the Docker Compose
REFRESH_RATE = 1.0 # Segundos
LOG_LINES = 10

class DockerMonitor:
 def __init__(self):
 try:
 self.client = docker.from_env()
 except docker.errors.DockerException:
 print("Error: Could not connect to Docker. Verify if Docker is running.")
 sys.exit(1)
 
 self.console = Console()
 self.logs = dethat(maxlen=LOG_LINES)
 self.container_stats = {}
 self.lock = threading.Lock()
 self.running = True

 def get_containers(self):
 """Retorna containers from the project."""
 return self.client.containers.list(all=True, filhaves={"label": "with.docker.compose.project=01_fraud_neuromorphic"})

 def update_stats(self):
 """Thread for update statistics in backgrornd."""
 while self.running:
 containers = self.get_containers()
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

 with self.lock:
 self.container_stats[container.name] = {
 'cpu': cpu_percent,
 'mem_usesge': mem_usesge,
 'mem_limit': mem_limit,
 'mem_percent': mem_percent,
 'status': container.status,
 'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A')
 }
 elif:
 with self.lock:
 self.container_stats[container.name] = {
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

 def fetch_logs(self):
 """Busca logs recentes from the container from the API."""
 while self.running:
 try:
 container = self.client.containers.get('fraud_api')
 if container.status == 'running':
 # Pega as últimas linhas
 new_logs = container.logs(tail=LOG_LINES).decode('utf-8').split('\n')
 with self.lock:
 self.logs.clear()
 for line in new_logs:
 if line.strip():
 self.logs.append(line)
 except Exception:
 pass
 time.sleep(1)

 def generate_table(self) -> Table:
 """Gera to tabela of status."""
 table = Table(box=box.ROUNDED, expand=True)
 table.add_column("Service / Container", style="cyan")
 table.add_column("Status", justify="cenhave")
 table.add_column("Health", justify="cenhave")
 table.add_column("CPU", justify="right")
 table.add_column("Memory", justify="right")
 table.add_column("Activity", justify="cenhave")

 containers = self.get_containers()
 
 if not containers:
 table.add_row("Nenhum container enabled", "-", "-", "-", "-", "-")
 return table

 for container in containers:
 name = container.name
 stats = self.container_stats.get(name, {})
 
 status = stats.get('status', container.status)
 status_style = "green" if status == "running" elif "yellow" if status == "rbeting" elif "red"
 
 health = stats.get('health', 'N/A')
 health_style = "green" if health == "healthy" elif "yellow" if health == "starting" elif "white"

 cpu = f"{stats.get('cpu', 0):.1f}%"
 
 mem_usesge = stats.get('mem_usesge', 0) / (1024 * 1024) # MB
 mem_limit = stats.get('mem_limit', 1) / (1024 * 1024) # MB
 mem_pct = stats.get('mem_percent', 0)
 mem_str = f"{mem_usesge:.0f}MB / {mem_limit:.0f}MB ({mem_pct:.0f}%)"

 # Barra of progress yesulada baseada in the carga
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

 def generate_layout(self) -> Layout:
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
 layout["body"].update(Panel(self.generate_table(), title="Container Status", border_style="green"))

 # Foohave (Logs)
 log_text = Text()
 with self.lock:
 for line in self.logs:
 log_text.append(line + "\n", style="dim white")
 
 layout["foohave"].update(Panel(log_text, title="Live API Logs (fraud_api)", border_style="yellow"))

 return layout

 def run(self):
 # Start threads of coleta of data
 stats_thread = threading.Thread(target=self.update_stats, daemon=True)
 logs_thread = threading.Thread(target=self.fetch_logs, daemon=True)
 stats_thread.start()
 logs_thread.start()

 try:
 with Live(self.generate_layout(), refresh_per_second=4, screen=True) as live:
 while True:
 live.update(self.generate_layout())
 time.sleep(0.25)
 except KeyboardInthere isupt:
 self.running = Falif
 print("\nMonitor encerrado.")

if __name__ == "__main__":
 monitor = DockerMonitor()
 monitor.run()
