#!/usr/bin/env python3
"""
**Descrição:** Monitor visual Docker para detecção de fraude neuromórfica.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Visual Docker Monitor for Neuromorphic Fraud Detection
======================================================

Description: Painel visual interativo para monitorar o estado interno,
             consumo de recursos e logs dos containers Docker do projeto.

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
from collections import deque

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
    print("Erro: Dependências não encontradas.")
    print("Por favor, instale: pip install docker rich")
    sys.exit(1)

# Configuração
PROJECT_NAME = "01_fraud_neuromorphic"  # Nome base do projeto no Docker Compose
REFRESH_RATE = 1.0  # Segundos
LOG_LINES = 10

class DockerMonitor:
    def __init__(self):
        try:
            self.client = docker.from_env()
        except docker.errors.DockerException:
            print("Erro: Não foi possível conectar ao Docker. Verifique se o Docker está rodando.")
            sys.exit(1)
        
        self.console = Console()
        self.logs = deque(maxlen=LOG_LINES)
        self.container_stats = {}
        self.lock = threading.Lock()
        self.running = True

    def get_containers(self):
        """Retorna containers do projeto."""
        return self.client.containers.list(all=True, filters={"label": "com.docker.compose.project=01_fraud_neuromorphic"})

    def update_stats(self):
        """Thread para atualizar estatísticas em background."""
        while self.running:
            containers = self.get_containers()
            for container in containers:
                try:
                    if container.status == 'running':
                        stats = container.stats(stream=False)
                        
                        # Calcular CPU %
                        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                                    stats['precpu_stats']['cpu_usage']['total_usage']
                        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                                       stats['precpu_stats']['system_cpu_usage']
                        
                        if system_delta > 0 and cpu_delta > 0:
                            cpu_percent = (cpu_delta / system_delta) * stats['cpu_stats']['online_cpus'] * 100.0
                        else:
                            cpu_percent = 0.0

                        # Calcular Memória
                        mem_usage = stats['memory_stats']['usage']
                        mem_limit = stats['memory_stats']['limit']
                        mem_percent = (mem_usage / mem_limit) * 100.0

                        with self.lock:
                            self.container_stats[container.name] = {
                                'cpu': cpu_percent,
                                'mem_usage': mem_usage,
                                'mem_limit': mem_limit,
                                'mem_percent': mem_percent,
                                'status': container.status,
                                'health': container.attrs.get('State', {}).get('Health', {}).get('Status', 'N/A')
                            }
                    else:
                        with self.lock:
                            self.container_stats[container.name] = {
                                'cpu': 0.0,
                                'mem_usage': 0,
                                'mem_limit': 0,
                                'mem_percent': 0.0,
                                'status': container.status,
                                'health': 'N/A'
                            }
                except Exception:
                    pass
            time.sleep(2)

    def fetch_logs(self):
        """Busca logs recentes do container da API."""
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
        """Gera a tabela de status."""
        table = Table(box=box.ROUNDED, expand=True)
        table.add_column("Service / Container", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Health", justify="center")
        table.add_column("CPU", justify="right")
        table.add_column("Memory", justify="right")
        table.add_column("Activity", justify="center")

        containers = self.get_containers()
        
        if not containers:
            table.add_row("Nenhum container encontrado", "-", "-", "-", "-", "-")
            return table

        for container in containers:
            name = container.name
            stats = self.container_stats.get(name, {})
            
            status = stats.get('status', container.status)
            status_style = "green" if status == "running" else "yellow" if status == "restarting" else "red"
            
            health = stats.get('health', 'N/A')
            health_style = "green" if health == "healthy" else "yellow" if health == "starting" else "white"

            cpu = f"{stats.get('cpu', 0):.1f}%"
            
            mem_usage = stats.get('mem_usage', 0) / (1024 * 1024) # MB
            mem_limit = stats.get('mem_limit', 1) / (1024 * 1024) # MB
            mem_pct = stats.get('mem_percent', 0)
            mem_str = f"{mem_usage:.0f}MB / {mem_limit:.0f}MB ({mem_pct:.0f}%)"

            # Barra de progresso simulada baseada na carga
            activity = "🟢" if stats.get('cpu', 0) > 5 else "⚪"
            if stats.get('cpu', 0) > 50: activity = "🔴"
            elif stats.get('cpu', 0) > 20: activity = "🟡"

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
            Layout(name="footer", size=12)
        )
        
        # Header
        header_text = Text("🧠 Neuromorphic Fraud Detection - System Monitor", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, style="blue"))

        # Body (Table)
        layout["body"].update(Panel(self.generate_table(), title="Container Status", border_style="green"))

        # Footer (Logs)
        log_text = Text()
        with self.lock:
            for line in self.logs:
                log_text.append(line + "\n", style="dim white")
        
        layout["footer"].update(Panel(log_text, title="Live API Logs (fraud_api)", border_style="yellow"))

        return layout

    def run(self):
        # Iniciar threads de coleta de dados
        stats_thread = threading.Thread(target=self.update_stats, daemon=True)
        logs_thread = threading.Thread(target=self.fetch_logs, daemon=True)
        stats_thread.start()
        logs_thread.start()

        try:
            with Live(self.generate_layout(), refresh_per_second=4, screen=True) as live:
                while True:
                    live.update(self.generate_layout())
                    time.sleep(0.25)
        except KeyboardInterrupt:
            self.running = False
            print("\nMonitor encerrado.")

if __name__ == "__main__":
    monitor = DockerMonitor()
    monitor.run()
