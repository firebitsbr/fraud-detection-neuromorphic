#!/usr/bin/env python3
"""
**Descrição:** Monitor de progresso de build Docker.

**Autor:** Mauro Risonho de Paula Assumpção
**Data de Criação:** 5 de Dezembro de 2025
**Licença:** MIT License
**Desenvolvimento:** Desenvolvedor Humano + Desenvolvimento por AI Assitida:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Docker Build Progress Monitor
==============================

Description: Monitora e exibe o progresso do build Docker em tempo real com ETA.

Author: Mauro Risonho de Paula Assumpção
Created: December 5, 2025
License: MIT License

Usage:
    python3 scripts/build_monitor.py
"""

import subprocess
import sys
import time
import re
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

console = Console()

def build_with_progress():
    """Build Docker images with visual progress tracking."""
    
    steps = [
        ("API Service", "docker compose -f config/docker-compose.yml build fraud-api", 120),
        ("Jupyter Lab", "docker compose -f config/docker-compose.yml build jupyter-lab", 120),
        ("Web Interface", "docker compose -f config/docker-compose.yml build web-interface", 60),
    ]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        
        overall = progress.add_task("[cyan]Overall Progress", total=len(steps))
        
        for step_name, command, estimated_time in steps:
            task = progress.add_task(f"[green]{step_name}", total=100)
            
            console.print(f"\n[bold blue]Building {step_name}...[/bold blue]")
            
            # Start build process
            process = subprocess.Popen(
                command.split(),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )
            
            start_time = time.time()
            last_output = ""
            
            # Monitor output
            for line in process.stdout:
                elapsed = time.time() - start_time
                progress_pct = min(95, (elapsed / estimated_time) * 100)
                progress.update(task, completed=progress_pct)
                
                # Show interesting lines
                if any(keyword in line.lower() for keyword in ['downloading', 'extracting', 'building', 'running', 'copying']):
                    last_output = line.strip()
                    console.print(f"[dim]{last_output}[/dim]")
            
            process.wait()
            
            if process.returncode == 0:
                progress.update(task, completed=100)
                progress.advance(overall)
                console.print(f"[bold green]✓ {step_name} complete![/bold green]")
            else:
                console.print(f"[bold red]✗ {step_name} failed![/bold red]")
                return False
        
        console.print("\n[bold green]🎉 All services built successfully![/bold green]")
        return True

if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold white]🐳 Docker Build Monitor[/bold white]\n"
        "[dim]Building Neuromorphic Fraud Detection System[/dim]",
        border_style="blue"
    ))
    
    success = build_with_progress()
    sys.exit(0 if success else 1)
