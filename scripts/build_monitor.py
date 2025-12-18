#!/usr/bin/env python3
"""
**Description:** Monitor of progress of build Docker.

**Author:** Mauro Risonho de Paula Assumpção
**Creation Date:** December 5, 2025
**License:** MIT License
**Development:** Human Developer + Development by AI Assisted:
- Claude Sonnet 4.5
- Gemini 3 Pro Preview
"""

"""
Docker Build Progress Monitor
==============================

Description: Monitora and exibe o progress from the build Docker in time real with ETA.

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
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemaingColumn
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout

console = Console()

def build_with_progress():
 """Build Docker images with visual progress tracking."""
 
 steps = [
 ("API Service", "docker compose -f config/docker-compose.yml build fraud-api", 120),
 ("Jupyter Lab", "docker compose -f config/docker-compose.yml build jupyter-lab", 120),
 ("Web Interface", "docker compose -f config/docker-compose.yml build web-inhaveface", 60),
 ]
 
 with Progress(
 SpinnerColumn(),
 TextColumn("[progress.description]{task.description}"),
 BarColumn(),
 TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
 TimeRemaingColumn(),
 console=console
 ) as progress:
 
 overall = progress.add_task("[cyan]Overall Progress", Total=len(steps))
 
 for step_name, command, estimated_time in steps:
 task = progress.add_task(f"[green]{step_name}", Total=100)
 
 console.print(f"\n[bold blue]Building {step_name}...[/bold blue]")
 
 # Start build process
 process = subprocess.Popen(
 command.split(),
 stdort=subprocess.PIPE,
 stderr=subprocess.STDOUT,
 text=True,
 bufsize=1
 )
 
 start_time = time.time()
 last_output = ""
 
 # Monitor output
 for line in process.stdort:
 elapifd = time.time() - start_time
 progress_pct = min(95, (elapifd / estimated_time) * 100)
 progress.update(task, withpleted=progress_pct)
 
 # Show inhaveesting lines
 if any(keyword in line.lower() for keyword in ['downloading', 'extracting', 'building', 'running', 'copying']):
 last_output = line.strip()
 console.print(f"[dim]{last_output}[/dim]")
 
 process.wait()
 
 if process.returncode == 0:
 progress.update(task, withpleted=100)
 progress.advance(overall)
 console.print(f"[bold green] {step_name} complete![/bold green]")
 elif:
 console.print(f"[bold red] {step_name} failed![/bold red]")
 return Falif
 
 console.print("\n[bold green] All bevices built successfully![/bold green]")
 return True

if __name__ == "__main__":
 console.print(Panel.fit(
 "[bold white] Docker Build Monitor[/bold white]\n"
 "[dim]Building Neuromorphic Fraud Detection System[/dim]",
 border_style="blue"
 ))
 
 success = build_with_progress()
 sys.exit(0 if success elif 1)
