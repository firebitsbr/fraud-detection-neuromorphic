#!/usr/bin/env python3
"""
Docker Build Monitor - Real-time progress tracking with ETA
Monitors Docker build process with progress bars and time estimation

Author: Mauro Risonho de Paula Assumpção
Email: mauro.risonho@gmail.com
LinkedIn: linkedin.com/in/maurorisonho
GitHub: github.com/maurorisonho
"""

import subprocess
import sys
import re
import time
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn
    )
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
except ImportError:
    print("Installing required package: rich")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "rich"])
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeRemainingColumn,
        TimeElapsedColumn
    )
    from rich.live import Live
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout

console = Console()


class DockerBuildMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.stages = {}
        self.current_stage = None
        self.total_stages = 0
        self.completed_stages = 0
        
    def parse_build_output(self, line):
        """Parse Docker build output line"""
        # Match stage progress: [builder 3/13]
        stage_match = re.search(r'\[(\w+)\s+(\d+)/(\d+)\]', line)
        if stage_match:
            stage_name = stage_match.group(1)
            current = int(stage_match.group(2))
            total = int(stage_match.group(3))
            return {
                'type': 'stage',
                'stage': stage_name,
                'current': current,
                'total': total,
                'line': line
            }
        
        # Match download progress
        download_match = re.search(r'Downloading.*\[=+>?\s*\]\s+(\d+\.?\d*\w+)/(\d+\.?\d*\w+)', line)
        if download_match:
            return {
                'type': 'download',
                'current': download_match.group(1),
                'total': download_match.group(2),
                'line': line
            }
        
        # Match layer extraction
        if 'extracting' in line.lower():
            return {'type': 'extract', 'line': line}
        
        # Match CACHED layers (instant completion)
        if 'CACHED' in line:
            return {'type': 'cached', 'line': line}
        
        # Match errors
        if 'ERROR' in line:
            return {'type': 'error', 'line': line}
        
        return {'type': 'info', 'line': line}

    def estimate_eta(self):
        """Estimate time remaining"""
        if self.completed_stages == 0:
            return "Calculating..."
        
        elapsed = time.time() - self.start_time
        avg_time_per_stage = elapsed / self.completed_stages
        remaining_stages = self.total_stages - self.completed_stages
        eta_seconds = avg_time_per_stage * remaining_stages
        
        return str(timedelta(seconds=int(eta_seconds)))

    def create_status_table(self):
        """Create status table"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan", width=20)
        table.add_column("Value", style="green")
        
        elapsed = str(timedelta(seconds=int(time.time() - self.start_time)))
        
        table.add_row("Elapsed Time", elapsed)
        table.add_row("Completed Stages", f"{self.completed_stages}/{self.total_stages}")
        table.add_row("Current Stage", self.current_stage or "Initializing...")
        table.add_row("ETA", self.estimate_eta())
        
        return table

    def run_build(self, dockerfile="Dockerfile", tag="fraud-detection-api:ubuntu24.04", use_sudo=False):
        """Run Docker build with monitoring"""
        cmd = ["docker", "build", "-t", tag, "-f", dockerfile, "--progress=plain", "."]
        
        # Add sudo if needed
        if use_sudo:
            cmd.insert(0, "sudo")
        
        console.print(f"\n[bold blue]🚀 Starting Docker Build[/bold blue]")
        console.print(f"[cyan]Command: {' '.join(cmd)}[/cyan]\n")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            # Create main build task
            build_task = progress.add_task(
                "[cyan]Building Docker image...",
                total=100
            )
            
            # Track individual stages
            stage_tasks = {}
            
            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    bufsize=1
                )
                
                # Type check for stdout
                if process.stdout is None:
                    console.print("[red]Error: Failed to capture build output[/red]")
                    return False
                
                for line in process.stdout:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parsed = self.parse_build_output(line)
                    
                    if parsed['type'] == 'stage':
                        stage = parsed['stage']
                        current = parsed['current']
                        total = parsed['total']
                        
                        self.current_stage = f"{stage} [{current}/{total}]"
                        self.total_stages = max(self.total_stages, total)
                        
                        # Create or update stage task
                        if stage not in stage_tasks:
                            stage_tasks[stage] = progress.add_task(
                                f"[yellow]Stage: {stage}",
                                total=total
                            )
                        
                        progress.update(
                            stage_tasks[stage],
                            completed=current,
                            description=f"[yellow]{stage} - {line[:60]}..."
                        )
                        
                        # Update main progress
                        overall_progress = (current / total) * 100
                        progress.update(build_task, completed=overall_progress)
                        
                        if current == total:
                            self.completed_stages += 1
                    
                    elif parsed['type'] == 'cached':
                        console.print(f"[green]✓ {line}[/green]")
                        self.completed_stages += 1
                    
                    elif parsed['type'] == 'error':
                        console.print(f"[red]✗ {line}[/red]")
                    
                    elif parsed['type'] == 'download':
                        console.print(f"[blue]⬇ {line}[/blue]")
                    
                    elif parsed['type'] == 'extract':
                        console.print(f"[magenta]📦 {line}[/magenta]")
                
                process.wait()
                
                if process.returncode == 0:
                    progress.update(build_task, completed=100)
                    console.print("\n[bold green]✅ Build completed successfully![/bold green]")
                    
                    # Show final stats
                    self.show_final_stats()
                    return True
                else:
                    console.print("\n[bold red]❌ Build failed![/bold red]")
                    return False
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]⚠️  Build interrupted by user[/yellow]")
                process.terminate()
                return False
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")
                return False

    def show_final_stats(self):
        """Show final build statistics"""
        elapsed = time.time() - self.start_time
        
        stats_table = Table(show_header=False, box=None)
        stats_table.add_column("", style="cyan", width=20)
        stats_table.add_column("", style="green")
        
        stats_table.add_row("Total Time", f"{timedelta(seconds=int(elapsed))}")
        stats_table.add_row("Stages Completed", str(self.completed_stages))
        stats_table.add_row("Avg Time/Stage", f"{elapsed/max(self.completed_stages, 1):.1f}s")
        
        console.print("\n")
        console.print(Panel(stats_table, title="[bold]Build Statistics[/bold]", border_style="green"))


class DockerContainerMonitor:
    """Monitor running Docker containers"""
    
    def __init__(self, project_name="neuromorphic-fraud-detection"):
        self.project_name = project_name
        
    def get_container_stats(self, container_name):
        """Get container statistics"""
        try:
            result = subprocess.run(
                ["docker", "stats", "--no-stream", "--format", 
                 "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}|{{.BlockIO}}", 
                 container_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split('|')
                return {
                    'cpu': parts[0],
                    'memory': parts[1],
                    'network': parts[2],
                    'disk': parts[3]
                }
        except Exception:
            pass
        
        return None
    
    def get_container_health(self, container_name):
        """Get container health status"""
        try:
            result = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Health.Status}}", container_name],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                health = result.stdout.strip()
                if health == "healthy":
                    return "✓ Healthy", "green"
                elif health == "unhealthy":
                    return "✗ Unhealthy", "red"
                elif health == "starting":
                    return "⟳ Starting", "yellow"
                elif health == "none" or not health:
                    return "− No check", "blue"
        except Exception:
            pass
        
        return "? Unknown", "white"
    
    def monitor_containers(self):
        """Monitor containers in real-time"""
        services = [
            "fraud-api", "jupyter-lab", "web-interface", 
            "redis", "prometheus", "grafana"
        ]
        
        console.print("\n[bold cyan]🐳 Docker Container Monitor[/bold cyan]\n")
        console.print("[yellow]Press Ctrl+C to exit[/yellow]\n")
        
        try:
            while True:
                table = Table(show_header=True, header_style="bold magenta")
                table.add_column("Service", style="cyan", width=20)
                table.add_column("Status", width=15)
                table.add_column("Health", width=15)
                table.add_column("CPU", width=10)
                table.add_column("Memory", width=20)
                table.add_column("Network", width=20)
                
                for service in services:
                    container = f"{self.project_name}-{service}-1"
                    
                    # Check if running
                    try:
                        result = subprocess.run(
                            ["docker", "ps", "--filter", f"name={container}", 
                             "--format", "{{.Status}}"],
                            capture_output=True,
                            text=True,
                            timeout=5
                        )
                        
                        if result.stdout.strip():
                            status = "[green]● Running[/green]"
                            
                            # Get health
                            health_text, health_color = self.get_container_health(container)
                            health = f"[{health_color}]{health_text}[/{health_color}]"
                            
                            # Get stats
                            stats = self.get_container_stats(container)
                            if stats:
                                table.add_row(
                                    service,
                                    status,
                                    health,
                                    stats['cpu'],
                                    stats['memory'],
                                    stats['network']
                                )
                            else:
                                table.add_row(service, status, health, "N/A", "N/A", "N/A")
                        else:
                            table.add_row(
                                service,
                                "[red]● Stopped[/red]",
                                "[red]−[/red]",
                                "−", "−", "−"
                            )
                    except Exception:
                        table.add_row(
                            service,
                            "[yellow]? Error[/yellow]",
                            "[yellow]?[/yellow]",
                            "−", "−", "−"
                        )
                
                console.clear()
                console.print(f"\n[bold cyan]🐳 Container Monitor[/bold cyan] - {datetime.now().strftime('%H:%M:%S')}\n")
                console.print(table)
                console.print("\n[yellow]Refreshing every 3 seconds... (Ctrl+C to exit)[/yellow]")
                
                time.sleep(3)
                
        except KeyboardInterrupt:
            console.print("\n[green]Monitor stopped.[/green]")


def check_docker_permission():
    """Check if we have Docker permissions"""
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception:
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        console.print("[bold]Docker Monitor - Usage:[/bold]\n")
        console.print("  [cyan]Build monitoring:[/cyan]")
        console.print("    python build-monitor.py build [dockerfile] [tag]")
        console.print("    sudo python build-monitor.py build [dockerfile] [tag]\n")
        console.print("  [cyan]Container monitoring:[/cyan]")
        console.print("    python build-monitor.py monitor [project-name]\n")
        console.print("[bold]Examples:[/bold]")
        console.print("  python build-monitor.py build")
        console.print("  sudo python build-monitor.py build")
        console.print("  python build-monitor.py build Dockerfile fraud-detection-api:latest")
        console.print("  python build-monitor.py monitor")
        console.print("  python build-monitor.py monitor neuromorphic-fraud-detection")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "build":
        # Check Docker permissions
        use_sudo = False
        if not check_docker_permission():
            console.print("[yellow]⚠️  Docker permission denied. Trying with sudo...[/yellow]\n")
            use_sudo = True
        
        dockerfile = sys.argv[2] if len(sys.argv) > 2 else "Dockerfile"
        tag = sys.argv[3] if len(sys.argv) > 3 else "fraud-detection-api:ubuntu24.04"
        
        monitor = DockerBuildMonitor()
        success = monitor.run_build(dockerfile, tag, use_sudo)
        sys.exit(0 if success else 1)
        
    elif command == "monitor":
        project_name = sys.argv[2] if len(sys.argv) > 2 else "neuromorphic-fraud-detection"
        
        monitor = DockerContainerMonitor(project_name)
        monitor.monitor_containers()
        
    else:
        console.print(f"[red]Unknown command: {command}[/red]")
        console.print("Use: build or monitor")
        sys.exit(1)


if __name__ == "__main__":
    main()
