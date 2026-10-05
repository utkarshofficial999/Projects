"""Scheduler: Handles randomized 24-hour execution intervals."""

import logging
import random
import time
from datetime import datetime, timedelta
from typing import Callable
from rich.console import Console
from rich.table import Table
from src.config import AppConfig

logger = logging.getLogger("GitAgentic.Scheduler")
console = Console(legacy_windows=False)

class RandomizedScheduler:
    """Manages randomized daily execution cycles."""

    def __init__(self, config: AppConfig):
        self.config = config

    def calculate_next_interval_seconds(self) -> float:
        """Calculate random seconds between min and max hours."""
        min_sec = self.config.min_hours_between_commits * 3600
        max_sec = self.config.max_hours_between_commits * 3600
        return random.uniform(min_sec, max_sec)

    def run_daemon(self, task_fn: Callable[[], bool]) -> None:
        """Run continuous loop with randomized intervals.
        
        Args:
            task_fn: Function to execute each cycle. Returns True if there are more steps, False if finished.
        """
        console.print("[bold cyan]🚀 GitAgentic Randomized 24-Hour Daemon Started![/bold cyan]")
        console.print(
            f"[dim]Commit interval configured between {self.config.min_hours_between_commits}h "
            f"and {self.config.max_hours_between_commits}h.[/dim]\n"
        )

        cycle_count = 1
        while True:
            console.print(f"\n[bold green]=== Starting Cycle #{cycle_count} ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===[/bold green]")
            
            try:
                has_more = task_fn()
                if not has_more:
                    console.print("[bold gold1]🎉 All roadmap steps completed! Project is complete.[/bold gold1]")
                    break
            except Exception as e:
                logger.error(f"Error during cycle #{cycle_count}: {e}", exc_info=True)
                console.print(f"[bold red]Cycle encountered an error: {e}[/bold red]")

            # Calculate next randomized interval
            wait_seconds = self.calculate_next_interval_seconds()
            hours = wait_seconds / 3600
            next_run = datetime.now() + timedelta(seconds=wait_seconds)

            console.print(f"\n[cyan]Next commit scheduled in ~[bold]{hours:.1f} hours[/bold] at [bold]{next_run.strftime('%Y-%m-%d %I:%M %p')}[/bold][/cyan]")
            console.print("[dim]Press Ctrl+C to stop daemon anytime.[/dim]\n")

            # Sleep in intervals of 10s to allow graceful shutdown
            end_time = time.time() + wait_seconds
            try:
                while time.time() < end_time:
                    remaining = int(end_time - time.time())
                    rem_h = remaining // 3600
                    rem_m = (remaining % 3600) // 60
                    # Update status in terminal every minute or so
                    time.sleep(min(30, max(1, remaining)))
            except KeyboardInterrupt:
                console.print("\n[yellow]Daemon stopped by user (Ctrl+C).[/yellow]")
                break

            cycle_count += 1
