"""GitAgentic: Autonomous GitHub Project Agent CLI."""

import argparse
import logging
import sys

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console(legacy_windows=False)

from src.config import get_config
from src.llm_client import LLMClient
from src.roadmap_engine import RoadmapEngine
from src.coder_agent import CoderAgent
from src.git_committer import GitCommitter
from src.scheduler import RandomizedScheduler

console = Console()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
# Silence chatty libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

def print_banner(config):
    """Display rich title banner."""
    console.print(
        Panel.fit(
            f"[bold magenta]🤖 GitAgentic: Autonomous GitHub Project Agent[/bold magenta]\n"
            f"[cyan]Topic:[/cyan] {config.project_topic}\n"
            f"[cyan]Provider:[/cyan] {config.llm_base_url} ([yellow]{config.llm_model}[/yellow])\n"
            f"[cyan]Workspace:[/cyan] {config.target_project_path}",
            border_style="magenta",
        )
    )

def show_status(config):
    """Show current progress and roadmap steps in a beautiful table."""
    llm = LLMClient(config)
    roadmap_eng = RoadmapEngine(config, llm)
    git = GitCommitter(config)

    roadmap = roadmap_eng.get_or_create_roadmap()
    progress = roadmap_eng.get_progress()

    print_banner(config)

    console.print(
        f"\n[bold]Progress:[/bold] {progress['completed']}/{progress['total']} milestones completed "
        f"([green]{progress['percentage']}%[/green])"
    )

    last_commit = git.get_last_commit()
    if last_commit:
        console.print(f"[bold]Latest Commit:[/bold] [italic green]{last_commit}[/italic green]\n")

    table = Table(title="Project Roadmap Milestones", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Phase", style="cyan", width=22)
    table.add_column("Milestone Title", style="bold")
    table.add_column("Status", width=12)
    table.add_column("Target Files", style="dim")

    for step in roadmap.steps:
        status_badge = (
            "[bold green]✔ DONE[/bold green]"
            if step.status == "completed"
            else "[yellow]⏳ PENDING[/yellow]"
        )
        table.add_row(
            str(step.step_number),
            step.phase,
            step.title,
            status_badge,
            ", ".join(step.target_files[:3]) + ("..." if len(step.target_files) > 3 else ""),
        )

    console.print(table)

def execute_single_step(config) -> bool:
    """Execute a single step from the roadmap, commit, and push."""
    config.validate_keys()

    llm = LLMClient(config)
    roadmap_eng = RoadmapEngine(config, llm)
    coder = CoderAgent(config, llm)
    git = GitCommitter(config)

    next_step = roadmap_eng.get_next_pending_step()
    if not next_step:
        console.print("[bold green]All roadmap milestones have been completed![/bold green]")
        return False

    console.print(
        f"\n[bold cyan]▶ Executing Milestone {next_step.step_number}/{len(roadmap_eng.get_or_create_roadmap().steps)}:[/bold cyan] "
        f"[yellow]{next_step.title}[/yellow] ([italic]{next_step.phase}[/italic])"
    )

    # 1. Coder agent implements the step
    result = coder.execute_step(next_step)
    console.print(f"[green]✔ Code generated. Files modified/created: {len(result.files_written)}[/green]")
    for f in result.files_written:
        console.print(f"  • [blue]{f}[/blue]")

    # 2. Mark step as completed in state BEFORE committing so state/roadmap.json is included in the commit
    roadmap_eng.mark_step_completed(next_step.id)

    # 3. Git stage, commit and push
    commit_hash = git.stage_and_commit(result.commit_message)
    if commit_hash:
        roadmap_eng.mark_step_completed(next_step.id, commit_hash=commit_hash)

    console.print(
        f"[bold green]✔ Milestone #{next_step.step_number} successfully finished and committed![/bold green]\n"
    )
    return True

def main():
    parser = argparse.ArgumentParser(description="GitAgentic: Autonomous GitHub Project Agent")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["daemon", "run", "status", "init", "reset"],
        help="Command to run: 'daemon', 'run', 'status', 'init', or 'reset'",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run 1 roadmap step immediately, commit, and exit (ideal for GitHub Actions / cron).",
    )
    parser.add_argument(
        "--start-daemon",
        action="store_true",
        help="Start background daemon with randomized ~24-hour commit cycles.",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="View project roadmap progress and status.",
    )
    parser.add_argument(
        "--init-roadmap",
        action="store_true",
        help="Initialize or preview the roadmap plan without executing code.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the project state and roadmap.",
    )

    args = parser.parse_args()
    config = get_config()

    if args.command == "reset" or args.reset:
        llm = LLMClient(config)
        eng = RoadmapEngine(config, llm)
        eng.reset()
        console.print("[bold yellow]State reset successfully.[/bold yellow]")
        return

    if args.command == "init" or args.init_roadmap:
        print_banner(config)
        llm = LLMClient(config)
        eng = RoadmapEngine(config, llm)
        eng.get_or_create_roadmap()
        show_status(config)
        return

    if args.command == "status" or args.status:
        show_status(config)
        return

    if args.command == "run" or args.run_once:
        print_banner(config)
        execute_single_step(config)
        return

    if args.command == "daemon" or args.start_daemon:
        print_banner(config)
        config.validate_keys()
        scheduler = RandomizedScheduler(config)
        scheduler.run_daemon(lambda: execute_single_step(config))
        return

    # If no flags or commands passed, default to showing help and status
    parser.print_help()
    console.print("\n")
    show_status(config)

if __name__ == "__main__":
    main()
