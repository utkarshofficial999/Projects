"""Git Committer: Manages repository initialization, staging, commits, and pushing."""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from src.config import AppConfig

logger = logging.getLogger("GitAgentic.Git")

class GitCommitter:
    """Handles git operations in the target project folder."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.repo_dir = Path(config.target_project_path).resolve()

    def _run_git(self, args: list[str]) -> Tuple[int, str, str]:
        """Execute a git command within the repo directory."""
        try:
            res = subprocess.run(
                ["git"] + args,
                cwd=str(self.repo_dir),
                capture_output=True,
                text=True,
                check=False,
            )
            return res.returncode, res.stdout.strip(), res.stderr.strip()
        except FileNotFoundError:
            raise RuntimeError("Git is not installed or not in PATH.")

    def ensure_repo(self) -> None:
        """Initialize git repository if it doesn't already exist."""
        self.repo_dir.mkdir(parents=True, exist_ok=True)
        git_folder = self.repo_dir / ".git"
        if not git_folder.exists():
            logger.info(f"Initializing new git repository at {self.repo_dir}")
            code, out, err = self._run_git(["init", "-b", self.config.git_branch])
            if code != 0:
                # Fallback for older git versions without -b flag
                self._run_git(["init"])
                self._run_git(["checkout", "-b", self.config.git_branch])

        # Configure author if specified in config
        if self.config.git_author_name:
            self._run_git(["config", "user.name", self.config.git_author_name])
        if self.config.git_author_email:
            self._run_git(["config", "user.email", self.config.git_author_email])

    def stage_and_commit(self, commit_message: str) -> Optional[str]:
        """Stage all changes, commit them, and return the new commit hash."""
        self.ensure_repo()

        # Check status
        code, out, err = self._run_git(["status", "--porcelain"])
        if not out:
            logger.warning("No changes detected in workspace to commit.")
            return None

        # Stage everything
        self._run_git(["add", "."])

        # Commit
        code, out, err = self._run_git(["commit", "-m", commit_message])
        if code != 0:
            logger.error(f"Git commit failed: {err}")
            return None

        # Get commit hash
        code, commit_hash, _ = self._run_git(["rev-parse", "HEAD"])
        logger.info(f"Committed changes [{commit_hash[:7]}]: {commit_message}")

        # Push if configured
        if self.config.git_auto_push:
            self.push()

        return commit_hash

    def push(self) -> bool:
        """Push commits to remote repository."""
        # Check if remote exists
        code, remotes, _ = self._run_git(["remote"])
        if self.config.git_remote not in remotes.split():
            logger.warning(
                f"Remote '{self.config.git_remote}' not found on repo. "
                "Commit saved locally. Run 'git remote add origin <repo_url>' to enable automatic pushing."
            )
            return False

        # Pull with rebase first to integrate any remote changes safely
        self._run_git(["pull", "--rebase", self.config.git_remote, self.config.git_branch])

        logger.info(f"Pushing to {self.config.git_remote}/{self.config.git_branch}...")
        code, out, err = self._run_git(
            ["push", "-u", self.config.git_remote, self.config.git_branch]
        )
        if code != 0:
            logger.warning(f"Git push warning/error: {err}")
            return False

        logger.info("Successfully pushed commits to GitHub!")
        return True

    def get_last_commit(self) -> Optional[str]:
        """Retrieve latest commit summary."""
        code, out, _ = self._run_git(["log", "-1", "--oneline"])
        return out if code == 0 and out else None
