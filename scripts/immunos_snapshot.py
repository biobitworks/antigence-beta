#!/usr/bin/env python3
import argparse
import datetime
import json
import os
import subprocess
from pathlib import Path


def get_git_info():
    """Get current git branch and last commit hash."""
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return {"branch": branch, "commit": commit}
    except subprocess.CalledProcessError:
        return {"branch": "unknown", "commit": "unknown"}


def get_file_content(path):
    """Read file content if it exists."""
    p = Path(path)
    if p.exists() and p.is_file():
        try:
            return p.read_text(encoding="utf-8")
        except Exception as e:
            return f"Error reading file: {e}"
    return None


def get_tree_structure(root_dir, max_depth=3):
    """Generate a simplified tree structure."""
    tree = []
    root = Path(root_dir)

    # Files to exclude from tree
    excludes = {
        ".git",
        "__pycache__",
        ".DS_Store",
        "venv",
        ".venv",
        ".pytest_cache",
        "node_modules",
    }

    for path in sorted(root.rglob("*")):
        # Skip excluded directories
        if any(part in excludes for part in path.parts):
            continue

        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue

        rel_path = str(path.relative_to(root))
        if path.is_dir():
            tree.append(f"{rel_path}/")
        else:
            tree.append(rel_path)

    return tree


def create_snapshot(args):
    """Create a snapshot of the current project state."""
    project_root = Path(__file__).resolve().parent.parent
    snapshot_dir = project_root / ".immunos" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    snapshot = {
        "timestamp": datetime.datetime.now().isoformat(),
        "trigger": args.trigger,
        "summary": args.summary,
        "git": get_git_info(),
        "files": {
            "PROJECT_STATUS.md": get_file_content(project_root / "PROJECT_STATUS.md"),
            "README.md": get_file_content(project_root / "README.md"),
        },
        "tree": get_tree_structure(project_root),
    }

    filename = f"{timestamp}_snapshot.json"
    output_path = snapshot_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    print(f"Snapshot created: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a project snapshot.")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Create command
    create_parser = subparsers.add_parser("create", help="Create a new snapshot")
    create_parser.add_argument("--trigger", default="manual", help="Trigger source (manual, auto)")
    create_parser.add_argument("--summary", required=True, help="Snapshot summary")

    args = parser.parse_args()

    if args.command == "create":
        create_snapshot(args)
    else:
        parser.print_help()
