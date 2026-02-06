#!/usr/bin/env python3
import glob
import json
import os
from pathlib import Path


def get_latest_snapshot(snapshot_dir):
    """Find the most recent snapshot JSON file."""
    files = glob.glob(str(snapshot_dir / "*.json"))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def format_recovery_doc(snapshot):
    """Format the snapshot data into a Markdown recovery document."""

    content = [
        "# 🚨 HANDOFF CONTEXT RECOVERY 🚨",
        "",
        f"**Date**: {snapshot['timestamp']}",
        f"**Trigger**: {snapshot['trigger']}",
        f"**Git Branch**: {snapshot['git']['branch']} ({snapshot['git']['commit'][:7]})",
        "",
        "## 📝 Summary",
        snapshot["summary"],
        "",
        "## 📂 Project Structure (Simplified)",
        "```",
    ]

    # Add first 50 lines of tree to avoid bloat
    tree = snapshot.get("tree", [])
    content.extend(tree[:50])
    if len(tree) > 50:
        content.append(f"... ({len(tree) - 50} more files)")
    content.append("```")

    content.extend(
        [
            "",
            "## 🟢 Current Project Status",
        ]
    )

    if snapshot["files"].get("PROJECT_STATUS.md"):
        content.append(snapshot["files"]["PROJECT_STATUS.md"])
    else:
        content.append("*PROJECT_STATUS.md not found in snapshot*")

    content.extend(
        [
            "",
            "## 📘 README (Excerpt)",
        ]
    )

    if snapshot["files"].get("README.md"):
        readme_lines = snapshot["files"]["README.md"].splitlines()
        content.extend(readme_lines[:50])
        if len(readme_lines) > 50:
            content.append(f"\n... ({len(readme_lines) - 50} more lines)")

    return "\n".join(content)


def recover_context():
    """Generate the recovery context file."""
    project_root = Path(__file__).resolve().parent.parent
    snapshot_dir = project_root / ".immunos" / "snapshots"
    recovery_dir = project_root / ".immunos" / "recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)

    latest_snapshot_path = get_latest_snapshot(snapshot_dir)
    if not latest_snapshot_path:
        print("No snapshots found in .immunos/snapshots/")
        return

    print(f"Loading snapshot: {latest_snapshot_path}")

    try:
        with open(latest_snapshot_path, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)

        recovery_content = format_recovery_doc(snapshot_data)

        output_path = recovery_dir / "CONTEXT_RECOVERY.md"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(recovery_content)

        print(f"Recovery context generated: {output_path}")

    except Exception as e:
        print(f"Error generating recovery context: {e}")


if __name__ == "__main__":
    recover_context()
