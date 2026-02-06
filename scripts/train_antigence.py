import csv
import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from immunos_mcp.agents.bcell_agent import BCellAgent
from immunos_mcp.agents.nk_cell_agent import NKCellAgent
from immunos_mcp.core.antigen import Antigen


def train_network_nk():
    print("Training Network NK-Cell Agent...")
    csv_path = "data/training/network/synthetic_network.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    antigens = []
    features = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["label"] == "normal":
                # Convert categorical to simple floats for demo
                feat = [
                    float(row["duration"]),
                    1.0 if row["protocol_type"] == "tcp" else 0.0,
                    float(row["src_bytes"]) / 10000.0,
                    float(row["dst_bytes"]) / 10000.0,
                ]
                # Pad to 20-dim if needed by agent
                feat += [0.0] * (20 - len(feat))
                antigens.append(
                    Antigen.from_dict({"data": row, "features": feat}, class_label="normal")
                )
                features.append(feat)

    agent = NKCellAgent(agent_name="network_nk_synthetic", mode="feature")
    agent.train_on_features(antigens, features)

    os.makedirs(".immunos/models", exist_ok=True)
    agent.save_state(".immunos/models/network_nk_synthetic.json")
    print("Saved Network NK Agent to .immunos/models/network_nk_synthetic.json")


def train_security_bcell():
    print("Training Security B-Cell Agent...")
    csv_path = "data/training/security/synthetic_security.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    antigens = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            antigen = Antigen.from_dict(
                {"data": row["content"], "class_label": row["label"]}, class_label=row["label"]
            )
            antigens.append(antigen)

    agent = BCellAgent(agent_name="security_bcell_synthetic")
    agent.train(antigens)

    os.makedirs(".immunos/models", exist_ok=True)
    agent.save_state(".immunos/models/security_bcell_synthetic.json")
    print("Saved Security B-Cell Agent to .immunos/models/security_bcell_synthetic.json")


if __name__ == "__main__":
    train_network_nk()
    print("-" * 30)
    train_security_bcell()
