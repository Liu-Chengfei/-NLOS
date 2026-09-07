"""Embed anchor_layout.json into per-seq manifest entries (bare seq_id keys)."""
import json
from pathlib import Path

for seed in range(10):
    seed_dir = Path(f"data/processed/sim_e9_paper/seed{seed}")
    pm_path = seed_dir / "prepare_manifest.json"
    data = json.loads(pm_path.read_text())
    raw_dir = Path(f"data/raw/sim_e9_paper/seed{seed}")
    for sid, entry in data["sequences"].items():
        seq_dir = raw_dir / sid
        al_path = seq_dir / "anchor_layout.json"
        if al_path.exists():
            entry["anchor_layout"] = json.loads(al_path.read_text())
        if "seq_dir" not in entry:
            entry["seq_dir"] = str(seq_dir).replace("\\", "/")
    pm_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
print("Done embedding anchor_layout for 10 seeds")