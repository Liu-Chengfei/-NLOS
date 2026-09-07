"""Fix manifest: dataset_manifest.sequences must stay as list (LSTM derivation),
top-level sequences must be dict (LSTM lookup)."""
import json
from pathlib import Path

for seed in range(10):
    seed_dir = Path(f"data/processed/sim_e9_paper/seed{seed}")
    pm_path = seed_dir / "prepare_manifest.json"
    data = json.loads(pm_path.read_text())

    raw_dir = Path(f"data/raw/sim_e9_paper/seed{seed}")
    dm_seqs = data.get("dataset_manifest", {}).get("sequences", [])
    # Ensure dataset_manifest.sequences is a list of dicts
    if not isinstance(dm_seqs, list):
        dm_seqs = list(dm_seqs.values()) if isinstance(dm_seqs, dict) else []
    bare_dict = {}
    for s in dm_seqs:
        if not isinstance(s, dict) or "seq_id" not in s:
            continue
        sid = s["seq_id"]
        bare_dict[sid] = s
        # Embed anchor_layout if missing
        if "anchor_layout" not in s:
            al_path = raw_dir / sid / "anchor_layout.json"
            if al_path.exists():
                s["anchor_layout"] = json.loads(al_path.read_text())
            seq_dir = raw_dir / sid
            s.setdefault("seq_dir", str(seq_dir).replace("\\", "/"))

    data["dataset_manifest"]["sequences"] = dm_seqs
    data["sequences"] = bare_dict
    pm_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))

# Verify
data0 = json.load(open("data/processed/sim_e9_paper/seed0/prepare_manifest.json"))
print("dataset_manifest.sequences type:", type(data0["dataset_manifest"]["sequences"]).__name__)
print("sequences type:", type(data0["sequences"]).__name__)
print("count:", len(data0["sequences"]))
ex = list(data0["sequences"].values())[0]
print("first seq keys (subset):", list(ex.keys())[:8])
print("first seq has anchor_layout:", "anchor_layout" in ex)