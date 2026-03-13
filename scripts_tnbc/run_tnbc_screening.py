from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path("data/TNBC_GSE210616").resolve()
SAMPLES = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("GSM_")])


def run_cmd(cmd: list[str]) -> int:
    print("\n" + "=" * 80)
    print("RUNNING:", " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd)
    return result.returncode


def main() -> None:
    print(f"Found {len(SAMPLES)} sample folders under {ROOT}")
    for sample_dir in SAMPLES:
        sample_id = sample_dir.name

        rc1 = run_cmd([
            "python", "-m", "scripts_tnbc.step1_tnbc_map",
            "--sample_dir", str(sample_dir)
        ])
        if rc1 != 0:
            print(f"[FAIL] Step 1 failed for {sample_id}")
            continue

        rc2 = run_cmd([
            "python", "-m", "scripts_tnbc.step2_tnbc_regions",
            "--sample_id", sample_id,
            "--sample_dir", str(sample_dir)
        ])
        if rc2 != 0:
            print(f"[FAIL] Step 2 failed for {sample_id}")
            continue

        print(f"[OK] Finished screening for {sample_id}")


if __name__ == "__main__":
    main()
