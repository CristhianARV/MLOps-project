# src/select_hitl.py

import csv
import json
from pathlib import Path

import yaml 


def main(
    predictions_csv: str = "data/hitl/predictions.csv",
    params_path: str = "params.yaml",
    output_dir: str = "data/hitl/to_label",
):
    predictions_csv = Path(predictions_csv)
    params_path = Path(params_path)
    output_dir = Path(output_dir)

    # Lecture du seuil dans params.yaml
    params = yaml.safe_load(params_path.read_text())
    threshold = float(params["hitl"]["threshold"])

    output_dir.mkdir(parents=True, exist_ok=True)

    n_selected = 0

    with predictions_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            max_prob = float(row["max_prob"])
            if max_prob < threshold:
                src = Path(row["filepath"])
                if src.exists():
                    dst = output_dir / src.name
                    if not dst.exists():
                        dst.write_bytes(src.read_bytes())
                        n_selected += 1

    print(f"Selected {n_selected} low-confidence images into {output_dir}")


if __name__ == "__main__":
    main()
