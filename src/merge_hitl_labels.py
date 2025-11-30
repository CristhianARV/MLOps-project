import json
import shutil
from pathlib import Path
import sys


def main():
    if len(sys.argv) != 4:
        print(
            "Usage:\n"
            "\tpython3.12 src/merge_hitl_labels.py "
            "<labelstudio_export_json> <raw_root> <hitl_labeled_root>\n"
        )
        print(
            "Example:\n"
            "\tpython3.12 src/merge_hitl_labels.py "
            "data/hitl/exports/labelstudio_export.json data/raw data/hitl/labeled\n"
        )
        sys.exit(1)

    export_path = Path(sys.argv[1])
    raw_root = Path(sys.argv[2])
    hitl_labeled_root = Path(sys.argv[3])

    if not export_path.exists():
        print(f"Export file not found: {export_path}")
        sys.exit(1)

    if not raw_root.exists():
        print(f"Raw data root not found: {raw_root}")
        sys.exit(1)

    # Dossier d'où viennent les images peu confiantes
    hitl_input_dir = Path("data/hitl/to_label")

    hitl_labeled_root.mkdir(parents=True, exist_ok=True)

    print(f"Reading Label Studio export from: {export_path}")
    data = json.loads(export_path.read_text())

    n_total = 0
    n_copied = 0

    for task in data:
        n_total += 1

        # Avec un export Label Studio "classification d'images"
        # task["data"]["image"]      -> chemin ou URL de l'image
        # task["annotations"][0]["result"][0]["value"]["choices"][0] -> label choisi
        try:
            image_ref = task["data"]["image"]
            annotations = task.get("annotations", [])
            if not annotations:
                print(f"Skipping task with no annotations: {task.get('id', 'no-id')}")
                continue

            result = annotations[0]["result"][0]
            label = result["value"]["choices"][0]
        except (KeyError, IndexError) as e:
            print(f"Skipping task due to unexpected format: {e}")
            continue

        # On garde le nom de fichier
        image_name = Path(image_ref).name

        # l'image vient de data/hitl/to_label
        src = hitl_input_dir / image_name
        if not src.exists():
            print(f"Warning: source image not found: {src}")
            continue

        # 1) Copie dans data/hitl/labeled/<label>/ pour traçabilité HITL
        labeled_class_dir = hitl_labeled_root / label
        labeled_class_dir.mkdir(parents=True, exist_ok=True)
        dst_hitl = labeled_class_dir / image_name
        shutil.copy(src, dst_hitl)

        # 2) Copie dans data/raw/<label>/ pour enrichir le dataset d'entraînement
        raw_class_dir = raw_root / label
        raw_class_dir.mkdir(parents=True, exist_ok=True)
        dst_raw = raw_class_dir / image_name
        shutil.copy(src, dst_raw)

        n_copied += 1

    print(f"Processed {n_total} tasks from Label Studio.")
    print(f"Copied {n_copied} images into:")
    print(f"  - HITL labeled dir : {hitl_labeled_root}")
    print(f"  - Raw data dir     : {raw_root}")


if __name__ == "__main__":
    main()
