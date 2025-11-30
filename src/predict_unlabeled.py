import sys
import json
import csv
from pathlib import Path
from typing import List

import numpy as np
import tensorflow as tf
import yaml
import bentoml
from PIL import Image

from utils.seed import set_seed  


def load_labels(prepared_dataset_folder: Path) -> List[str]:
    """Charge la liste des labels à partir de labels.json"""
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)
    return labels


def load_model(model_folder: Path) -> tf.keras.Model:
    """
    Charge le modèle via BentoML, comme dans evaluate.py.

    """
    bentomodel_path = model_folder / "trash_classifier_model.bentomodel"
    if bentomodel_path.exists():
        try:
            bentoml.models.import_model(str(bentomodel_path))
        except bentoml.exceptions.BentoMLException:
            # déjà importé -> on ignore
            pass

    model = bentoml.keras.load_model("trash_classifier_model")
    return model


def preprocess_image(
    img_path: Path,
    image_size: List[int],
    grayscale: bool,
) -> np.ndarray:
    """
    Préprocessing avec :
    - prepare.py (image_size, grayscale)
    - train.py (preprocess dans bentoml : resize + /255)
    """
    img = Image.open(img_path)
    img = img.convert("L" if grayscale else "RGB")
    img = img.resize(tuple(image_size))

    arr = np.array(img).astype("float32") / 255.0  # normalisation comme dans prepare/train

    # Ajout du canal pour grayscale (H, W) -> (H, W, 1)
    if grayscale:
        if arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

    # Ajout de la dimension batch
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)

    return arr


def main() -> None:
    if len(sys.argv) != 5:
        print("Arguments error. Usage:\n")
        print(
            "\tpython3 src/predict_unlabeled.py "
            "<unlabeled-dir> <prepared-dataset-folder> <model-folder> <output-csv>\n"
        )
        sys.exit(1)

    unlabeled_dir = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    model_folder = Path(sys.argv[3])
    output_csv = Path(sys.argv[4])

    if not unlabeled_dir.exists():
        print(f"Unlabeled directory not found: {unlabeled_dir}")
        sys.exit(1)

    # Charge les paramètres (prepare + train) depuis params.yaml
    params = yaml.safe_load(open("params.yaml"))
    prepare_params = params["prepare"]
    train_params = params["train"]

    image_size = prepare_params["image_size"]  # ex: [32, 32]
    grayscale = prepare_params["grayscale"]    # True / False
    seed = train_params["seed"]

    # Seed
    set_seed(seed)

    # Charge les labels et le modèle
    labels = load_labels(prepared_dataset_folder)
    model = load_model(model_folder)

    # Récupère toutes les images dans data/unlabeled
    image_paths = sorted(
        [
            p
            for p in unlabeled_dir.glob("**/*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ]
    )

    if not image_paths:
        print(f"No images found in {unlabeled_dir}")
        sys.exit(0)

    # Prépare le fichier de sortie
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        fieldnames = ["filepath", "pred_label", "max_prob", "probs"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for img_path in image_paths:
            x = preprocess_image(img_path, image_size, grayscale)
            logits = model.predict(x, verbose=0)[0]  # logits 
            probs = tf.nn.softmax(logits).numpy()    # proba par classe

            pred_idx = int(np.argmax(probs))
            pred_label = labels[pred_idx]
            max_prob = float(np.max(probs))

            writer.writerow(
                {
                    "filepath": str(img_path),
                    "pred_label": pred_label,
                    "max_prob": max_prob,
                    "probs": json.dumps(probs.tolist()),
                }
            )

    print(f"Saved predictions for {len(image_paths)} images to {output_csv}")


if __name__ == "__main__":
    main()
