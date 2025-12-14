# python3 ./src/hitl_selection.py ./data/raw2/train/images ./data/uncertain_images

from pathlib import Path
import sys
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

import bentoml
from bentoml.models import BentoModel
from bentoml.keras import load_model
import yaml
from utils.seed import set_seed
import shutil

import re
def natural_key(path):
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', path.name)]


def get_model ():
    model_ref = bentoml.models.get("trash_classifier_model:latest")
    print(model_ref)

    # Charger le modèle
    model = load_model(model_ref)
    return model

def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 hitl_selection.py <raw-dataset-folder> <uncertain-images-dataset-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    hitl_params = yaml.safe_load(open("hitl_parameters.yaml"))["hitl_selection"]

    raw_dataset_folder = Path(sys.argv[1])
    uncertain_images_dataset_folder = Path(sys.argv[2])
    seed = prepare_params["seed"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    batch_size = prepare_params["batch_size"]

    num_uncertain = hitl_params["num_uncertain"]

    # Set seed for reproducibility
    set_seed(seed)

    # Read data
    ds = tf.keras.utils.image_dataset_from_directory(
        raw_dataset_folder,
        labels=None,
        label_mode=None,
        color_mode="grayscale" if grayscale else "rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
    )

    file_paths = ds.file_paths  # liste de strings

    if not uncertain_images_dataset_folder.exists():
        uncertain_images_dataset_folder.mkdir(parents=True)

    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(
        1.0 / 255
    )
    ds = ds.map(lambda x: normalization_layer(x))

    # Load model
    model = get_model()

    # Predict
    preds = model.predict(ds)
    probs = tf.nn.softmax(preds, axis=1).numpy()
    y_pred = np.argmax(probs, axis=-1)

    # Select uncertain images (x images with lowest max probability)
    max_probs = np.max(probs, axis=1)
    uncertain_indices = np.argsort(max_probs)[:num_uncertain]
    print(f"Indices des {num_uncertain} images les plus incertaines :", uncertain_indices)

    # 6) Copier les fichiers d'origine correspondants
    for i in uncertain_indices:
        src = Path(file_paths[i])
        dst = uncertain_images_dataset_folder / src.name
        shutil.copy(src, dst)
    print(f"Images incertaines copiées dans le dossier {uncertain_images_dataset_folder}")

    print("Prédictions :", y_pred)

if __name__ == "__main__":
    main()

