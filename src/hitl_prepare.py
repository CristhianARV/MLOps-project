# python3 src/hitl_prepare.py ./data/uncertain_images  ./data/annotations_hitl1.csv
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
import pandas as pd

from utils.seed import set_seed

name_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']


def get_preview_plot(ds: tf.data.Dataset, labels: List[str]) -> plt.Figure:
    """Plot a preview of the prepared dataset"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds.take(1):
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(name_labels[label_idxs[i].numpy()])
            plt.axis("off")

    return fig

def transform_labels_to_int(label : str) -> int:
    match label:
        case 'cardboard':
            return 0
        case 'glass':
            return 1
        case 'metal':
            return 2
        case 'paper':
            return 3
        case 'plastic':
            return 4
        case 'trash':
            return 5



def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 hitl_prepare.py <uncertain-images-dataset-folder> <labels-with-labels-studio.csv>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("hitl_parameters.yaml"))["prepare"]

    uncertain_images_dataset_folder = Path(sys.argv[1])
    labels_csv_path = Path(sys.argv[2])

    seed = prepare_params["seed"]
    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    batch_size = prepare_params["batch_size"]

    # Set seed for reproducibility
    set_seed(seed)

    # Retrieve labels from CSV
    df = pd.read_csv(labels_csv_path)
    labels = df['choice'].tolist()

    labels = [transform_labels_to_int(label) for label in labels]

    # Read data
    ds_hitl = tf.keras.utils.image_dataset_from_directory(
        uncertain_images_dataset_folder,
        labels=labels,
        label_mode="int",
        color_mode="grayscale" if grayscale else "rgb",
        batch_size=batch_size,
        image_size=image_size,
        shuffle=True,
        seed=seed,
    )
    class_names = ds_hitl.class_names

    hitl_prepared_dataset_folder = Path('data/prepared_hitl/hitl')

    if not hitl_prepared_dataset_folder.exists():
        hitl_prepared_dataset_folder.mkdir(parents=True)

    # Save the preview plot
    preview_plot = get_preview_plot(ds_hitl, class_names)
    preview_plot.savefig(hitl_prepared_dataset_folder / "preview.png")

    # Normalize the data
    normalization_layer = tf.keras.layers.Rescaling(
        1.0 / 255
    )
    ds_hitl = ds_hitl.map(lambda x, y: (normalization_layer(x), y))

    # Save the prepared dataset
    with open(hitl_prepared_dataset_folder / "labels.json", "w") as f:
        json.dump(class_names, f)
    tf.data.Dataset.save(ds_hitl, str(hitl_prepared_dataset_folder / "hitl"))

    print(f"\nDataset saved at {hitl_prepared_dataset_folder.absolute()}")

if __name__ == "__main__":
    main()