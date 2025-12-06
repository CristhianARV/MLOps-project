import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import bentoml
from sklearn.metrics import classification_report
from datetime import datetime
from google.cloud import storage
from google.cloud.exceptions import NotFound
import pandas as pd



def push_scores_to_file(
    report: dict,
    bucket_name: str,
    blob_name: str = "scores/eval_scores.csv",
) -> None:
    """
    Append global scores to a CSV stored in a Google Cloud Storage bucket.

    report:
        Dictionary returned by sklearn.metrics.classification_report(output_dict=True)
    bucket_name:
        Name of the GCS bucket
    blob_name:
        Path of the CSV inside the bucket (default: scores/eval_scores.csv)
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # Récupérer le CSV existant (s'il existe)
    try:
        csv_text = blob.download_as_text()
    except Exception:
        # Blob n'existe pas encore
        csv_text = ""

    # Ajouter l'en-tête si fichier vide
    if not csv_text:
        csv_text = "timestamp;accuracy;precision;recall;f1_score;support\n"

    # Extraire les scores du report
    now = datetime.now().isoformat(timespec="seconds")
    accuracy = report["accuracy"]
    macro = report["macro avg"]
    precision = macro["precision"]
    recall = macro["recall"]
    f1_score = macro["f1-score"]
    support = macro["support"]

    # Ajouter une nouvelle ligne
    line = f"{now};{accuracy};{precision};{recall};{f1_score};{support}\n"
    csv_text += line

    # Réécrire le CSV dans le bucket
    blob.upload_from_string(csv_text, content_type="text/csv")



def push_class_scores_to_file(
    report: dict,
    bucket_name: str,
    blob_name: str = "scores/class_scores.csv",
    classes_file: str = "classes.txt",
) -> None:
    """
    Récupère un CSV dans un bucket GCS, ajoute une ligne avec les scores
    par classe, puis réécrit le CSV modifié dans le bucket.

    - report : dict issu de classification_report (sklearn) en format json
    - bucket_name : nom du bucket GCS
    - blob_name : chemin/nom du fichier CSV dans le bucket
    - classes_file : fichier local contenant les classes (séparées par ';')
    """
    now = datetime.now().isoformat(timespec="seconds")
    classes = pd.read_csv(classes_file, sep=";", header=None).values.tolist()[0]
    metrics = ["precision", "recall", "f1-score", "support"]

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    # 1) Récupérer le contenu actuel du CSV dans le bucket (s'il existe)
    try:
        csv_text = blob.download_as_text(encoding="utf-8")
        has_content = bool(csv_text.strip())
    except NotFound:
        csv_text = ""
        has_content = False

    # 2) Générer l'en-tête si le fichier est nouveau ou vide
    if not has_content:
        header_parts = ["timestamp"]
        for cls in classes:
            for metric in metrics:
                header_parts.append(f"{cls}_{metric}")
        csv_text = ";".join(header_parts) + "\n"

    # 3) Construire la nouvelle ligne
    row_parts = [now]
    for cls in classes:
        for metric in metrics:
            row_parts.append(str(report[cls][metric]))
    csv_text += ";".join(row_parts) + "\n"

    # 4) Réécrire le CSV dans le bucket
    blob.upload_from_string(csv_text, content_type="text/csv")


def get_training_plot(model_history: dict) -> plt.Figure:
    """Plot the training and validation loss"""
    epochs = range(1, len(model_history["loss"]) + 1)

    fig = plt.figure(figsize=(10, 4))
    plt.plot(epochs, model_history["loss"], label="Training loss")
    plt.plot(epochs, model_history["val_loss"], label="Validation loss")
    plt.xticks(epochs)
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    return fig


def get_pred_preview_plot(
    model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
) -> plt.Figure:
    """Plot a preview of the predictions"""
    fig = plt.figure(figsize=(10, 5), tight_layout=True)
    for images, label_idxs in ds_test.take(1):
        preds = model.predict(images)
        for i in range(10):
            plt.subplot(2, 5, i + 1)
            img = (images[i].numpy() * 255).astype("uint8")
            # Convert image to rgb if grayscale
            if img.shape[-1] == 1:
                img = np.squeeze(img, axis=-1)
                img = np.stack((img,) * 3, axis=-1)
            true_label = labels[label_idxs[i].numpy()]
            pred_label = labels[np.argmax(preds[i])]
            # Add red border if the prediction is wrong else add green border
            img = np.pad(img, pad_width=((1, 1), (1, 1), (0, 0)))
            if true_label != pred_label:
                img[0, :, 0] = 255  # Top border
                img[-1, :, 0] = 255  # Bottom border
                img[:, 0, 0] = 255  # Left border
                img[:, -1, 0] = 255  # Right border
            else:
                img[0, :, 1] = 255
                img[-1, :, 1] = 255
                img[:, 0, 1] = 255
                img[:, -1, 1] = 255

            plt.imshow(img)
            plt.title(f"True: {true_label}\n" f"Pred: {pred_label}")
            plt.axis("off")

    return fig


def get_confusion_matrix_plot(
    model: tf.keras.Model, ds_test: tf.data.Dataset, labels: List[str]
) -> plt.Figure:
    """Plot the confusion matrix"""
    fig = plt.figure(figsize=(6, 6), tight_layout=True)
    preds = model.predict(ds_test)

    conf_matrix = tf.math.confusion_matrix(
        labels=tf.concat([y for _, y in ds_test], axis=0),
        predictions=tf.argmax(preds, axis=1),
        num_classes=len(labels),
    )

    # Plot the confusion matrix
    conf_matrix = conf_matrix / tf.reduce_sum(conf_matrix, axis=1)
    plt.imshow(conf_matrix, cmap="Blues")

    # Plot cell values
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = conf_matrix[i, j].numpy()
            if value == 0:
                color = "lightgray"
            elif value > 0.5:
                color = "white"
            else:
                color = "black"
            plt.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )

    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Confusion matrix")

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    model_folder = Path(sys.argv[1])
    prepared_dataset_folder = Path(sys.argv[2])
    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")

    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)

    # Load files
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))
    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Import the model to the model store from a local model folder
    try:
        bentoml.models.import_model(f"{model_folder.absolute()}/trash_classifier_model.bentomodel")
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")

    # Load model
    model = bentoml.keras.load_model("trash_classifier_model")
    model_history = np.load(model_folder.absolute() / "history.npy", allow_pickle=True).item()

    preds = model.predict(ds_test)
    y_true = tf.concat([y for _, y in ds_test], axis=0).numpy()
    y_pred = tf.argmax(preds, axis=1).numpy()

    # Push scores to files
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)

    # Push overall scores
    push_scores_to_file(report, bucket_name="mlops-cris-bucket")
    # Push label-wise scores
    push_class_scores_to_file(report, bucket_name="mlops-cris-bucket")

    # Log metrics
    val_loss, val_acc = model.evaluate(ds_test)
    print(f"Validation loss: {val_loss:.2f}")
    print(f"Validation accuracy: {val_acc * 100:.2f}%")
    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"val_loss": val_loss, "val_acc": val_acc}, f)

    # Save training history plot
    fig = get_training_plot(model_history)
    fig.savefig(evaluation_folder / plots_folder / "training_history.png")

    # Save predictions preview plot
    fig = get_pred_preview_plot(model, ds_test, labels)
    fig.savefig(evaluation_folder / plots_folder / "pred_preview.png")

    # Save confusion matrix plot
    fig = get_confusion_matrix_plot(model, ds_test, labels)
    fig.savefig(evaluation_folder / plots_folder / "confusion_matrix.png")

    print(
        f"\nEvaluation metrics and plot files saved at {evaluation_folder.absolute()}"
    )


if __name__ == "__main__":
    main()