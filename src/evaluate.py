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

dst_global_score = "data/scores/eval_scores.csv"
dst_label_score = "data/scores/class_scores.csv"

def push_scores_to_file(report : dict, dst : str):
    """
    report: dict
        Dictionary returned by sklearn.metrics.classification_report(output_dict=True)
    """
    
    path = Path(dst)
    new_file = not path.is_file()

    now = datetime.now().isoformat(timespec="seconds")

    # extraction depuis le dict sklearn
    accuracy = report["accuracy"]
    macro = report["macro avg"]
    precision = macro["precision"]
    recall = macro["recall"]
    f1_score = macro["f1-score"]
    support = macro["support"]

    with path.open("a", encoding="utf-8") as f:
        if new_file:
            f.write("timestamp;accuracy;precision;recall;f1_score;support\n")
        
        f.write(f"{now};{accuracy};{precision};{recall};{f1_score};{support}\n")


def push_labels_score_to_file(report : dict, dst : str):
    """
    report: dict
        Dictionary returned by sklearn.metrics.classification_report(output_dict=True)
    """
    
    now = datetime.now().isoformat(timespec="seconds")
    metrics = ['precision', 'recall', 'f1-score', 'support']

    # Charger les classes depuis classes.txt
    with open("classes.txt", "r", encoding="utf-8") as f:
        classes = f.read().strip().split(";")

    dst_path = Path(dst)

    # Écrire l'en-tête si fichier inexistant
    if not dst_path.is_file():
        header = ["timestamp"]
        for cls in classes:
            for metric in metrics:
                header.append(f"{cls}_{metric}")

        with dst_path.open("a", encoding="utf-8") as f:
            f.write(";".join(header) + "\n")

    # Écrire la ligne de données
    row = [now]
    for cls in classes:
        for metric in metrics:
            row.append(str(report[cls][metric]))

    with dst_path.open("a", encoding="utf-8") as f:
        f.write(";".join(row) + "\n")


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
    push_scores_to_file(report, dst_global_score)
    # Push label-wise scores
    push_labels_score_to_file(report, dst_label_score)

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