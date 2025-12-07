import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import yaml
import bentoml
from PIL.Image import Image

from utils.seed import set_seed


def get_model(
    image_shape: Tuple[int, int, int],
    dropout_1: float,
    dense_size: int,
    regularization_l2_1: float,
    dropout_2: float,
    output_classes: int,
    regularization_l2_2: float,
) -> tf.keras.Model:
    """Transfer learning model using MobileNetV2 as base model"""

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=image_shape,
        include_top=False,
        weights='imagenet'
    )

    base_model.trainable = True

    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(dropout_1),
        tf.keras.layers.Dense(dense_size, activation='mish', kernel_regularizer=regularizers.l2(regularization_l2_1)),
        tf.keras.layers.Dropout(dropout_2),
        tf.keras.layers.Dense(output_classes, kernel_regularizer=regularizers.l2(regularization_l2_2))
    ])

    return model



def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <prepared-dataset-folder> <model-folder>\n")
        exit(1)

    # Load parameters
    prepare_params = yaml.safe_load(open("params.yaml"))["prepare"]
    train_params = yaml.safe_load(open("params.yaml"))["train"]

    prepared_dataset_folder = Path(sys.argv[1])
    model_folder = Path(sys.argv[2])

    image_size = prepare_params["image_size"]
    grayscale = prepare_params["grayscale"]
    image_shape = (*image_size, 1 if grayscale else 3)

    seed = train_params["seed"]
    lr = train_params["lr"]
    epochs = train_params["epochs"]
    dropout_1 = train_params["dropout_1"]
    dropout_2 = train_params["dropout_2"]
    regularization_l2_1 = train_params["regularization_l2_1"]
    regularization_l2_2 = train_params["regularization_l2_2"]
    dense_size = train_params["dense_size"]
    output_classes = train_params["output_classes"]

    # Set seed for reproducibility
    set_seed(seed)

    # Load data
    ds_train = tf.data.Dataset.load(str(prepared_dataset_folder / "train"))
    ds_test = tf.data.Dataset.load(str(prepared_dataset_folder / "test"))

    labels = None
    with open(prepared_dataset_folder / "labels.json") as f:
        labels = json.load(f)

    # Define the model
    model = get_model(image_shape, 
                      dropout_1, 
                      dense_size, 
                      regularization_l2_1, 
                      dropout_2, 
                      output_classes, 
                      regularization_l2_2)
    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary()

    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=3, min_lr=1e-6)
    early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

    # Train the model
    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
        callbacks=[lr_scheduler, early_stopping]
    )

    # Save the model
    model_folder.mkdir(parents=True, exist_ok=True)

    def preprocess(x: Image):
        # convert PIL image to tensor
        x = x.convert('L' if grayscale else 'RGB')
        x = x.resize(image_size)
        x = np.array(x)
        x = x / 255.0
        # add batch dimension
        x = np.expand_dims(x, axis=0)
        return x

    def postprocess(x: Image):
        return {
            "prediction": labels[tf.argmax(x, axis=-1).numpy()[0]],
            "probabilities": {
                labels[i]: prob
                for i, prob in enumerate(tf.nn.softmax(x).numpy()[0].tolist())
            },
        }

    # Save the model using BentoML to its model store
    # https://docs.bentoml.com/en/latest/reference/frameworks/keras.html#bentoml.keras.save_model
    bentoml.keras.save_model(
        "trash_classifier_model",
        model,
        include_optimizer=True,
        custom_objects={
            "preprocess": preprocess,
            "postprocess": postprocess,
        }
    )

    # Export the model from the model store to the local model folder
    bentoml.models.export_model(
        "trash_classifier_model:latest",
        f"{model_folder.absolute()}/trash_classifier_model.bentomodel",
    )

    # Save the model history
    np.save(model_folder.absolute() / "history.npy", model.history.history)

    print(f"\nModel saved at {model_folder.absolute()}")


if __name__ == "__main__":
    main()