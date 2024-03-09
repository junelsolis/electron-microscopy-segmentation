"""
Trains a U-Net segmentation model using the TFRecord dataset created in util/create_dataset.py.
"""

import os

os.environ["SM_FRAMEWORK"] = "tf.keras"
import argparse

import numpy as np
import segmentation_models as sm
import tensorflow as tf


DATASET_PATH = os.path.join(os.path.dirname(__file__), "..", "dataset")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "logs")

BACKBONE = "inceptionresnetv2"
LEARNING_RATE = 0.0001


def _parse_tf_record(proto):
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "mask": tf.io.FixedLenFeature([], tf.string),
    }

    example = tf.io.parse_single_example(proto, feature_description)

    image = tf.io.decode_png(example["image"], channels=1)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.decode_png(example["mask"], channels=1)
    mask = tf.cast(mask, tf.float32) / 255.0

    return image, mask


if __name__ == "__main__":

    # configure command line args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default=DATASET_PATH, help="Path to TFRecord dataset."
    )
    parser.add_argument(
        "--model-dir", type=str, default=MODEL_DIR, help="Directory to save models to."
    )
    parser.add_argument(
        "--model-name", type=str, default="em-model", help="Name of the model to save."
    )
    parser.add_argument(
        "--log-dir", type=str, default=LOG_DIR, help="Directory to save logs to."
    )
    parser.add_argument(
        "--batch-size", type=int, default=12, help="Batch size to use for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=250, help="Number of epochs to train for."
    )

    args = parser.parse_args()

    # configure callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(args.model_dir, f"{args.model_name}.keras"),
            save_best_only=True,
        ),
        tf.keras.callbacks.TensorBoard(log_dir=args.log_dir),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=10, min_delta=0.0001, min_lr=1e-10
        ),
    ]

    # data parallelism mirrored strategy
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        # setup the model
        model = sm.Unet(
            BACKBONE,
            classes=1,
            activation="sigmoid",
            encoder_weights=None,
            input_shape=(None, None, 1),
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
            loss=sm.losses.jaccard_loss,
            metrics=[sm.metrics.iou_score, sm.metrics.f1_score],
        )

        # configure datasets
        train_dataset = tf.data.TFRecordDataset(f"{args.dataset}/train.tfrecord")
        train_dataset = train_dataset.map(_parse_tf_record).batch(args.batch_size)

        val_dataset = tf.data.TFRecordDataset(f"{args.dataset}/val.tfrecord")
        val_dataset = val_dataset.map(_parse_tf_record).batch(args.batch_size)

        test_dataset = tf.data.TFRecordDataset(f"{args.dataset}/test.tfrecord")
        test_dataset = test_dataset.map(_parse_tf_record).batch(args.batch_size)

        # train the model
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=args.epochs,
            callbacks=callbacks,
            steps_per_epoch=tf.data.experimental.cardinality(train_dataset).numpy()
            // args.batch_size,
            validation_steps=tf.data.experimental.cardinality(val_dataset).numpy()
            // args.batch_size,
        )

        # evaluate the model
        model.evaluate(test_dataset)
