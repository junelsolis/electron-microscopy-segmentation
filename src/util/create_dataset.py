"""
Script to convert EPFL Mitochondria dataset to TIFF image/mask pairs and TFRecords. 
The dataset is available at https://www.epfl.ch/labs/cvlab/data/data-em/
Author: Junel Solis, 2024
"""

import argparse
import os
import random
import shutil
from glob import glob

import numpy as np
import tensorflow as tf
from skimage import io
from tiler import Tiler
import io as bytesio
from PIL import Image

SOURCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "epfl_em_mitochondria"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def create_tfrecord(fpaths, output_file):
    with tf.io.TFRecordWriter(output_file) as writer:
        for fpath in fpaths:
            img = io.imread(fpath[0])
            mask = io.imread(fpath[1])

            # random horizontal flip
            if np.random.random() > 0.5:
                img = np.flip(img, axis=1)
                mask = np.flip(mask, axis=1)

            # random vertical flip
            if np.random.random() > 0.5:
                img = np.flip(img, axis=0)
                mask = np.flip(mask, axis=0)

            # conver to PIL
            img = Image.fromarray(img)
            mask = Image.fromarray(mask)

            img_byte_arr = bytesio.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            mask_byte_arr = bytesio.BytesIO()
            mask.save(mask_byte_arr, format="PNG")
            mask_bytes = mask_byte_arr.getvalue()

            feature = {
                "image": _bytes_feature(img_bytes),
                "mask": _bytes_feature(mask_bytes),
            }

            example_proto = tf.train.Example(
                features=tf.train.Features(feature=feature)
            )
            writer.write(example_proto.SerializeToString())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert EPFL Mitochondria dataset to TIFF image/mask pairs and TFRecords."
    )

    parser.add_argument(
        "--source_path",
        type=str,
        default=SOURCE_PATH,
        help="Path to EPFL Mitochondria dataset",
    )
    parser.add_argument(
        "--output_path", type=str, default=OUTPUT_PATH, help="Path to save the dataset"
    )
    parser.add_argument("--tile-size", type=int, default=256, help="Size of the tiles")

    args = parser.parse_args()

    source_path = args.source_path
    output_path = args.output_path

    # configure output dir
    shutil.rmtree(os.path.join(output_path, "images"), ignore_errors=True)
    shutil.rmtree(os.path.join(output_path, "masks"), ignore_errors=True)
    os.makedirs(os.path.join(output_path, "images"))
    os.makedirs(os.path.join(output_path, "masks"))

    # define the filenames in the dataset and pair them
    img_fnames = ["training.tif", "testing.tif"]
    mask_fnames = ["training_groundtruth.tif", "testing_groundtruth.tif"]

    fnames = list(zip(img_fnames, mask_fnames))

    for fname in fnames:
        img = io.imread(os.path.join(source_path, fname[0]))
        mask = io.imread(os.path.join(source_path, fname[1]))

        # iterate through the Z-stack and save the tiles at each slice.
        for z in range(img.shape[0]):
            img_z_slice = img[z]
            mask_z_slice = mask[z]

            tiler = Tiler(
                data_shape=img_z_slice.shape,
                tile_shape=(args.tile_size, args.tile_size),
            )

            for tile_id, tile in tiler.iterate(img_z_slice):
                tile_fname = f"{fname[0].split('.')[0]}_{z}_{tile_id}.tif"
                io.imsave(
                    os.path.join(output_path, "images", tile_fname),
                    tile,
                    check_contrast=False,
                )

            for tile_id, tile in tiler.iterate(mask_z_slice):
                tile_fname = f"{fname[1].split('.')[0]}_{z}_{tile_id}.tif"
                io.imsave(
                    os.path.join(output_path, "masks", tile_fname),
                    tile,
                    check_contrast=False,
                )

    # Now create a TFRecord datasets with train / val / test subsets.
    random.seed(4887)

    img_paths = sorted(glob(os.path.join(output_path, "images", "*.tif")))
    mask_paths = sorted(glob(os.path.join(output_path, "masks", "*.tif")))

    fpaths = list(zip(img_paths, mask_paths))
    random.shuffle(fpaths)

    # split the image/mask pair paths into train/val/test 60/20/20
    split_idx_1 = int(0.6 * len(fpaths))
    split_idx_2 = int(0.8 * len(fpaths))

    train_fpaths = fpaths[:split_idx_1]
    val_fpaths = fpaths[split_idx_1:split_idx_2]
    test_fpaths = fpaths[split_idx_2:]

    create_tfrecord(train_fpaths, os.path.join(output_path, "train.tfrecord"))
    create_tfrecord(val_fpaths, os.path.join(output_path, "val.tfrecord"))
    create_tfrecord(test_fpaths, os.path.join(output_path, "test.tfrecord"))
