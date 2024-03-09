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
from skimage import io, img_as_ubyte
from tqdm import tqdm
from tiler import Tiler
import io as bytesio
from PIL import Image

SOURCE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "epfl_em_mitochondria"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "dataset")


def create_tf_record(fpaths: tuple, output_filename: str):
    writer = tf.io.TFRecordWriter(output_filename)

    for img_path, mask_path in tqdm(
        fpaths, desc=f"Creating {os.path.basename(output_filename)}..."
    ):
        assert os.path.basename(img_path) == os.path.basename(mask_path)

        img = io.imread(img_path)
        mask = io.imread(mask_path)

        # convert to 8-bit
        img = img_as_ubyte(img)
        mask = img_as_ubyte(mask)
        # mask[mask > 0] = 255
        
        # random vertical flip
        if random.choice([True, False]):
            img = np.flip(img, axis=0)
            mask = np.flip(mask, axis=0)
            
        # random horizontal flip
        if random.choice([True, False]):
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)

        # convert image and mask to png
        img_buffer = bytesio.BytesIO()
        Image.fromarray(img).save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        mask_buffer = bytesio.BytesIO()
        Image.fromarray(mask).save(mask_buffer, format="PNG")
        mask_bytes = mask_buffer.getvalue()

        # serialize
        feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes])),
            "mask": tf.train.Feature(bytes_list=tf.train.BytesList(value=[mask_bytes])),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

    writer.close()


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
                tile_fname = f"{fname[1].replace('_groundtruth.tif', '')}_{z}_{tile_id}.tif"
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

    create_tf_record(train_fpaths, os.path.join(output_path, "train.tfrecord"))
    create_tf_record(val_fpaths, os.path.join(output_path, "val.tfrecord"))
    create_tf_record(test_fpaths, os.path.join(output_path, "test.tfrecord"))
