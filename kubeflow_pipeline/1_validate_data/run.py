# 1_validate_data

# validate data
# 1. validate data before training
# !do not load data such as "Dataloader" in PyTorch

import os
import logging
import argparse

import glob
import numpy as np

def validation_data(train_data_file, test_data_file, faiss_train_data_file, faiss_test_data_file, **kwargs):
    # validate data
    # check data path, shape, type

    def validate(npy_image, npy_label, **kwargs):
        images = np.load(npy_image)
        labels = np.load(npy_label)
        
        npy_image_shape = images.shape
        npy_label_shape = labels.shape
        
        assert npy_image[-7:] == npy_label[-7:]
        assert len(npy_image_shape) == 3
        assert npy_image_shape[0] == npy_label_shape[0]
        assert npy_image_shape[1] == kwargs["image_width"]
        assert npy_image_shape[2] == kwargs["image_height"]
        assert images.dtype == kwargs["image_type"]
        assert labels.dtype == kwargs["label_type"]

        del images
        del labels
    
    def iter_npy(data_file, **kwargs):
        npy_images = glob.glob(data_file + "*images*.npy")
        npy_labels = glob.glob(data_file + "*labels*.npy")

        npy_images.sort()
        npy_labels.sort()

        for npy_image, npy_label in zip(npy_images, npy_labels):
            validate(npy_image, npy_label, **kwargs)
            print("[+] {}, {} validated".format(npy_image, npy_label))

    iter_npy(train_data_file, **kwargs)
    iter_npy(test_data_file, **kwargs)
    iter_npy(faiss_train_data_file, **kwargs)
    iter_npy(faiss_test_data_file, **kwargs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_data_path', type=str, default="/data/mnist/train")
    parser.add_argument('--test_data_path', type=str, default="/data/mnist/test")
    parser.add_argument('--train_data_file', type=str, default="train_mnist")
    parser.add_argument('--test_data_file', type=str, default="test_mnist")
    
    parser.add_argument('--faiss_train_data_path', type=str, default="/data/faiss/train")
    parser.add_argument('--faiss_test_data_path', type=str, default="/data/faiss/test")
    parser.add_argument('--faiss_train_data_file', type=str, default="faiss_train")
    parser.add_argument('--faiss_test_data_file', type=str, default="faiss_test")

    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)
    parser.add_argument('--npy_interval', type=int, default=5000)

    args = parser.parse_args()
    
    print("Validating data...")
    validation_data(
        train_data_file=os.path.join(args.train_data_path, args.train_data_file),
        test_data_file=os.path.join(args.test_data_path, args.test_data_file),
        faiss_train_data_file=os.path.join(args.faiss_train_data_path, args.faiss_train_data_file),
        faiss_test_data_file=os.path.join(args.faiss_test_data_path, args.faiss_test_data_file),
        image_width=args.image_width,
        image_height=args.image_height,
        image_channel=args.image_channel,
        image_type=np.float32,
        label_type=np.int64
    )