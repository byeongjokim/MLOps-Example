# 0_data

# process data
# 1. collect data
# 2. preprocess data
# 3. save the data to readable files
# !do not load data such as "Dataloader" in PyTorch

import os
import logging
import argparse

import glob
import cv2
import numpy as np

def collect_data(train_data_path, test_data_path, faiss_train_data_path, faiss_test_data_path):
    # collect data

    train_data = glob.glob(os.path.join(train_data_path, "**/*.png"))
    train_labels = [int(i.split(os.sep)[-2]) for i in train_data]
    print("[+] collect {} train data".format(str(len(train_labels))))

    test_data = glob.glob(os.path.join(test_data_path, "**/*.png"))
    test_labels = [int(i.split(os.sep)[-2]) for i in test_data]
    print("[+] collect {} test data".format(str(len(test_labels))))

    faiss_train_data = glob.glob(os.path.join(faiss_train_data_path, "**/*.png"))
    faiss_train_labels = [int(i.split(os.sep)[-2]) for i in faiss_train_data]
    print("[+] collect {} faiss train data".format(str(len(faiss_train_labels))))

    faiss_test_data = glob.glob(os.path.join(faiss_test_data_path, "**/*.png"))
    faiss_test_labels = [int(i.split(os.sep)[-2]) for i in faiss_test_data]
    print("[+] collect {} faiss test data".format(str(len(faiss_test_labels))))

    return {"train":{"image_paths": train_data, "labels": train_labels},
            "test":{"image_paths": test_data, "labels": test_labels},
            "faiss_train":{"image_paths": faiss_train_data, "labels": faiss_train_labels},
            "faiss_test":{"image_paths": faiss_test_data, "labels": faiss_test_labels}
    }

def preprocess_data(data, train_data_file, test_data_file, faiss_train_data_file, faiss_test_data_file, interval, **kwargs):
    # preprocess data
    # save data to npy for training/testing/faiss training

    def load_preprocess_image(image_path, **kwargs):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (kwargs["image_width"], kwargs["image_height"]))
        return image

    def save_npy(filename, npy_num, image_paths, labels, **kwargs):
        images = np.asarray([load_preprocess_image(image_path, **kwargs) for image_path in image_paths], dtype=kwargs["image_type"])
        labels = np.asarray(labels, dtype=kwargs["label_type"])
        np.save(filename + "_images_" + str(npy_num).zfill(3) + ".npy", images)
        np.save(filename + "_labels_" + str(npy_num).zfill(3) + ".npy", labels)
        
        print("{} saved ".format(filename + "_images_" + str(npy_num).zfill(3) + ".npy"))
        print("{} saved ".format(filename + "_labels_" + str(npy_num).zfill(3) + ".npy"))

        del images
        del labels

    def iter_data(data, filename, interval, **kwargs):
        npy_num = 0
        for i in range(0, len(data["image_paths"]), interval):
            save_npy(
                filename=filename,
                npy_num=npy_num,
                image_paths=data["image_paths"][i:i+interval],
                labels=data["labels"][i:i+interval],
                **kwargs
            )
            npy_num = npy_num + 1
        
    iter_data(data["train"], train_data_file, interval, **kwargs)
    iter_data(data["test"], test_data_file, interval, **kwargs)
    iter_data(data["faiss_train"], faiss_train_data_file, interval, **kwargs)
    iter_data(data["faiss_test"], faiss_test_data_file, interval, **kwargs)

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

    print("Collecting data...")
    data = collect_data(
        train_data_path=args.train_data_path,
        test_data_path=args.test_data_path,
        faiss_train_data_path=args.faiss_train_data_path,
        faiss_test_data_path=args.faiss_test_data_path
    )

    print("Preprocessing data...")
    preprocess_data(
        data=data, 
        train_data_file=os.path.join(args.train_data_path, args.train_data_file),
        test_data_file=os.path.join(args.test_data_path, args.test_data_file),
        faiss_train_data_file=os.path.join(args.faiss_train_data_path, args.faiss_train_data_file),
        faiss_test_data_file=os.path.join(args.faiss_test_data_path, args.faiss_test_data_file),
        interval=args.npy_interval,
        image_width=args.image_width,
        image_height=args.image_height,
        image_channel=args.image_channel,
        image_type=np.float32,
        label_type=np.int64
    )