# 0_data

# do only processing data
# 1. collect data
# 2. preprocess data
# 3. save the data to readable files
# 4. validate data before training
# !do not load data such as "Dataloader" in PyTorch

import os
import glob
import cv2
import numpy as np

def collect_data(train_data_path, test_data_path, faiss_train_data_path, faiss_test_data_path):
    # collect data

    train_data = glob.glob(os.path.join(train_data_path, "**/*.png"))
    train_labels = [int(i.split(os.sep)[-2]) for i in train_data]
    
    test_data = glob.glob(os.path.join(test_data_path, "**/*.png"))
    test_labels = [int(i.split(os.sep)[-2]) for i in test_data]

    faiss_train_data = glob.glob(os.path.join(faiss_train_data_path, "**/*.png"))
    faiss_train_labels = [int(i.split(os.sep)[-2]) for i in faiss_train_data]

    faiss_test_data = glob.glob(os.path.join(faiss_test_data_path, "**/*.png"))
    faiss_test_labels = [int(i.split(os.sep)[-2]) for i in faiss_test_data]

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

    iter_npy(train_data_file, **kwargs)
    iter_npy(test_data_file, **kwargs)
    iter_npy(faiss_train_data_file, **kwargs)
    iter_npy(faiss_test_data_file, **kwargs)

if __name__ == "__main__":
    train_data_path = os.getenv('TRAIN_DATA', "/data/mnist/train")
    test_data_path = os.getenv('TEST_DATA', "/data/mnist/test")
    train_data_file = os.getenv("TRAIN_DATA_FILE", "train_mnist")
    test_data_file = os.getenv("TEST_DATA_FILE", "test_mnist")

    faiss_train_data_path = os.getenv('FAISS_TRAIN_DATA', "/data/faiss/train")
    faiss_test_data_path = os.getenv('FAISS_TEST_DATA', "/data/faiss/test")
    faiss_train_data_file = os.getenv('FAISS_TRAIN_DATA_FILE', "faiss_train")
    faiss_test_data_file = os.getenv('FAISS_TEST_DATA_FILE', "faiss_test")

    image_width = os.getenv("IMAGE_WIDTH", 28)
    image_height = os.getenv("IMAGE_HEIGHT", 28)
    image_channel = os.getenv("IMAGE_CAHNNEL", 1)
    npy_interval = os.getenv("NPY_INTERVAL", 5000)

    print("Collecting data...")
    data = collect_data(
        train_data_path=train_data_path,
        test_data_path=test_data_path,
        faiss_train_data_path=faiss_train_data_path,
        faiss_test_data_path=faiss_test_data_path
    )

    print("Preprocessing data...")
    preprocess_data(
        data=data, 
        train_data_file=os.path.join(train_data_path, train_data_file),
        test_data_file=os.path.join(test_data_path, test_data_file),
        faiss_train_data_file=os.path.join(faiss_train_data_path, faiss_train_data_file),
        faiss_test_data_file=os.path.join(faiss_test_data_path, faiss_test_data_file),
        interval=npy_interval,
        image_width=image_width,
        image_height=image_height,
        image_channel=image_channel,
        image_type=np.float32,
        label_type=np.int64
    )

    print("Validating data...")
    validation_data(
        train_data_file=os.path.join(train_data_path, train_data_file),
        test_data_file=os.path.join(test_data_path, test_data_file),
        faiss_train_data_file=os.path.join(faiss_train_data_path, faiss_train_data_file),
        faiss_test_data_file=os.path.join(faiss_test_data_path, faiss_test_data_file),
        image_width=image_width,
        image_height=image_height,
        image_channel=image_channel,
        image_type=np.float32,
        label_type=np.int64
    )