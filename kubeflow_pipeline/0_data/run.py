# do only preprocess data
# 1. collect data 
# 2. preprocess data
# 3. save the data to readable files
# 4. validate data before training
# !do not load data such as "Dataloader" in PyTorch

import os

def collect_data(input_folder):
    # get data
    data = []
    return data


def preprocess_data(data, data_path):
    # preprocess data
    # save data to npy for training/testing
    print("save data")

def validation_data(data_path):
    # check data path, shape, type
    print("validation data")


if __name__ == "__main__":
    input_folder = "" #os.env["input_folder"]
    train_data_path = ""
    test_data_path = ""

    print("Collecting data...")
    data = collect_data(input_folder)

    print("Preprocessing data...")
    preprocess_data(data=data, data_path={"train":train_data_path, "test":test_data_path})

    print("Validating data...")
    validation_data(data_path={"train":train_data_path, "test":test_data_path})

