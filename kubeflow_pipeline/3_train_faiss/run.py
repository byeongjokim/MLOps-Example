# 2_train_faiss

# training faiss with preprocessed data
# 1. build face index
# 2. fit faiss
# parameters for katlib
# - args.k(k of kNN)

import os
import logging
import argparse
import ujson

import torch

from data import MnistDataset

def fit(embeddings):
    face_index = faiss.IndexFlatL2(embeddings.shape[1])
    face_index.add(embeddings)
    
    return face_index

def save_model(face_index, labels, model_file_name, label_file_name):
    faiss.write_index(face_index, model_file_name)
    
    with open(label_file_name, "w") as l_f:
        ujson.dump(labels, l_f)

def load_npy(npy_path):
    npy_embeddings_files = glob.glob(os.path.join(npy_path, "*embeddings*.npy"))
    npy_labels_files = glob.glob(os.path.join(npy_path, "*labels*.npy"))

    npy_embeddings_files.sort()
    npy_labels_files.sort()

    return npy_embeddings, npy_labels

def train_faiss(args):
    npy_embeddings, npy_labels = load_npy(args.npy_path)

    face_index = faiss.IndexFlatL2(args.d_embedding)

    i = 0
    labels = {}
    while True:
        np.load()
        face_index.add()


    save_model(face_index, labels, FAISS_MODEL_FILE, LABEL_FILE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--npy_path', type=str, default="../../data/faiss/train")
    parser.add_argument('--npy_path_eval', type=str, default="../../data/faiss/test")
    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='../../model/')    
    parser.add_argument('--faiss_index_path', type=str, default='faiss_index.bin')
    parser.add_argument('--label_path', type=str, default='label.json')

    parser.add_argument('--logfile', type=str, default='./log.log')


    args = parser.parse_args()

    train_faiss(args)