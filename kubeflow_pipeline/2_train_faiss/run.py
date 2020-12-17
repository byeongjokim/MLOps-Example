#

import os
import logging
import argparse
import ujson

import torch

from data import MnistDataset

# def load_nn_model(faiss_model_file, label_file):
#     face_index = faiss.read_index(faiss_model_file)
#     with open(label_file, "r") as l_f:
#         face_label = ujson.load(l_f)
#     return face_index, face_label    
# face_index, face_label = load_nn_model(args.faiss_model_file, args.label_file)

def fit(embeddings):
    face_index = faiss.IndexFlatL2(embeddings.shape[1])
    face_index.add(embeddings)
    
    return face_index

def save_model(face_index, labels, model_file_name, label_file_name):
    faiss.write_index(face_index, model_file_name)
    
    with open(label_file_name, "w") as l_f:
        ujson.dump(labels, l_f)

def load_npy(npy_path):
    npy_images_files = glob.glob(os.path.join(npy_path, "*images*.npy"))
    npy_labels_files = glob.glob(os.path.join(npy_path, "*labels*.npy"))

    image_memmaps = [np.load(npy_images_file, mmap_mode="r") for npy_images_file in npy_images_files]

    total_memmaps = 


def train_faiss(args):
    model = torch.jit.load(args.embedding_model_path)


    # face_index = fit(embeddings)
    # save_model(face_index, labels, FAISS_MODEL_FILE, LABEL_FILE)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--npy_path', type=str, default="../../data/faiss/train")
    parser.add_argument('--npy_path_eval', type=str, default="../../data/faiss/test")
    parser.add_argument('--input_width', type=int, default=28)
    parser.add_argument('--input_height', type=int, default=28)
    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--embedding_model_path', type=str, default='../../model/model.pt')

    parser.add_argument('--save_dir', type=str, default='../../model/')    
    parser.add_argument('--faiss_index_path', type=str, default='faiss_index.bin')
    parser.add_argument('--label_path', type=str, default='label.json')

    parser.add_argument('--logfile', type=str, default='./log.log')


    args = parser.parse_args()

    train_faiss(args)