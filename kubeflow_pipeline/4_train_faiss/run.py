# 3_train_faiss

# training faiss with preprocessed data
# 1. build face index
# 2. fit faiss

import os
import logging
import argparse
import ujson
import glob

import numpy as np
import faiss

def save_model(face_index, labels, model_file_name, label_file_name):
    faiss.write_index(face_index, model_file_name)
    
    with open(label_file_name, "w") as l_f:
        ujson.dump(labels, l_f)

def parse_npy_files(npy_path):
    npy_embeddings_files = glob.glob(os.path.join(npy_path, "*embeddings*.npy"))
    npy_labels_files = glob.glob(os.path.join(npy_path, "*labels*.npy"))

    npy_embeddings_files.sort()
    npy_labels_files.sort()

    return npy_embeddings_files, npy_labels_files

def main(args):
    print("train faiss")
    npy_embeddings_files, npy_labels_files = parse_npy_files(args.npy_path)
    
    face_index = faiss.IndexFlatL2(args.d_embedding)

    total_labels = []
    for npy_embeddings_file, npy_labels_file in zip(npy_embeddings_files, npy_labels_files):
        embeddings = np.load(npy_embeddings_file)

        face_index.add(embeddings)

        labels = np.load(npy_labels_file).tolist()

        total_labels += labels

        del embeddings

    print("evaluate faiss")
    npy_embeddings_files, npy_labels_files = parse_npy_files(args.npy_path_eval)
    total_labels = np.asarray(total_labels)
    
    correct = 0
    l = 0
    for npy_embeddings_file, npy_labels_file in zip(npy_embeddings_files, npy_labels_files):
        embeddings = np.load(npy_embeddings_file)
        
        dists, inds = face_index.search(embeddings, 1)

        labels = np.load(npy_labels_file)

        predicts = total_labels[inds[:,0]]
        
        correct += (predicts == labels).sum()
        l += labels.size

    acc = correct/l
    print("Accuracy: " + str(acc))

    save_model(
        face_index,
        total_labels.tolist(),
        os.path.join(args.save_dir, args.faiss_model),
        os.path.join(args.save_dir, args.faiss_label)
    )
    print("Saved faiss model, faiss label")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--npy_path', type=str, default="/data/faiss/train")
    parser.add_argument('--npy_path_eval', type=str, default="/data/faiss/test")

    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--save_dir', type=str, default='/model')    
    parser.add_argument('--faiss_model', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label', type=str, default='label.json')

    # parser.add_argument('--logfile', type=str, default='./log.log')

    args = parser.parse_args()

    main(args)