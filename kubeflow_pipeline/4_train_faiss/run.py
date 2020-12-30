# 4_train_faiss

# training faiss with preprocessed data
# 1. build face index
# 2. fit faiss
# 3. evaluate faiss

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
    print("[+] Start to train faiss")
    npy_embeddings_files, npy_labels_files = parse_npy_files(args.faiss_train_data_path)
    print("[+] " + str(len(npy_labels_files)) + " train npy")

    face_index = faiss.IndexFlatL2(args.d_embedding)

    total_labels = []
    for npy_embeddings_file, npy_labels_file in zip(npy_embeddings_files, npy_labels_files):
        embeddings = np.load(npy_embeddings_file)

        face_index.add(embeddings)

        labels = np.load(npy_labels_file).tolist()

        total_labels += labels

        del embeddings
    print("[+] Trained with {} images".format(str(len(total_labels))))

    print("[+] Start to evaluate faiss")
    npy_embeddings_files, npy_labels_files = parse_npy_files(args.faiss_test_data_path)
    print("[+] " + str(len(npy_labels_files)) + " evaluation npy")

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

        del embeddings

    acc = correct/l
    print("[+] Evaluated with {} images".format(str(l)))
    print("[+] Accuracy for evaluate set: " + str(acc*100) + "%")

    faiss_model_file_path = os.path.join(args.model_dir, args.faiss_model_file)
    faiss_label_file_path = os.path.join(args.model_dir, args.faiss_label_file)
    save_model(
        face_index,
        total_labels.tolist(),
        faiss_model_file_path,
        faiss_label_file_path
    )
    print("[+] Saved faiss model({}), faiss label({})".format(faiss_model_file_path, faiss_label_file_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--faiss_train_data_path', type=str, default="/data/faiss/train")
    parser.add_argument('--faiss_test_data_path', type=str, default="/data/faiss/test")

    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--model_dir', type=str, default='/model')    
    parser.add_argument('--faiss_model_file', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label_file', type=str, default='faiss_label.json')

    args = parser.parse_args()

    main(args)