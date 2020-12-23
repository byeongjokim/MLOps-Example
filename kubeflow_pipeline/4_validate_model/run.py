# 4_validate_model

# training faiss with preprocessed data
# 1. build face index
# 2. fit faiss

import os
import logging
import argparse
import ujson

import torch
import faiss

def load_nn_model(model_path, label_path):
    face_index = faiss.read_index(model_path)
    
    with open(label_path, "r") as l:
        face_label = ujson.load(l)
    
    return face_index, face_label

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(args.embedding_model, map_location=device)


    

    return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--test_data', type=str, default="../../data/faiss/test")

    parser.add_argument('--save_dir', type=str, default='../../model/')    
    parser.add_argument('--embedding_model', type=str, default='model.pt')
    parser.add_argument('--faiss_model', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label', type=str, default='label.json')

    parser.add_argument('--logfile', type=str, default='./log.log')

    args = parser.parse_args()

    main(args)
    