# 4_validate_model

# validate & analysis models
# 1. build face index & model
# 2. inference
# 3. analysis
#   save as png
#   - about distance
#   - about classes
#       - correct rate per classes

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
    
    return face_index, np.asarray(face_label)

def analysis(results):
    return 1

def main(args):
    image_files = glob.glob(os.path.join(args.test_data, "**/*.png"))
    total_labels = [int(i.split(os.sep)[-2]) for i in test_data]

    face_index, face_label = load_nn_model(args.faiss_model, args.faiss_label)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(args.embedding_model, map_location=device)

    results = {"predicts":[], "labels":total_labels, "distances":[]}

    for i in range(0, len(image_files), 16):
        batch_images = image_files[i: i+16]

        images = torch.from_numpy(np.asarray([
            cv2.resize(
                cv2.imread(batch_image),
                (args.image_width, args.image_height)
            )
            for batch_image in batch_images
        ])).to(device)
        
        embeddings = model(images)
        dists, inds = face_index.search(embeddings, 3)
        
        predicts = face_label[inds]

        results["predicts"] += predicts.tolist()
        results["distances"] += dists.tolist()
    
    print("finnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--test_data', type=str, default="../../data/faiss/test")

    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)

    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--embedding_model', type=str, default='../../model/model.pt')
    parser.add_argument('--faiss_model', type=str, default='../../model/faiss_index.bin')
    parser.add_argument('--faiss_label', type=str, default='../../model/label.json')

    parser.add_argument('--logfile', type=str, default='./log.log')

    args = parser.parse_args()

    main(args)
    