import os
import logging
import argparse
import ujson
import glob
import time

import cv2
import torch
import faiss

import pandas as pd
from sklearn.metrics import confusion_matrix

def load_nn_model(faiss_model_path, faiss_label_path):
    face_index = faiss.read_index(faiss_model_path)
    
    with open(faiss_label_path, "r") as l:
        face_label = ujson.load(l)
    
    return face_index, np.asarray(face_label)

def save_cm(results, num_classes):
    labels = [i for i in range(num_classes)]
    cm = confusion_matrix(
        results["labels"],
        results["predicts"],
        labels=labels
    )

    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((labels[target_index], labels[predicted_index], count))
    
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])

    output = '.'
    cm_file = os.path.join(output, 'confusion_matrix.csv')
    with open(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)
    
    lines = ''
    with open(cm_file, 'r') as f:
        lines = f.read()
    
    metadata = {
        'outputs': [{
                'type': 'confusion_matrix',
                'format': 'csv',
                'schema': [
                    {'name': 'target', 'type': 'CATEGORY'},
                    {'name': 'predicted', 'type': 'CATEGORY'},
                    {'name': 'count', 'type': 'NUMBER'},
                ],
                'source': lines,
                'storage': 'inline',
                'labels': list(map(str, labels)),
            }]
    }
    output_filename = "/cm.json"
    with open(output_filename, 'w') as f:
        ujson.dump(metadata, f)

    return output_filename


def main(args):
    image_files = glob.glob(os.path.join(args.test_data_path, "**/*.png"))
    total_labels = [int(i.split(os.sep)[-2]) for i in test_data]

    face_index, face_label = load_nn_model(
        os.path.join(args.model_dir, args.faiss_model_file),
        os.path.join(args.model_dir, args.faiss_label_file)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(os.path.join(args.model_dir, args.model_file), map_location=device)

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

        del images
        del embeddings

    start_time = time.time()
    batch_images = image_files[0]

    images = torch.from_numpy(np.asarray([
        cv2.resize(
            cv2.imread(batch_image),
            (args.image_width, args.image_height)
        )
        for batch_image in batch_images
    ])).to(device)
    
    embeddings = model(images)
    dists, inds = face_index.search(embeddings, 3)
    t = time.time() - start_time
    
    print("[+] Analysis using {} image data".format(str(len(total_labels))))
    
    print("[+] {} seconds per one image".format(str(t)))
    
    save_cm(results, args.class_nums)
    print(results)
    
    print("finnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--test_data_path', type=str, default="/data/mnist/test")

    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)

    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--model_dir', type=str, default='/model')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--faiss_model_file', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label_file', type=str, default='faiss_label.json')

    args = parser.parse_args()

    main(args)
    