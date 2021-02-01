import os
import logging
import argparse
import ujson
import glob
import time
import requests

import torch
import faiss
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score

from dataset import MnistDataset

def load_nn_model(faiss_model_path, faiss_label_path):
    face_index = faiss.read_index(faiss_model_path)
    
    with open(faiss_label_path, "r") as l:
        face_label = ujson.load(l)
    
    return face_index, np.asarray(face_label)

def send_manage(accuracy):
    text = "Result of the Model!!!"
    text2 = "Accuracy for trained model is {}".format(str(accuracy))
    manage_url = os.getenv('MANAGE_URL')

    data = {"text": text, "text2": text2}
    try:
        requests.post(manage_url, data=data)
    except:
        pass

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

    cm_file = '/confusion_matrix.csv'
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

    with open("/mlpipeline-ui-metadata.json", 'w') as f:
        ujson.dump(metadata, f)

    accuracy = accuracy_score(results["labels"], results["predicts"])
    send_manage(accuracy)

    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue':  accuracy,
            'format': "PERCENTAGE",
        }]
    }

    with open('/accuracy.json', 'w') as f:
        ujson.dump(accuracy, f)
    with open('/mlpipeline-metrics.json', 'w') as f:
        ujson.dump(metrics, f)

def main(args):
    analysis_dataset = MnistDataset(args.test_data_path, num_classes=args.class_nums, shape=(args.image_width, args.image_height))
    analysis_dataloader = torch.utils.data.DataLoader(
        analysis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    face_index, face_label = load_nn_model(
        os.path.join(args.model_dir, args.faiss_model_file),
        os.path.join(args.model_dir, args.faiss_label_file)
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(os.path.join(args.model_dir, args.model_file), map_location=device)

    results = {"predicts":[], "labels":analysis_dataset.labels, "distances":[]}
    for i, data in enumerate(analysis_dataloader):
        images, _ = data
        
        with torch.no_grad():
            embeddings = model(images.to(device))
        
        dists, inds = face_index.search(embeddings.detach().cpu().numpy(), 3)
        
        predicts = face_label[inds[:,0]]

        results["predicts"] += predicts.tolist()
        results["distances"] += dists.tolist()

        del images
        del embeddings

    print("[+] Analysis using {} image data".format(str(len(analysis_dataset))))
    
    save_cm(results, args.class_nums)
    
    print("finnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--test_data_path', type=str, default="/data/mnist/test")

    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)

    parser.add_argument('--class_nums', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--model_dir', type=str, default='/model')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--faiss_model_file', type=str, default='faiss_index.bin')
    parser.add_argument('--faiss_label_file', type=str, default='faiss_label.json')

    args = parser.parse_args()

    main(args)
    