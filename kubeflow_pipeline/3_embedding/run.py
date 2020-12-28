# 2_embedding

# embedding with preprocessed data
# 1. build model
# 2. build dataloader
# 3. save embedding

import os
import logging
import argparse

import torch
import numpy as np

from dataset import EmbedDataset

def inference_and_save(dataloader, model, npy_interval, npy_path, filename, d_embedding, device):
    npy_num = 0
    total_embeddings = np.empty((0, d_embedding))
    for data in dataloader:
        images = data.to(device)
        
        embeddings = model(images)

        total_embeddings = np.append(total_embeddings, embeddings.detach().numpy(), axis=0)

        if len(total_embeddings) == npy_interval:
            np.save(
                os.path.join(
                    npy_path,
                    filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"
                ),
                total_embeddings
            )
            npy_num = npy_num + 1
            total_embeddings = np.empty((0, d_embedding))
    
    if total_embeddings.size:
        np.save(
            os.path.join(
                npy_path,
                filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"
            ),
            total_embeddings
        )
    
    del total_embeddings

def main(args):
    # logging.basicConfig(filename=args.logfile, level=logging.INFO, format='[+] %(asctime)s %(message)s', datefmt='%Y%m%d %I:%M:%S %p')

    train_dataset = EmbedDataset(args.npy_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    eval_dataset = EmbedDataset(args.npy_path_eval)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(args.embedding_model_path, map_location=device)

    # logging.info("Start to save train dataset")
    print("Start to save train dataset")
    inference_and_save(
        dataloader=train_dataloader,
        model=model,
        npy_interval=args.npy_interval,
        npy_path=args.npy_path,
        filename="faiss_train",
        d_embedding=args.d_embedding,
        device=device
    )

    # logging.info("Start to save eval dataset")
    print("Start to save eval dataset")
    inference_and_save(
        dataloader=eval_dataloader,
        model=model,
        npy_interval=args.npy_interval,
        npy_path=args.npy_path_eval,
        filename="faiss_test",
        d_embedding=args.d_embedding,
        device=device
    )

    del train_dataset
    del eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--embedding_model_path', type=str, default="/model/model.pt")
    
    parser.add_argument('--npy_path', type=str, default="/data/faiss/train")
    parser.add_argument('--npy_path_eval', type=str, default="/data/faiss/test")
    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--npy_interval', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    # parser.add_argument('--logfile', type=str, default='./log.log')

    args = parser.parse_args()

    main(args)