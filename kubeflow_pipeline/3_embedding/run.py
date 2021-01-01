# 3_embedding

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
        
        with torch.no_grad():
            embeddings = model(images)

        total_embeddings = np.append(total_embeddings, embeddings.detach().numpy(), axis=0)

        if len(total_embeddings) == npy_interval:
            np.save(
                os.path.join(
                    npy_path,
                    filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"
                ),
                total_embeddings.astype(np.float32)
            )
            print("{} saved ".format(filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"))
            
            npy_num = npy_num + 1
            total_embeddings = np.empty((0, d_embedding))
    
    if total_embeddings.size:
        np.save(
            os.path.join(
                npy_path,
                filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"
            ),
            total_embeddings.astype(np.float32)
        )
        print("{} saved ".format(filename + "_embeddings_" + str(npy_num).zfill(3) + ".npy"))
    
    del total_embeddings

def main(args):
    train_dataset = EmbedDataset(args.faiss_train_data_path)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    print("[+] " + str(len(train_dataset)) + " train dataset")

    eval_dataset = EmbedDataset(args.faiss_test_data_path)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    print("[+] " + str(len(eval_dataset)) + " evaluate dataset")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.jit.load(os.path.join(args.model_dir, args.model_file), map_location=device)
    print(model)

    print("[+] Start to save embedding train dataset")
    inference_and_save(
        dataloader=train_dataloader,
        model=model,
        npy_interval=args.npy_interval,
        npy_path=args.faiss_train_data_path,
        filename=args.faiss_train_data_file,
        d_embedding=args.d_embedding,
        device=device
    )

    # logging.info("Start to save eval dataset")
    print("[+] Start to save embedding eval dataset")
    inference_and_save(
        dataloader=eval_dataloader,
        model=model,
        npy_interval=args.npy_interval,
        npy_path=args.faiss_test_data_path,
        filename=args.faiss_test_data_file,
        d_embedding=args.d_embedding,
        device=device
    )

    del train_dataset
    del eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--model_dir', type=str, default='/model/')
    parser.add_argument('--model_file', type=str, default='model.pt')
    
    parser.add_argument('--faiss_train_data_path', type=str, default="/data/faiss/train")
    parser.add_argument('--faiss_test_data_path', type=str, default="/data/faiss/test")
    parser.add_argument('--faiss_train_data_file', type=str, default="faiss_train")
    parser.add_argument('--faiss_test_data_file', type=str, default="faiss_test")

    parser.add_argument('--d_embedding', type=int, default=128)

    parser.add_argument('--npy_interval', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=10)

    parser.add_argument('--num_workers', type=int, default=2)

    args = parser.parse_args()

    main(args)