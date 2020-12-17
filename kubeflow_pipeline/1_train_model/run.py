# 1_train_model

# training with preprocessed data
# 1. build model
# 2. build dataloader
# 3. train
# parameters for katlib
# - args.scale_size(scale size)
# - args.lr(learning rate)
# - args.batch_size(batch size)
# - args.d_embedding(dimension of embeddings)
# - args.optimizer

import os
import logging
import argparse

import torch
from torch.nn import DataParallel
from torch.optim import lr_scheduler

import numpy as np

from data import MnistDataset
from models import *

def train_embedding(args):
    logging.basicConfig(filename=args.logfile, level=logging.INFO, format='[+] %(asctime)s %(message)s', datefmt='%Y%m%d %I:%M:%S %p')

    train_dataset = MnistDataset(args.npy_path, num_classes=args.class_nums)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )

    eval_dataset = MnistDataset(args.npy_path_eval, num_classes=args.class_nums, transforms=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )

    model = Embedding(
        input_shape=(args.input_width, args.input_height),
        d_embedding=args.d_embedding
    )

    metric = ArcMarginProduct(args.d_embedding, args.class_nums, s=args.scale_size)
    
    if args.resume:
        model.load_state_dict(
            torch.load(args.model_path)["model_state_dict"]
        )
        metric.load_state_dict(
            torch.load(args.metric_path)["metric_state_dict"]
        )
    
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": metric.parameters()}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            [
                {"params": model.parameters()},
                {"params": metric.parameters()}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    # lr_sched = lr_scheduler.MultiStepLR(optimizer, gamma=0.1)

    if args.n_gpus > 1:
        net = DataParallel(net)
        metric = DataParallel(metric)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    metric.to(device)
    criterion.to(device)

    it = 0
    for epoch in range(1, args.epoch + 1):
        logging.info('{} epoch started'.format(str(epoch).zfill(3)))

        for data in train_dataloader:
            images, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            embeddings = model(images)
            output = metric(embeddings, labels)

            total_loss = criterion(output, labels)
            total_loss.backward()
            optimizer.step()

            if it % args.log_iter == 0:
                _, predict = torch.max(output.data, 1)
                correct = (np.array(predict.cpu()) == np.array(labels.data.cpu())).sum()
                now_accuracy = correct/labels.size(0)

                logging.info('{} iterations Accuracy :{}'.format(str(it).zfill(5), str(now_accuracy)))

            if it % args.save_iter == 0:
                
                if args.n_gpus > 1:
                    model_state_dict = model.module.state_dict()
                    metric_state_dict = metric.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    metric_state_dict = metric.state_dict()

                if not os.path.exists(args.save_dir):
                    os.mkdir(args.save_dir)

                torch.save(
                    {
                        "epoch": epoch,
                        "iters": it,
                        "model_state_dict": model_state_dict
                    },
                    os.path.join(args.save_dir, "iter_{}_model.ckpt".format(str(it).zfill(5)))
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "iters": it,
                        "metric_state_dict": metric_state_dict
                    },
                    os.path.join(args.save_dir, "iter_{}_metric.ckpt".format(str(it).zfill(5)))
                )
            
            if it % args.eval_iter == 0:
                model.eval()
                metric.eval()
                correct = 0
                with torch.no_grad():
                    for data in eval_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        
                        embeddings = model(images)
                        output = metric(embeddings, labels)

                        _, predict = torch.max(output.data, 1)
                        correct = correct + (np.array(predict.cpu()) == np.array(labels.data.cpu())).sum()
                    acc = correct / len(eval_dataloader.dataset)
                    logging.info('{} iterations Eval Accuracy :{}'.format(str(it).zfill(5), str(acc)))
                
                model.train()
                metric.train()

            it = it + 1

    with torch.no_grad():
        model.eval()
        metric.eval()

        if args.n_gpus > 1:
            model_ = torch.jit.script(model.module)
            metric_ = torch.jit.script(metric.module)
        else:
            model_ = torch.jit.script(model)
            metric_ = torch.jit.script(metric)

        torch.jit.save(
            model_,
            os.path.join(args.save_dir, args.save_model)
        )
        
        torch.jit.save(
            metric_,
            os.path.join(args.save_dir, args.save_metric)
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch for deep face recognition')
    
    parser.add_argument('--npy_path', type=str, default="../../data/mnist/train")
    parser.add_argument('--npy_path_eval', type=str, default="../../data/mnist/test")
    parser.add_argument('--input_width', type=int, default=28)
    parser.add_argument('--input_height', type=int, default=28)
    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--scale_size', type=int, default=32)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--save_dir', type=str, default='../../model/')
    parser.add_argument('--save_model', type=str, default='model.pt')
    parser.add_argument('--save_metric', type=str, default='metric.pt')

    parser.add_argument('--resume', type=int, default=False)
    parser.add_argument('--model_path', type=str, default='.')
    parser.add_argument('--metric_path', type=str, default='.')
    parser.add_argument('--logfile', type=str, default='./log.log')

    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--log_iter', type=int, default=500)
    parser.add_argument('--save_iter', type=int, default=500)
    parser.add_argument('--eval_iter', type=int, default=500)

    args = parser.parse_args()

    train_embedding(args)