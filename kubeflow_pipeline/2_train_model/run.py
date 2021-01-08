# 2_train_model

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

from dataset import MnistDataset
from models import *

def main(args):
    train_dataset = MnistDataset(args.train_data_path, num_classes=args.class_nums)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False
    )
    print("[+] " + str(len(train_dataset)) + " train dataset")

    eval_dataset = MnistDataset(args.test_data_path, num_classes=args.class_nums, transforms=False)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    print("[+] " + str(len(eval_dataset)) + " evaluate dataset")

    model = Embedding(
        input_shape=(args.image_width, args.image_height),
        input_channel=args.image_channel,
        d_embedding=args.d_embedding
    )
    print(model)

    metric = ArcMarginProduct(args.d_embedding, args.class_nums, s=args.scale_size)
    print(metric)
    
    if args.resume:
        print("[+] Resume the model({}) and metric({})".format(args.model_resume, args.metric_resume))
        model.load_state_dict(
            torch.load(args.model_resume)["model_state_dict"]
        )
        metric.load_state_dict(
            torch.load(args.metric_resume)["metric_state_dict"]
        )
    
    criterion = torch.nn.CrossEntropyLoss()

    if args.optimizer == "adam":
        print("[+] Using Adam Optimizer")
        optimizer = torch.optim.Adam(
            [
                {"params": model.parameters()},
                {"params": metric.parameters()}
            ],
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        print("[+] Using SGD Optimizer")
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
        # logging.info('{} epoch started'.format(str(epoch).zfill(3)))
        print('[+] {} epoch started'.format(str(epoch).zfill(3)))

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

                # logging.info('{} iterations Accuracy :{}'.format(str(it).zfill(5), str(now_accuracy)))
                print('[+] {} iterations Accuracy :{}'.format(str(it).zfill(5), str(now_accuracy)))

            if it % args.save_iter == 0:
                
                if args.n_gpus > 1:
                    model_state_dict = model.module.state_dict()
                    metric_state_dict = metric.module.state_dict()
                else:
                    model_state_dict = model.state_dict()
                    metric_state_dict = metric.state_dict()

                if not os.path.exists(args.ckpt_dir):
                    os.makedirs(args.ckpt_dir)

                torch.save(
                    {
                        "epoch": epoch,
                        "iters": it,
                        "model_state_dict": model_state_dict
                    },
                    os.path.join(args.ckpt_dir, "iter_{}_".format(str(it).zfill(5)) + args.model_ckpt)
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "iters": it,
                        "metric_state_dict": metric_state_dict
                    },
                    os.path.join(args.ckpt_dir, "iter_{}_".format(str(it).zfill(5)) + args.metric_ckpt)
                )
                print('[+] {} iterations model saved'.format(str(it).zfill(5)))
            
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
                    # logging.info('{} iterations Eval Accuracy :{}'.format(str(it).zfill(5), str(acc)))
                    print('[+] {} iterations Eval Accuracy :{}'.format(str(it).zfill(5), str(acc)))
                
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

        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

        torch.jit.save(
            model_,
            os.path.join(args.model_dir, args.model_file)
        )
        
        torch.jit.save(
            metric_,
            os.path.join(args.model_dir, args.metric_file)
        )
        
        print("[+] Saved final Models")

    del train_dataset
    del eval_dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
        
    parser.add_argument('--train_data_path', type=str, default="/data/mnist/train")
    parser.add_argument('--test_data_path', type=str, default="/data/mnist/test")

    parser.add_argument('--image_width', type=int, default=28)
    parser.add_argument('--image_height', type=int, default=28)
    parser.add_argument('--image_channel', type=int, default=1)

    parser.add_argument('--d_embedding', type=int, default=128)
    parser.add_argument('--scale_size', type=int, default=32)
    parser.add_argument('--class_nums', type=int, default=10)

    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    
    parser.add_argument('--n_gpus', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=2)

    parser.add_argument('--model_dir', type=str, default='/model/')
    parser.add_argument('--model_file', type=str, default='model.pt')
    parser.add_argument('--metric_file', type=str, default='metric.pt')

    parser.add_argument('--ckpt_dir', type=str, default='/model/ckpt/')
    parser.add_argument('--model_ckpt', type=str, default='model.ckpt')
    parser.add_argument('--metric_ckpt', type=str, default='metric.ckpt')

    parser.add_argument('--resume', type=int, default=False)
    parser.add_argument('--model_resume', type=str, default='.')
    parser.add_argument('--metric_resume', type=str, default='.')

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=2000)
    parser.add_argument('--eval_iter', type=int, default=1000)

    args = parser.parse_args()
    
    print("Start Training")
    main(args)