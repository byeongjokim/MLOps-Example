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

import torch

from models import *

def train():

    # train_dataset = Dataset()
    # train_dataloader = DataLoader(train_dataset)

    model = Embedding(
        input_shape=args.input_shape,
        d_embedding=args.d_embedding
    )


    metric = ArcMarginProduct(args.d_embedding, train_dataset.class_nums, s=args.scale_size)
    
    # if args.resume:
    #     model.load_state_dict()
    #     metric.load_state_dict()
    
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

    # if args.n_gpus > 1:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    netric.to(device)
    criterion.to(device)

    logfile = args.logfile

    


if __name__ == "__main__":
    