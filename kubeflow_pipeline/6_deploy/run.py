import os
import logging
import argparse
import ujson
import glob
import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--model_file', type=str, default="/data/mnist/test")

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
    