import argparse
import torch
import numpy as np
from datasets import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default='dataset-sample', help="path to Dataset")
    parser.add_argument("--num_classes", type=int, default=None, help="num classes (default: None)")
    parser.add_argument("--dataset", type=str, default='dataset-sample', choices=['dataset-sample', 'dataset-medium'])
    
    parser.add_argument("--batch_size", type=int, default=16, help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=8, help='batch size for validation (default: 4)')
    parser.add_argument('--crop_size', type=int, default=128, help="size of the crop size  during transform")
    
    
    args = parser.parse_args()
    
    train_dst, val_dst = get_dataset(args)
    train_loader = data.DataLoader(train_dst, batch_size=args.batch_size, shuffle=True, num_workers=8)
    val_loader = data.DataLoader(val_dst, batch_size=args.val_batch_size, shuffle=True, num_workers=8)
    
    print("Train set: %d, Val set: %d" % (len(train_dst), len(val_dst)))