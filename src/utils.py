import torch
import numpy as np
import random
import argparse
import os
import logging
import sys

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def parse_args():
    parser = argparse.ArgumentParser()
    # job configuration
    parser.add_argument("--mode", type=str, default="pretrain", choices=["pretrain", "finetune"])
    parser.add_argument("--downstream_task", type=str, default="diff", choices=["diff", "similarity", "cluster"])
    parser.add_argument("--n_cluster", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--save_step", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=114)
    parser.add_argument("--out_path", type=str, default="../output")
    parser.add_argument("--pretrain_path", type=str, default="../output/pretrain/checkpoint_epoch_29")
    parser.add_argument("--log_path", type=str, default="../log")
    args = parser.parse_args()
    return args

def setuplogger(args, out_path):
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)

    fh = logging.FileHandler(os.path.join(out_path, args.mode + ".log"))
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    root.addHandler(fh)