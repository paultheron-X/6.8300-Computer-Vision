import argparse
import time

def data_loading_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/processed')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    return parser.parse_args()