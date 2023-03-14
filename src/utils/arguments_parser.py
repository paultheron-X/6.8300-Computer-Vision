import argparse
import time


def data_loading_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--vid_window", type=int, default=3)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument(
        "--verbose", "-v", help="Sets the lebel of verbose", action="store_true"
    )

    return parser.parse_args()
