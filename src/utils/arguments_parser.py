import argparse
import time

# TODO: change the layout of this file: decide what should go in a config file and what shoud go in a parser

def data_loading_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/processed")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--rolling_window", type=int, default=3)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument(
        "--verbose", "-v", help="Sets the lebel of verbose", action="store_true"
    )

    return parser.parse_args()

def model_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--pretrained", type=bool, default=True)
    parser.add_argument("--n_channels", type=int, default=3)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    return parser.parse_args()

def args_parser():
    # concats the results of data_loading_parser and model_parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v", help="Sets the lebel of verbose", action="store_true"
    )
    parser.add_argument("--config", type=str, default="config_base.cfg")
    parser.add_argument("--seed", "-s", type=int, default=42)
    return parser.parse_args()

    
