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

def basic_vsr_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_dir', default='./REDS/train_sharp')
    parser.add_argument('--lq_dir', default='./REDS/train_sharp_bicubic/X4')
    parser.add_argument('--log_dir', default='./log_dir')
    parser.add_argument('--spynet_pretrained', default='spynet_20210409-c6c1bd09.pth')
    parser.add_argument('--scale_factor', default=4,type=int)
    parser.add_argument('--batch_size', default=8,type=int)
    parser.add_argument('--patch_size', default=64,type=int)
    parser.add_argument('--epochs', default=300000,type=int)
    parser.add_argument('--num_input_frames', default=15,type=int)
    parser.add_argument('--val_interval', default=1000,type=int)
    parser.add_argument('--max_keys', default=270,type=int)
    parser.add_argument('--filename_tmpl', default='{:08d}.png')
    args = parser.parse_args()

def args_parser():
    # concats the results of data_loading_parser and model_parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verbose", "-v", help="Sets the lebel of verbose", action="store_true"
    )
    parser.add_argument("--config", type=str, default="config_base.cfg")
    parser.add_argument("--seed", "-s", type=int, default=42)
    return parser.parse_args()

    
