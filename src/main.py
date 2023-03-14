import logging
from torch.utils.data import DataLoader, random_split
import torch
import numpy as np

from utils.logging_config import logger_setup
from utils.arguments_parser import args_parser
from config import return_config

from data_handlers.loading import VideoDataset


def main(args):
    # set the seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    
    logging.info("Starting main training script")
    
    logging.info("Loading data")
    logging.debug(f"Creating dataset from path: {args.data_path}")
    
    dataset = VideoDataset(
        data_dir=args.data_path, rolling_window=args.rolling_window
    )
    
    logging.debug(f"Splitting 90/10 train/test") 
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    

    logging.debug(f"Creating train and test dataloaders")
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )
    
    #TODO: add model, optimizer, loss, training loop, validation loop, saving, loading, etc.
    

if __name__ == '__main__':
    args = args_parser()
    
    #TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    #ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    #config = return_config(args.config)
    
    #print(config['data_path'])
    
    logger_setup(args)
    
    main(args)