import logging
import torch
import numpy as np

from utils.logging_config import logger_setup
from utils.arguments_parser import args_parser
from config import return_config


def main(config):
    # set the seeds
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    logging.info("Starting main training script")
    
    logging.info("Loading data")
    logging.debug(f"Creating dataset from path: {config['data_path']}")
    
    

if __name__ == '__main__':
    args = args_parser()
    
    #TODO: add config file and merge the args and the config: rolling_window ... things like that should be config
    #ARGS: verbose, path to config, saving path, pretrained path, saving or not ..  things that could be true or false for the same config
    config = return_config(args.config)
    
    logger_setup(args)
    
    config.update(vars(args))

    main(config)