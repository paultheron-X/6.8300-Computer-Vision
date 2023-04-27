from configparser import ConfigParser, BasicInterpolation
from os.path import join
import os
import json
import logging

class OsEnvInterpolation(BasicInterpolation):
    '''
    This class is used to interpolate environment variables into the config file.
    Those variable can have the following form: ${ENV_VAR_NAME}
    They were previously set in the bash file as follows: export ENV_VAR_NAME=value
    '''
    def before_get(self, parser, section, option, value, defaults):
        return os.path.expandvars(value)


def get_config_parser(filename="config_gan.cfg"):
    config = ConfigParser(allow_no_value=True, interpolation=OsEnvInterpolation())
    config.read(os.getcwd() + "/src/config/" + filename)
    logging.info("Getting config file: " + os.getcwd() + "/src/config/" + filename)
    return config


def fill_config_with(config, config_parser, modifier, section, option):  # handles if the option is not in the config file
    if config_parser.has_section(section) and config_parser.has_option(section, option):
        config[option.lower()] = modifier(config_parser.get(section, option))
    return config


def get_config(config_parser):
    config = {}

    # dataset
    fill_config_with(config, config_parser, str, "dataset", "DATA_PATH")
    if "GOOGLE_COLAB" in os.environ:
        config["data_path"] = config["data_path"].replace("data", "/content/drive/MyDrive/data_vision")
    fill_config_with(config, config_parser, str, "dataset", "LR_DATA_DIR")
    fill_config_with(config, config_parser, str, "dataset", "HR_DATA_DIR")
    fill_config_with(config, config_parser, int, "dataset", "PREPARE_DATA")

    # data
    fill_config_with(config, config_parser, int, "data", "ROLLING_WINDOW")
    fill_config_with(config, config_parser, int, "data", "NUM_INPUT_FRAMES")
    fill_config_with(config, config_parser, int, "data", "BATCH_SIZE")
    fill_config_with(config, config_parser, int, "data", "PATCH_SIZE")

    fill_config_with(config, config_parser, int, "data", "SCALE_FACTOR")
    fill_config_with(config, config_parser, int, "data", "VAL_INTERVAL")
    fill_config_with(config, config_parser, int, "data", "MAX_KEYS")

    # model
    fill_config_with(config, config_parser, str, "model", "SPYNET_PRETRAINED")
    fill_config_with(config, config_parser, str, "model", "BASIC_VSR_PRETRAINED")
    fill_config_with(config, config_parser, int, "model", "RESET_SPYNET")

    # training
    fill_config_with(config, config_parser, str, "training", "LOG_DIR")
    fill_config_with(config, config_parser, int, "training", "EPOCHS")
    
    # result
    fill_config_with(config, config_parser, str, "result", "EXP_NAME")
    fill_config_with(config, config_parser, str, "result", "RESULT_DIR")
    
    config['result_dir'] = join(config['result_dir'], config['exp_name'])
    return config
