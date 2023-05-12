from configparser import ConfigParser, BasicInterpolation
from os.path import join
import os
import json
import logging


class OsEnvInterpolation(BasicInterpolation):
    """
    This class is used to interpolate environment variables into the config file.
    Those variable can have the following form: ${ENV_VAR_NAME}
    They were previously set in the bash file as follows: export ENV_VAR_NAME=value
    """

    def before_get(self, parser, section, option, value, defaults):
        return os.path.expandvars(value)


def get_config_parser(filename="config_gan.cfg"):
    config = ConfigParser(allow_no_value=True, interpolation=OsEnvInterpolation())
    config.read(os.getcwd() + "/src/config/" + filename)
    logging.info("Getting config file: " + os.getcwd() + "/src/config/" + filename)
    return config


def fill_config_with(
    config, config_parser, modifier, section, option
):  # handles if the option is not in the config file
    if config_parser.has_section(section) and config_parser.has_option(section, option):
        config[option.lower()] = modifier(config_parser.get(section, option))
    return config


def get_config(config_parser, config_name):
    config = {}

    # dataset
    fill_config_with(config, config_parser, str, "dataset", "DATA_PATH")
    if "GOOGLE_COLAB" in os.environ:
        config["data_path"] = config["data_path"].replace(
            "data", "/content/drive/MyDrive/data_vision"
        )
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
    fill_config_with(config, config_parser, int, "data", "SKIP_FRAMES")

    # model
    fill_config_with(config, config_parser, str, "model", "SPYNET_PRETRAINED")
    fill_config_with(config, config_parser, str, "model", "BASIC_VSR_PRETRAINED")
    fill_config_with(config, config_parser, str, "model", "OPTICAL_FLOW_MODULE")
    fill_config_with(config, config_parser, int, "model", "RESET_SPYNET")
    fill_config_with(config, config_parser, int, "model", "ATTENTION_HEADS")
    fill_config_with(config, config_parser, str, "model", "MSTAGE_VSR_PRETRAINED")

    # training
    fill_config_with(config, config_parser, int, "training", "EPOCHS")
    fill_config_with(config, config_parser, int, "training", "GRAD_ACCUM_STEPS")

    # result
    fill_config_with(config, config_parser, str, "result", "EXP_NAME")
    fill_config_with(config, config_parser, str, "result", "RESULT_DIR")

    config["result_dir"] = join(config["result_dir"], config["exp_name"])
    
    # copy past the config file in the result directory, as a cfg file
    if not os.path.exists(config["result_dir"]):
        os.makedirs(config["result_dir"])
    config_path =  os.getcwd() + "/src/config/" + config_name
    # copy paste
    with open(config_path, 'r') as f:
        config_file = f.read()

    config_name = config_name.replace("tests/", "")
    with open(join(config["result_dir"], config_name), 'w') as f:
        f.write(config_file)

    return config
