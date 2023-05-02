import logging
import coloredlogs


def logger_setup(args):
    level = "INFO" if not args.verbose else "DEBUG"
    config = dict(fmt="[[{relativeCreated:7,.0f}ms]] {levelname} [{module}] {message}", style="{", level=level)

    # fh = logging.FileHandler('tests.txt')
    # fh.setLevel(logging.DEBUG)

    coloredlogs.DEFAULT_LEVEL_STYLES["debug"] = {"color": 201}
    coloredlogs.DEFAULT_LEVEL_STYLES["warning"] = {"color": "red", "style": "bright", "bold": True, "italic": True}
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {"color": "blue", "style": "bright", "bold": True}
    coloredlogs.DEFAULT_FIELD_STYLES["relativeCreated"] = {"color": 10, "style": "bright"}
    coloredlogs.DEFAULT_FIELD_STYLES["module"] = {"color": "yellow", "bold": True}

    coloredlogs.install(**config)

    # logger = logging.getLogger()
    # logger.addHandler(fh)

    pil_logger = logging.getLogger("PIL")
    pil_logger.setLevel(logging.INFO)
    
    torch_logger = logging.getLogger("torch")
    # remove the output of all torch logger
    torch_logger.setLevel(logging.ERROR)
