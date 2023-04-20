import logging
import coloredlogs


def logger_setup(args):
    level = "INFO" if not args.verbose else "DEBUG"
    config = dict(
        fmt="[[{relativeCreated:7,.0f}ms]] {levelname} [{module}] {message}",
        style="{",
        level=level,
    )
    coloredlogs.DEFAULT_LEVEL_STYLES["debug"] = {"color": 201}
    coloredlogs.DEFAULT_LEVEL_STYLES["warning"] = {
        "color": "red",
        "style": "bright",
        "bold": True,
        "italic": True,
    }
    coloredlogs.DEFAULT_FIELD_STYLES["levelname"] = {
        "color": "blue",
        "style": "bright",
        "bold": True,
    }
    coloredlogs.DEFAULT_FIELD_STYLES["relativeCreated"] = {
        "color": 10,
        "style": "bright",
    }

    coloredlogs.install(**config)
    
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel(logging.INFO)
