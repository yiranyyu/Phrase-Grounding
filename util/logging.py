import logging

FMT = "[%(asctime)s] %(levelname)s: %(message)s"
DATEFMT = "%m/%d/%Y %H:%M:%S"


def basicConfig(**kwargs):
    level = kwargs.pop('level', logging.INFO)
    format = kwargs.pop('format', FMT)
    datefmt = kwargs.pop('datefmt', DATEFMT)
    logging.getLogger().handlers.clear()
    logging.basicConfig(level=level, format=format, datefmt=datefmt, **kwargs)
