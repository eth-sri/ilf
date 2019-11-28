import sys
import logging


def set_logging(verbosity=2, log_file=None):
    logger = logging.getLogger()

    if log_file is None:
        handler = logging.StreamHandler(sys.stdout)
    else:
        handler = logging.FileHandler(log_file)

    formatter = logging.Formatter('[%(asctime)s][%(created)f][%(name)s][%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level=[logging.NOTSET, logging.INFO, logging.DEBUG, logging.ERROR][verbosity])
