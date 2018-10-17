import logging

logger = logging.getLogger(name="pythons")
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                  datefmt='%H:%M:%S'))
logger.addHandler(ch)
