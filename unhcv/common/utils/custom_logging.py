import logging

from accelerate.logging import MultiProcessAdapter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def get_logger(name: str, log_level: str = None, accelerate_mode=False):
    logger = logging.getLogger(name.split(".")[-1])
    if log_level is not None:
        logger.setLevel(log_level.upper())
        logger.root.setLevel(log_level.upper())
    if accelerate_mode:
        logger = MultiProcessAdapter(logger, {})
    return logger


if __name__ == "__main__":
    logger = get_logger("abc")
    logger.info("test")